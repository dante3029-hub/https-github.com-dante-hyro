# The orchestrator edit — exact before and after

Two blocks in `bot/orchestrator.py::run_cycle`. Everything above (compliance,
equity marking, risk overlay, kill switch) and everything below (cadence
gating, `size_portfolio`, execution) is untouched.

---

## Block 1 — the snapshot call

### BEFORE

```python
        # 2. pull market snapshot (signals + event-sleeve slot updates)
        snap: MarketSnapshot = self.data_feed.get_snapshot(
            as_of_date, self.state.short_tracker_state, self.state.bos_tracker_state
        )
        report["data_source"] = snap.data_source
```

### AFTER

```python
        # 2. pull market snapshot.
        #
        # TWO SOURCES, deliberately:
        #   - the existing engine still computes delta and relvol, unchanged
        #   - book_v2 supplies the seven new sleeves
        #
        # Splitting them means the delta/relvol path that has been running for
        # months is not disturbed by this change, and a failure in the new book
        # cannot silently take out the old one.
        snap: MarketSnapshot = self.data_feed.get_snapshot(
            as_of_date, self.state.short_tracker_state, self.state.bos_tracker_state
        )
        report["data_source"] = snap.data_source

        # the new book. LiveDataFeed RAISES on stale data rather than returning
        # weights computed on old prices -- that is the point of it, so this is
        # NOT wrapped in a bare except.
        from bot.live_feed import LiveDataFeed, StaleDataError
        try:
            live = LiveDataFeed()
            live_snap = live.get_snapshot(self.state.tracker_states)
            book_weights = live_snap.weights
            book_stops = live_snap.stops
            report["book_data_age_h"] = round(live_snap.data_age_hours, 2)
        except StaleDataError as e:
            # a stale feed is a HARD STOP for the new book. delta/relvol
            # continue on their own data; the seven new sleeves go flat and the
            # health check will say so loudly.
            report["flags"].append(f"NEW BOOK HALTED -- stale data: {e}")
            logger.error("new book halted on stale data: %s", e)
            book_weights = {s: {} for s in
                            ("skew", "oirank", "cascade", "sr", "srflip",
                             "pattern", "fvg")}
            book_stops = {}
```

---

## Block 2 — assembling the weights

### BEFORE

```python
        raw_weights_this_cycle = {
            "main": snap.signal_snapshot.main_weights,
            "flow": snap.signal_snapshot.flow_weights,
            "delta": snap.signal_snapshot.delta_weights,
            "relvol": snap.signal_snapshot.relvol_weights,
            "short": snap.short_weights,
            "bos": snap.bos_weights,
        }
```

### AFTER

```python
        raw_weights_this_cycle = {
            # kept from the old book
            "delta": snap.signal_snapshot.delta_weights,
            "relvol": snap.signal_snapshot.relvol_weights,
            # the new book
            **book_weights,
        }
        # main, flow, short and bos are GONE. Not commented out -- removed.
        #
        #   main/flow  never earned a place in the validated book
        #   short      superseded by the event sleeves
        #   bos        0.33 at 4h against a random-direction control of 0.74 --
        #              the one setting where random beats it. It could not fire
        #              at all until the taker-data fix; after that fix it can,
        #              which makes leaving it enabled actively harmful.
```

---

## Block 3 — protective stops

Find the existing `sync_protective_stops` call and replace its stop dict:

```python
        # stops live ON THE EXCHANGE, not in the bot. If the process dies, the
        # stop still works. book_stops is {coin: price}; convert to symbols.
        from bot.exchange_client import to_bybit_symbol
        sync_protective_stops(
            self.exchange_client,
            {to_bybit_symbol(c): px for c, px in book_stops.items()},
            dry_run=self.dry_run,
        )
```

---

## Block 4 — the health gate, at the END of the cycle, before orders

```python
        # ── HEALTH GATE ──
        from sleeve_health import (run_health_check, format_report, CRITICAL,
                                   book_expectations)
        from bot.book_v2 import _taker_dir, _oi_dir

        findings = run_health_check(
            book_expectations(_taker_dir(), _oi_dir()),
            weights_by_sleeve=raw_weights_this_cycle,
            flat_cycles=self.state.flat_cycles,
            hours_since_rebalance={
                s: _hours_since(self.state.last_rebalance.get(s), now)
                for s in raw_weights_this_cycle
            },
            sanity_files={"delta": f"{_taker_dir()}/SOL_1h.csv"},
        )
        logger.info(format_report(findings))
        report["health"] = [str(f) for f in findings if f.severity != "INFO"]

        if any(f.severity == CRITICAL for f in findings) and config.HALT_ON_CRITICAL:
            report["halted"] = True
            report["flags"].append("HALTED on CRITICAL health findings -- no orders placed")
            # alert BEFORE returning, or a halted bot is a silent bot
            if self.alerter:
                self.alerter.from_findings(findings)
            return report        # <-- NO ORDERS

        if self.alerter:
            self.alerter.from_findings(findings)
            self.alerter.cycle(
                equity=equity_mark,
                daily_pnl=running_session_pnl_dollars,
                opened=report.get("opened", []),
                closed=report.get("closed", []),
                gross=report.get("gross", 0.0),
                vol_target=config.daily_vol_target(equity_mark),
            )
        self.state.last_cycle_ts = time.time()    # the watchdog reads this
```

---

## State additions

`bot/state.py` needs three new fields, all JSON-serialisable:

```python
    tracker_states: dict = field(default_factory=lambda: {
        s: {"slots": {}} for s in ("sr", "srflip", "pattern", "fvg")})
    flat_cycles: dict = field(default_factory=dict)
    last_cycle_ts: float = 0.0
```

`short_tracker_state` and `bos_tracker_state` can stay for now -- the old feed
still takes them, and removing them is a separate cleanup.

---

## Why the gate returns BEFORE placing orders

BOS produced zero weights for 300+ live cycles while the log read
`rebalanced=['short','bos']`. `run/` went 408 hours stale while cycles kept
"succeeding". The taker column had `taker_buy > volume` on 60% of bars. The OI
files had a column misalignment that would have broken every OI-gated sleeve.
`oirank` went flat whenever OI lagged price by one bar.

**Five silent failures. Every one ran clean and logged success.**

A sleeve that cannot be verified does not trade.

---

## Then, in order

```bash
python3 doctor.py                        # must exit 0
python3 bot/burn_in.py --mode offline --cycles 12
BYBIT_API_KEY=... BYBIT_API_SECRET=... \
  python3 bot/burn_in.py --mode demo --cycles 12
```

Only then mainnet, at $300/day, with `DISCORD_WEBHOOK_BOT` set and the
watchdog cron running.
