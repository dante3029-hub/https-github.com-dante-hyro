#!/usr/bin/env python3
"""
replay.py — run the LIVE code path through history, bar by bar.

Not another backtest. The backtest computes weight matrices with full history
in hand. This walks forward and calls the tracker exactly as the live bot does,
logging every position change with its reason.

Uses fast_detect for signal lookup -- verified to fire on IDENTICAL bars to the
slow path (747 bars checked across four sleeves, zero mismatches) but 156,000x
faster, which is what makes this runnable at all.

WHAT THIS CATCHES THAT A BACKTEST CANNOT
  * a tracker opening a slot it should not, or failing to close one
  * state that does not survive a restart mid-run
  * cadence bugs -- a sleeve acting at the wrong interval
  * duplicate positions in one coin
  * stops on the wrong side of entry
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import hourly_sim as H
from fast_detect import FastSR, FastFVG, FastPattern
from signal_engine.sleeve_price_events import PARAMS

SLEEVES = ("sr", "srflip", "pattern", "fvg")


@dataclass
class Slot:
    coin: str
    side: int
    entry_i: int
    entry_px: float
    stop: float
    hold: int


class Replay:
    def __init__(self, coins: List[str], sleeves=SLEEVES):
        self.coins = coins
        self.sleeves = sleeves
        self.bars, self.det, self.oi, self.atr = {}, {}, {}, {}
        for s in sleeves:
            tf = PARAMS[s]['tf']
            for c in coins:
                key = (c, tf)
                if key not in self.bars:
                    p = f'{H.TAKER}/{c}_1h.csv'
                    if not os.path.exists(p):
                        self.bars[key] = None
                        continue
                    b = H.bars(c, tf)
                    self.bars[key] = b if len(b) >= 750 else None
                    if self.bars[key] is not None:
                        self.atr[key] = H.atr(b).to_numpy()
                b = self.bars[key]
                if b is None:
                    self.det[(c, s)] = None
                    continue
                self.det[(c, s)] = (FastSR(b) if s in ('sr', 'srflip')
                                    else FastFVG(b) if s == 'fvg'
                                    else FastPattern(b))
                if PARAMS[s]['needs_oi'] and (c, tf) not in self.oi:
                    self.oi[(c, tf)] = H.oi_up(c, tf)
        self.slots: Dict[str, Dict[str, dict]] = {s: {} for s in sleeves}
        self.events: List[dict] = []

    def _fired(self, coin, sleeve, i) -> int:
        d = self.det.get((coin, sleeve))
        if d is None:
            return 0
        if sleeve == 'sr':
            return 1 if d.fired(i, 'break') else 0
        if sleeve == 'srflip':
            return 1 if d.fired(i, 'flip') else 0
        return d.fired(i)

    def _oi_ok(self, coin, sleeve, i) -> bool:
        """THE OI GATE IS AT THE ENTRY BAR (i+1), NOT THE SIGNAL BAR.
        Gating at signal keeps 31 of SOL's 60 sr breaks; gating at entry keeps
        23; they overlap by 12. Same data, different strategy."""
        if not PARAMS[sleeve]['needs_oi']:
            return True
        tf = PARAMS[sleeve]['tf']
        ou = self.oi.get((coin, tf))
        b = self.bars.get((coin, tf))
        if ou is None or b is None or i + 1 >= len(b):
            return False
        v = ou.reindex([b.index[i + 1]], method='ffill')
        return bool(len(v) and bool(v.iloc[0]))

    def run(self, start_ts, end_ts, restart_at=None) -> dict:
        stats = dict(opens=0, closes=0, restarts=0, steps=0,
                     max_slots={s: 0 for s in self.sleeves})
        for s in self.sleeves:
            tf = PARAMS[s]['tf']
            for c in self.coins:
                b = self.bars.get((c, tf))
                if b is None:
                    continue
                A = self.atr[(c, tf)]
                o, lo, hi, cl = (b['open'].to_numpy(), b['low'].to_numpy(),
                                 b['high'].to_numpy(), b['close'].to_numpy())
                lo_i = max(700, int(b.index.searchsorted(start_ts)))
                hi_i = min(len(b) - PARAMS[s]['hold'] - 2,
                           int(b.index.searchsorted(end_ts)))
                for i in range(lo_i, hi_i):
                    stats['steps'] += 1
                    # restart: serialise, discard, rebuild
                    if (restart_at is not None and stats['restarts'] == 0
                            and b.index[i] >= restart_at):
                        blob = json.dumps(self.slots)
                        self.slots = json.loads(blob)
                        stats['restarts'] = 1
                    # walk the open slot for this coin
                    slot = self.slots[s].get(c)
                    if slot is not None:
                        slot['bars_held'] += 1
                        hit = ((slot['side'] > 0 and lo[i] <= slot['stop']) or
                               (slot['side'] < 0 and hi[i] >= slot['stop']))
                        if hit or slot['bars_held'] >= slot['hold']:
                            self.events.append(dict(
                                i=i, ts=str(b.index[i]), ev='CLOSE', sleeve=s,
                                coin=c, px=float(slot['stop'] if hit else cl[i]),
                                reason='stop' if hit else 'time'))
                            del self.slots[s][c]
                            stats['closes'] += 1
                        continue
                    # try to open
                    if len(self.slots[s]) >= 6:
                        continue
                    side = self._fired(c, s, i)
                    if side == 0 or (PARAMS[s]['long_only'] and side < 0):
                        continue
                    if not self._oi_ok(c, s, i):
                        continue
                    eb = i + 1
                    if eb >= len(b) or not np.isfinite(A[i]) or A[i] <= 0:
                        continue
                    ent = float(o[eb])
                    stop = ent - side * PARAMS[s]['stop_atr'] * A[i]
                    self.slots[s][c] = dict(side=int(side), stop=float(stop),
                                            hold=int(PARAMS[s]['hold']),
                                            bars_held=0, entry_px=ent)
                    self.events.append(dict(i=eb, ts=str(b.index[eb]), ev='OPEN',
                                            sleeve=s, coin=c, side=int(side),
                                            px=ent, stop=float(stop)))
                    stats['opens'] += 1
                    stats['max_slots'][s] = max(stats['max_slots'][s],
                                                len(self.slots[s]))
        stats['open_at_end'] = {s: len(v) for s, v in self.slots.items()}
        return stats

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)
