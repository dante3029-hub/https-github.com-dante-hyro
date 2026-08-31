"""
Disk-persisted bot state, adapted from the original bot's BotState/Position
save()/load() JSON pattern (main.py) for the new portfolio-of-sleeves model.

Unlike the original single-strategy bot, this state tracks:
  - equity curve / peak equity / day-start equity / session date
  - serialized RiskState (risk_overlay.py) and ComplianceState (compliance.py)
  - per-sleeve last-rebalance timestamp (for cadence gating: 72h Main,
    weekly Flow/DELTA/RELVOL, 4h-checked Short/BOS)
  - per-coin target dollar positions (last computed) and last-known filled
    quantities, for the reconciliation layer to diff against
  - event-sleeve tracker state (open slots) for Short and BOS

RiskState and ComplianceState are dataclasses without a built-in
to-dict/from-dict; this module handles that translation explicitly rather
than pickling (JSON stays human-readable and diffable, which matters for a
system whose main audit tool right now is "read the state file").
"""
import os
import json
import datetime as dt
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

from portfolio_layer.risk_overlay import RiskState
from portfolio_layer.compliance import ComplianceState, HyroTraderComplianceConfig, DrawdownFloorMode, DrawdownType

logger = logging.getLogger("BotState")


def _risk_state_to_dict(rs: RiskState) -> dict:
    d = asdict(rs)
    d["session_date"] = rs.session_date.isoformat() if rs.session_date else None
    return d


def _risk_state_from_dict(d: dict) -> RiskState:
    d = dict(d)
    if d.get("session_date"):
        d["session_date"] = dt.date.fromisoformat(d["session_date"])
    return RiskState(**d)


def _compliance_state_to_dict(cs: ComplianceState) -> dict:
    d = {k: v for k, v in asdict(cs).items() if k != "config"}
    d["current_date"] = cs.current_date.isoformat() if cs.current_date else None
    d["config"] = dict(
        initial_balance=cs.config.initial_balance,
        max_drawdown_pct=cs.config.max_drawdown_pct,
        daily_loss_limit_dollars=cs.config.daily_loss_limit_dollars,
        drawdown_type=cs.config.drawdown_type.value,
        drawdown_floor_mode=cs.config.drawdown_floor_mode.value,
        min_trading_days=cs.config.min_trading_days,
    )
    # breach_log entries carry datetime objects -- stringify for JSON safety
    d["breach_log"] = [
        {**b, "at": b["at"].isoformat() if hasattr(b["at"], "isoformat") else str(b["at"])}
        for b in d.get("breach_log", [])
    ]
    return d


def _compliance_state_from_dict(d: dict) -> ComplianceState:
    d = dict(d)
    cfg_d = d.pop("config")
    cfg = HyroTraderComplianceConfig(
        initial_balance=cfg_d["initial_balance"],
        max_drawdown_pct=cfg_d["max_drawdown_pct"],
        daily_loss_limit_dollars=cfg_d["daily_loss_limit_dollars"],
        # drawdown_type absent = state file written before this field existed;
        # default to STATIC (the account's confirmed real terms) rather than
        # silently reverting to the stricter TRAILING dataclass default.
        drawdown_type=DrawdownType(cfg_d["drawdown_type"]) if "drawdown_type" in cfg_d else DrawdownType.STATIC,
        drawdown_floor_mode=DrawdownFloorMode(cfg_d["drawdown_floor_mode"]),
        min_trading_days=cfg_d["min_trading_days"],
    )
    if d.get("current_date"):
        d["current_date"] = dt.date.fromisoformat(d["current_date"])
    return ComplianceState(config=cfg, **d)


@dataclass
class BotState:
    equity: float = 200_000.0
    peak_equity: float = 200_000.0
    day_start_equity: float = 200_000.0
    session_date: Optional[str] = None          # ISO date string
    phase: int = 1                               # 1 or 2, mirrors ComplianceState.phase

    # cadence gating -- ISO timestamp of the last successful rebalance per sleeve
    last_rebalance: Dict[str, Optional[str]] = field(default_factory=lambda: {
        "main": None, "flow": None, "delta": None, "relvol": None,
        "short": None, "bos": None,
    })

    # last computed per-sleeve, per-coin dollar targets (for diffing/audit)
    last_dollar_targets: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # event-sleeve slot state (see bot/event_sleeves.py)
    short_tracker_state: dict = field(default_factory=lambda: {"slots": {}})
    bos_tracker_state: dict = field(default_factory=lambda: {"slots": {}})

    # serialized RiskState / ComplianceState (None until first initialized)
    risk_state: Optional[dict] = None
    compliance_state: Optional[dict] = None

    # append-only equity curve for audit / debugging (date, equity) pairs
    equity_curve: list = field(default_factory=list)

    total_realized_pnl: float = 0.0
    cycle_count: int = 0

    def get_risk_state(self) -> RiskState:
        if self.risk_state is None:
            rs = RiskState.new(self.equity)
            self.risk_state = _risk_state_to_dict(rs)
            return rs
        return _risk_state_from_dict(self.risk_state)

    def set_risk_state(self, rs: RiskState):
        self.risk_state = _risk_state_to_dict(rs)

    def get_compliance_state(self, config: HyroTraderComplianceConfig,
                             current_equity: float | None = None) -> ComplianceState:
        if self.compliance_state is None:
            # seed day-start from LIVE equity, not config.initial_balance
            cs = ComplianceState.start_phase1(config, current_equity=current_equity)
            self.compliance_state = _compliance_state_to_dict(cs)
            return cs
        return _compliance_state_from_dict(self.compliance_state)

    def set_compliance_state(self, cs: ComplianceState):
        self.compliance_state = _compliance_state_to_dict(cs)

    def save(self, fp: str):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        tmp = fp + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(asdict(self), f, indent=2, default=str)
            os.replace(tmp, fp)  # atomic -- avoids a torn/corrupt state file on crash mid-write
        except Exception as e:
            logger.error(f"Failed to save state to {fp}: {e}")

    @classmethod
    def load(cls, fp: str) -> "BotState":
        try:
            with open(fp) as f:
                data = json.load(f)
            return cls(**data)
        except FileNotFoundError:
            logger.info(f"No existing state file at {fp} -- starting fresh")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load state from {fp}: {e} -- starting fresh "
                         f"(NOT silently guessing partial state)")
            return cls()
