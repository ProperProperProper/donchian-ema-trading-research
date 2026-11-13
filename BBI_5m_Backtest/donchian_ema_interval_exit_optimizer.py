#!/usr/bin/env python3
"""Exit-only optimization for the best sampled entry per interval."""
from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

import pandas as pd

from bbi_bull_switch_backtest import fetch_candles
from donchian_ema_10m_baseline_backtest import add_features
from donchian_ema_10m_exit_optimizer import ExitParams, metric, run_exits

warnings.filterwarnings("ignore", category=FutureWarning)

ENTRY = {
    5: {"lookback": 41, "fast": 9, "slow": 193},
    10: {"lookback": 33, "fast": 17, "slow": 26},
    15: {"lookback": 22, "fast": 2, "slow": 92},
    30: {"lookback": 60, "fast": 21, "slow": 150},
    1440: {"lookback": 28, "fast": 4, "slow": 10},
}

CONTROL_EXIT = {
    5: {"tp": 0.4507, "initial_stop": 0.0410},
    10: {"tp": 0.0274, "initial_stop": 0.1324},
    15: {"tp": 0.1680, "initial_stop": 0.1710},
    30: {"tp": 0.4554, "initial_stop": 0.1510},
    1440: {"tp": 0.0942, "initial_stop": 0.0213},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--end", default="2026-11-12")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--output", default="interval_exit_results.json")
    args = parser.parse_args()
    end = pd.Timestamp(args.end)
    start = end - pd.Timedelta(days=args.days)
    rng = random.Random(args.seed)
    all_results = []

    for interval, entry in ENTRY.items():
        daily = interval == 1440
        candles = fetch_candles(args.symbol, "2025-01-01", args.end, interval, False)
        features = add_features(candles, entry["lookback"], entry["fast"], entry["slow"])
        if daily:
            features.index = features.index.normalize() + pd.Timedelta(hours=10)
        features = features.loc[features.index >= start]
        results = []
        control = ExitParams(**CONTROL_EXIT[interval])
        results.append({"entry": entry, "exit": control.__dict__,
                        "metrics": metric(run_exits(features, control, daily=daily)),
                        "interval_min": interval, "control": True})
        for _ in range(args.samples):
            if rng.random() < 0.65:
                p = ExitParams(
                    tp=round(rng.uniform(0.02, 0.50), 4),
                    initial_stop=round(rng.uniform(0.01, 0.25), 4),
                    be_trigger=rng.choice([0.0, 0.03, 0.05, 0.08]),
                    be_offset=rng.choice([0.0, 0.005]),
                    trail_activation=rng.choice([0.0, 0.03, 0.05, 0.08]),
                    trail_distance=rng.choice([0.0, 0.01, 0.02, 0.04]),
                    red_mode=rng.choice(["none", "ema_slow"]),
                )
                if (p.trail_activation == 0) != (p.trail_distance == 0):
                    continue
            else:
                p = ExitParams(
                    tp=1.0, initial_stop=0.99,
                    atr_period=rng.randint(5, 40),
                    atr_multiple=round(rng.uniform(2, 15), 2),
                    red_mode=rng.choice(["none", "ema_slow"]),
                    time_cap_bars=rng.choice([0, 20, 40, 60, 80, 120]),
                    time_cap_loss_only=True,
                )
            score = metric(run_exits(features, p, daily=daily))
            results.append({"entry": entry, "exit": p.__dict__, "metrics": score,
                            "interval_min": interval, "control": False})
        results.sort(key=lambda r: (r["metrics"]["compounded_pct"],
                                    r["metrics"]["max_drawdown_pct"]), reverse=True)
        all_results.extend(results)
        print(json.dumps({"interval_min": interval, "top": results[:5]}))

    Path(args.output).write_text(json.dumps({"window_start": str(start.date()),
                                              "window_end": str(end.date()),
                                              "results": all_results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
