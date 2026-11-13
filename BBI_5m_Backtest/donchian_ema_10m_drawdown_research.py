#!/usr/bin/env python3
"""Focused drawdown research recommended by NVIDIA; entries remain frozen."""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from datetime import time
from pathlib import Path

from bbi_bull_switch_backtest import fetch_candles
from donchian_ema_10m_baseline_backtest import add_features
from donchian_ema_10m_exit_optimizer import ExitParams, metric, run_exits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-11-30")
    parser.add_argument("--output", default="drawdown_research_results.json")
    args = parser.parse_args()
    candles = fetch_candles(args.symbol, args.start, args.end, 10, False)
    candles = candles.loc[(candles.index.time >= time(9, 30)) & (candles.index.time <= time(16, 0))]
    features = add_features(candles, 45, 8, 89)
    results = []
    for stop, be, cap, adverse, adverse_bars in itertools.product(
        (0.0375, 0.0395), (0.03, 0.035), (0, 12, 24, 36),
        (0.0, 0.02, 0.025), (0, 12)):
        if adverse == 0 and adverse_bars != 0:
            continue
        params = ExitParams(tp=0.08, initial_stop=stop, be_trigger=be,
                            time_cap_bars=cap, adverse_stop=adverse,
                            adverse_bars=adverse_bars)
        score = metric(run_exits(features, params))
        results.append({"params": asdict(params), "metrics": score})
    results.sort(key=lambda x: (x["metrics"]["compounded_pct"], x["metrics"]["max_drawdown_pct"]), reverse=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    eligible = [x for x in results if x["metrics"]["compounded_pct"] >= 51]
    best_dd = min(eligible, key=lambda x: x["metrics"]["max_drawdown_pct"], default=None)
    print(json.dumps({"best_profit": results[0], "best_drawdown_at_51": best_dd}, indent=2))
    print(f"Tested {len(results)} exit-only risk configurations; report: {args.output}")


if __name__ == "__main__":
    main()
