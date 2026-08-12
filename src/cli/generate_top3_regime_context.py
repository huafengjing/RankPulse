from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config.settings import AppSettings
from src.research.top3_regime_generator import (
    DEFAULT_OUTPUT_DIR,
    compare_with_reference,
    full_timeline,
    generate_context,
    rebuild_opportunity_csv_from_cache,
    write_timeline_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FR3/YR1 Top3 regime context.")
    parser.add_argument("--as-of", type=str, default="", help="UTC timestamp, e.g. 2026-07-06T00:00:00Z")
    parser.add_argument("--output", type=str, default="", help="Context JSON output path. Defaults to TOP3_REGIME_CONTEXT_PATH.")
    parser.add_argument("--inspect", action="store_true", help="Print context but do not write JSON.")
    parser.add_argument("--rebuild-opportunities", action="store_true", help="Rebuild Bucket B Rank3 opportunities from local kline cache.")
    parser.add_argument("--replay", action="store_true", help="Write full replay timeline and compare with frozen reference.")
    args = parser.parse_args()

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.rebuild_opportunities:
        path = rebuild_opportunity_csv_from_cache()
        print(f"已从本地缓存重建 opportunity 文件: {path}")

    if args.replay:
        rows = full_timeline()
        replay_path = output_dir / "full_timeline_equality.csv"
        write_timeline_csv(rows, replay_path)
        mismatches, summary = compare_with_reference(rows)
        mismatch_path = output_dir / "mismatch_audit.csv"
        pd.DataFrame(mismatches, columns=["signal_time", "field", "generated", "reference", "abs_error"]).to_csv(
            mismatch_path,
            index=False,
            encoding="utf-8-sig",
        )
        summary_path = output_dir / "historical_replay_comparison.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        print(f"Replay timeline: {replay_path}")
        print(f"Mismatch audit: {mismatch_path}")
        return

    settings = AppSettings.from_env_file()
    output = args.output or settings.top3_regime_context_path
    if not output and not args.inspect:
        output = str(output_dir / "regime_context.json")
    evaluation_time_ms = _parse_as_of_ms(args.as_of)
    result = generate_context(
        evaluation_time_ms=evaluation_time_ms,
        output_path=output if output else None,
        write_file=not args.inspect,
    )
    payload = result.evaluation.to_context_json(result.generated_at_ms)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if result.context_path is not None and not args.inspect:
        print(f"Context JSON 已写入: {result.context_path}")


def _parse_as_of_ms(value: str) -> int:
    if value:
        text = value.replace("Z", "+00:00")
        return int(pd.Timestamp(text).timestamp() * 1000)
    now = pd.Timestamp.utcnow().floor("h")
    return int(now.timestamp() * 1000)


if __name__ == "__main__":
    main()
