from __future__ import annotations

import argparse
import json

from vehicle_damage.open_images_download import download_open_images_originals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a bounded set of official-metadata originals into quarantine"
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--class-descriptions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--max-image-bytes", type=int, default=20_000_000)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--report-name", default="DOWNLOAD_REPORT.json")
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--max-consecutive-rate-limits", type=int, default=3)
    args = parser.parse_args()
    report = download_open_images_originals(
        metadata=args.metadata,
        boxes=args.boxes,
        class_descriptions=args.class_descriptions,
        output_dir=args.output_dir,
        limit=args.limit,
        max_attempts=args.max_attempts,
        max_image_bytes=args.max_image_bytes,
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
        start_offset=args.start_offset,
        report_name=args.report_name,
        delay_seconds=args.delay_seconds,
        max_consecutive_rate_limits=args.max_consecutive_rate_limits,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
