from __future__ import annotations

import argparse
import json

from vehicle_damage.open_images_intake import prepare_open_images_candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fail-closed, non-training review inventory from local Open Images V7 files"
        )
    )
    parser.add_argument("--metadata", required=True, help="Official image metadata CSV")
    parser.add_argument("--boxes", required=True, help="Official bounding-box annotation CSV")
    parser.add_argument(
        "--class-descriptions", required=True, help="Official boxable class descriptions CSV"
    )
    parser.add_argument("--images-dir", required=True, help="Already-downloaded original images")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    report = prepare_open_images_candidates(
        metadata=args.metadata,
        boxes=args.boxes,
        class_descriptions=args.class_descriptions,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
