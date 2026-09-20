from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

from vehicle_damage.calibration import Calibration


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a higher-confidence segmentation threshold while preserving "
            "the calibrated recall-first threshold for case triage"
        )
    )
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--segmentation-threshold", required=True, type=float)
    parser.add_argument(
        "--basis",
        required=True,
        help="auditable description of the data and rule used to choose the threshold",
    )
    args = parser.parse_args()

    calibration = Calibration.load(args.calibration)
    result = replace(
        calibration,
        segmentation_threshold=args.segmentation_threshold,
        segmentation_threshold_basis=args.basis.strip(),
    )
    result.save(args.output)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
