from __future__ import annotations

import argparse
import json

from vehicle_damage.capture_audit import audit_capture_directory, write_capture_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen a capture directory for recapture and manual quality review"
    )
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True, help="JSON audit output")
    parser.add_argument("--csv", help="optional CSV output; defaults beside JSON")
    args = parser.parse_args()
    report = audit_capture_directory(args.images_dir)
    json_path, csv_path = write_capture_audit(report, args.output, args.csv)
    print(
        json.dumps(
            {
                "output": str(json_path),
                "csv": str(csv_path),
                "summary": report["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
