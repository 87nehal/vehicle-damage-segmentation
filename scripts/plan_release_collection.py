"""Create an evidence-based fresh-data collection plan from an evaluation report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vehicle_damage.collection_plan import plan_release_collection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate failed release gates into independent-group quotas"
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    plan = plan_release_collection(report)
    plan["source_report"] = {
        "path": str(report_path),
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "bytes": report_path.stat().st_size,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
