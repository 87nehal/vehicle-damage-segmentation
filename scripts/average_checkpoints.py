from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from vehicle_damage.checkpoint_soup import average_state_dicts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average two architecture-compatible, commercially approved checkpoints"
    )
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--weight-b", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    path_a = Path(args.checkpoint_a).resolve()
    path_b = Path(args.checkpoint_b).resolve()
    checkpoint_a = torch.load(path_a, map_location="cpu", weights_only=True)
    checkpoint_b = torch.load(path_b, map_location="cpu", weights_only=True)
    for field in ("architecture", "classes"):
        if checkpoint_a.get(field) != checkpoint_b.get(field):
            raise ValueError(f"checkpoint {field} values differ")
    config_a = checkpoint_a.get("config", {})
    config_b = checkpoint_b.get("config", {})
    for field in ("architecture", "backbone_name"):
        if config_a.get(field) != config_b.get(field):
            raise ValueError(f"checkpoint config {field} values differ")

    result = dict(checkpoint_a)
    result["model"] = average_state_dicts(
        checkpoint_a["model"], checkpoint_b["model"], args.weight_b
    )
    result["epoch"] = -1
    result["checkpoint_soup"] = {
        "method": "linear_weight_average",
        "weight_a": 1.0 - args.weight_b,
        "weight_b": args.weight_b,
        "checkpoint_a": str(path_a),
        "checkpoint_a_sha256": _sha256(path_a),
        "checkpoint_b": str(path_b),
        "checkpoint_b_sha256": _sha256(path_b),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    report = {
        **result["checkpoint_soup"],
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "output_bytes": output.stat().st_size,
    }
    Path(str(output) + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
