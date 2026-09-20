from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from vehicle_damage.model import Dinov2FourLayerRoleSplitDamageSegmenter


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one model with parent triage heads and four-layer mask heads"
    )
    parser.add_argument("--parent", required=True)
    parser.add_argument("--four-layer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    parent_path = Path(args.parent).resolve()
    candidate_path = Path(args.four_layer).resolve()
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if parent.get("architecture") != "dinov2" or candidate.get("architecture") != "dinov2_4layer":
        raise ValueError("expected dinov2 parent and dinov2_4layer candidate")
    if parent["classes"] != candidate["classes"]:
        raise ValueError("checkpoint classes differ")
    for name, value in parent["model"].items():
        if name.startswith("backbone.") and not torch.equal(value, candidate["model"][name]):
            raise ValueError(f"frozen backbone tensor differs: {name}")

    classes = parent["classes"]
    config = dict(candidate.get("config", {}))
    model = Dinov2FourLayerRoleSplitDamageSegmenter(
        len(classes),
        backbone_name=str(config.get("backbone_name", "vit_small_patch14_dinov2.lvd142m")),
        pretrained_backbone=False,
        case_classifier=False,
    )
    state = model.state_dict()
    for name in state:
        if name.startswith("triage_presence_head."):
            state[name] = parent["model"][name.replace("triage_", "", 1)]
        elif name.startswith("triage_type_head."):
            state[name] = parent["model"][name.replace("triage_", "", 1)]
        elif name.startswith("triage_exterior_head."):
            state[name] = parent["model"][name.replace("triage_", "", 1)]
        elif name.startswith("backbone."):
            state[name] = parent["model"][name]
        elif name in candidate["model"]:
            state[name] = candidate["model"][name]
    model.load_state_dict(state, strict=True)

    result = dict(candidate)
    result["model"] = model.state_dict()
    result["architecture"] = "dinov2_4layer_role_split"
    result["config"] = {**config, "architecture": "dinov2_4layer_role_split"}
    result["role_split_hybrid"] = {
        "method": "parent_triage_four_layer_segmentation",
        "parent": str(parent_path),
        "parent_sha256": _sha256(parent_path),
        "four_layer": str(candidate_path),
        "four_layer_sha256": _sha256(candidate_path),
        "backbone_equality": "all backbone tensors bit-identical",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    report = {
        **result["role_split_hybrid"],
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "output_bytes": output.stat().st_size,
    }
    Path(str(output) + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
