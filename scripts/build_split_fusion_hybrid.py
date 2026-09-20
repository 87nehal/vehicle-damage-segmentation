from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from vehicle_damage.model import Dinov2FourLayerSplitDamageSegmenter


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build one split-fusion model using four-layer binary heads and the "
            "single-layer parent type head"
        )
    )
    parser.add_argument("--parent", required=True)
    parser.add_argument("--four-layer", required=True)
    parser.add_argument(
        "--type-weight-four-layer",
        type=float,
        default=0.0,
        help="interpolate only the type head/fusion toward the four-layer candidate",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.type_weight_four_layer <= 1.0:
        parser.error("--type-weight-four-layer must be in [0, 1]")

    parent_path = Path(args.parent).resolve()
    candidate_path = Path(args.four_layer).resolve()
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if parent.get("architecture") != "dinov2":
        raise ValueError("parent must use the single-layer dinov2 architecture")
    if candidate.get("architecture") != "dinov2_4layer":
        raise ValueError("candidate must use the shared four-layer architecture")
    if parent["classes"] != candidate["classes"]:
        raise ValueError("checkpoint classes differ")
    for name, value in parent["model"].items():
        if name.startswith("backbone.") and not torch.equal(value, candidate["model"][name]):
            raise ValueError(f"frozen backbone tensor differs: {name}")

    classes = parent["classes"]
    config = dict(candidate.get("config", {}))
    model = Dinov2FourLayerSplitDamageSegmenter(
        len(classes),
        backbone_name=str(
            config.get("backbone_name", "vit_small_patch14_dinov2.lvd142m")
        ),
        pretrained_backbone=False,
        case_classifier=False,
    )
    state = model.state_dict()
    for name in state:
        if name.startswith("presence_head.") or name.startswith("exterior_head."):
            state[name] = candidate["model"][name]
        elif name.startswith("type_head."):
            state[name] = torch.lerp(
                parent["model"][name],
                candidate["model"][name],
                args.type_weight_four_layer,
            )
        elif name.startswith("backbone."):
            state[name] = parent["model"][name]
    fusion = candidate["model"]["feature_fusion.weight"]
    state["presence_feature_fusion.weight"] = fusion
    state["exterior_feature_fusion.weight"] = fusion
    state["type_feature_fusion.weight"] = torch.lerp(
        state["type_feature_fusion.weight"],
        fusion,
        args.type_weight_four_layer,
    )
    model.load_state_dict(state, strict=True)

    result = dict(candidate)
    result["model"] = model.state_dict()
    result["architecture"] = "dinov2_4layer_split"
    result["config"] = {**config, "architecture": "dinov2_4layer_split"}
    result["split_fusion_hybrid"] = {
        "method": "four_layer_presence_exterior_single_layer_type",
        "parent": str(parent_path),
        "parent_sha256": _sha256(parent_path),
        "four_layer": str(candidate_path),
        "four_layer_sha256": _sha256(candidate_path),
        "type_weight_four_layer": args.type_weight_four_layer,
        "backbone_equality": "all backbone tensors bit-identical",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    report = {
        **result["split_fusion_hybrid"],
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
