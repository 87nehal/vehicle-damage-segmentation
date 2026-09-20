from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from vehicle_damage.model import Dinov2FourLayerDualTypeDamageSegmenter


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a four-layer binary model with dual type-probability heads"
    )
    parser.add_argument("--parent", required=True)
    parser.add_argument("--four-layer", required=True)
    parser.add_argument(
        "--fusion",
        choices=("average", "maximum", "classwise", "classwise_maximum"),
        default="average",
    )
    parser.add_argument("--weight-four-layer", type=float, default=0.5)
    parser.add_argument(
        "--class-weights-four-layer",
        type=float,
        nargs="+",
        help="one parent/four-layer interpolation weight per damage class",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.weight_four_layer <= 1.0:
        parser.error("--weight-four-layer must be in [0, 1]")

    parent_path = Path(args.parent).resolve()
    candidate_path = Path(args.four_layer).resolve()
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if parent.get("architecture") != "dinov2" or candidate.get("architecture") != "dinov2_4layer":
        raise ValueError("expected dinov2 parent and dinov2_4layer candidate")
    if parent["classes"] != candidate["classes"]:
        raise ValueError("checkpoint classes differ")
    class_count = len(parent["classes"]) - 1
    if args.class_weights_four_layer is not None and (
        len(args.class_weights_four_layer) != class_count
        or any(not 0.0 <= value <= 1.0 for value in args.class_weights_four_layer)
    ):
        parser.error(
            f"--class-weights-four-layer requires {class_count} values in [0, 1]"
        )
    if args.fusion in {"classwise", "classwise_maximum"} and args.class_weights_four_layer is None:
        parser.error("classwise fusion requires --class-weights-four-layer")
    for name, value in parent["model"].items():
        if name.startswith("backbone.") and not torch.equal(value, candidate["model"][name]):
            raise ValueError(f"frozen backbone tensor differs: {name}")

    classes = parent["classes"]
    config = dict(candidate.get("config", {}))
    model = Dinov2FourLayerDualTypeDamageSegmenter(
        len(classes),
        backbone_name=str(config.get("backbone_name", "vit_small_patch14_dinov2.lvd142m")),
        pretrained_backbone=False,
        case_classifier=False,
        type_probability_fusion=args.fusion,
        type_probability_weight_four_layer=args.weight_four_layer,
        type_probability_class_weights_four_layer=args.class_weights_four_layer,
    )
    state = model.state_dict()
    for name in state:
        if name.startswith("presence_head.") or name.startswith("exterior_head."):
            state[name] = candidate["model"][name]
        elif name.startswith("type_head_four_layer."):
            source_name = name.replace("type_head_four_layer.", "type_head.", 1)
            state[name] = candidate["model"][source_name]
        elif name.startswith("type_head.") or name.startswith("backbone."):
            state[name] = parent["model"][name]
    fusion = candidate["model"]["feature_fusion.weight"]
    state["presence_feature_fusion.weight"] = fusion
    state["exterior_feature_fusion.weight"] = fusion
    state["type_four_layer_feature_fusion.weight"] = fusion
    model.load_state_dict(state, strict=True)

    result = dict(candidate)
    result["model"] = model.state_dict()
    result["architecture"] = "dinov2_4layer_dual_type"
    result["config"] = {
        **config,
        "architecture": "dinov2_4layer_dual_type",
        "type_probability_fusion": args.fusion,
        "type_probability_weight_four_layer": args.weight_four_layer,
        "type_probability_class_weights_four_layer": args.class_weights_four_layer,
    }
    result["dual_type_hybrid"] = {
        "method": f"{args.fusion}_type_probability_fusion",
        "weight_four_layer": args.weight_four_layer,
        "class_weights_four_layer": args.class_weights_four_layer,
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
        **result["dual_type_hybrid"],
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
