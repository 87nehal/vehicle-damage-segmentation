from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from vehicle_damage.model import Dinov2FourLayerDamageSegmenter


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a single-layer DINOv2 checkpoint to the identity-initialized "
            "four-layer architecture without changing its predictions"
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_path = Path(args.checkpoint).resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    if source.get("architecture") != "dinov2":
        raise ValueError("source checkpoint must use the single-layer dinov2 architecture")
    classes = source["classes"]
    config = dict(source.get("config", {}))
    model = Dinov2FourLayerDamageSegmenter(
        len(classes),
        backbone_name=str(
            config.get("backbone_name", "vit_small_patch14_dinov2.lvd142m")
        ),
        pretrained_backbone=False,
        case_classifier=any(
            key.startswith("case_classifier.") for key in source["model"]
        ),
        case_pooling=str(config.get("case_classifier", {}).get("pooling", "avg")),
    )
    incompatible = model.load_state_dict(source["model"], strict=False)
    if incompatible.missing_keys != ["feature_fusion.weight"] or incompatible.unexpected_keys:
        raise ValueError("source model differs beyond the expected fusion projection")

    result = dict(source)
    result["model"] = model.state_dict()
    result["architecture"] = "dinov2_4layer"
    result["config"] = {**config, "architecture": "dinov2_4layer"}
    result["architecture_upgrade"] = {
        "method": "four_layer_identity_fusion",
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "prediction_equivalence": "final-layer identity initialization",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    report = {
        **result["architecture_upgrade"],
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
