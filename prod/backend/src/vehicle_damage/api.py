from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy import ndimage

from .calibration import Calibration
from .inference import (
    assess_quality,
    predict_profile_probabilities,
    probabilities_to_mask,
    route_inference_decision,
)
from .model import load_checkpoint


ROOT = Path(__file__).resolve().parents[2]
SELECTED_MODEL = ROOT / "runs" / "SELECTED_DEVELOPMENT_MODEL.json"
MAX_UPLOAD_BYTES = 20_000_000
MAX_IMAGE_PIXELS = 50_000_000
CLASS_COLORS = (
    (234, 88, 12),
    (2, 132, 199),
    (220, 38, 38),
    (202, 138, 4),
    (147, 51, 234),
    (5, 150, 105),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_artifacts(selection_path: str | Path = SELECTED_MODEL) -> dict[str, object]:
    path = Path(selection_path).resolve()
    selection = json.loads(path.read_text(encoding="utf-8"))
    if (
        selection.get("status") != "development_only_not_production_approved"
        or selection.get("production_approved") is not False
    ):
        raise ValueError("selected model must retain its development-only safety status")
    result: dict[str, object] = {"selection": selection, "selection_path": path}
    for key in ("checkpoint", "development_profile"):
        declared = selection["artifacts"][key]
        artifact = ROOT / declared["path"]
        if not artifact.is_file():
            raise ValueError(f"missing selected {key}: {artifact}")
        if artifact.stat().st_size != declared["bytes"] or _sha256(artifact) != declared["sha256"]:
            raise ValueError(f"selected {key} failed its pinned integrity check")
        result[key] = artifact
    return result


def _score_map(
    damage_probability: torch.Tensor,
    exterior_probability: torch.Tensor,
    exterior_floor: float,
) -> np.ndarray:
    return (
        (1.0 - damage_probability[0])
        * (exterior_floor + (1.0 - exterior_floor) * exterior_probability)
    ).numpy()


def _components(
    mask: np.ndarray,
    score: np.ndarray,
    classes: list[str],
    *,
    kind: str,
    minimum_pixels: int,
    maximum_components: int = 100,
) -> list[dict[str, object]]:
    height, width = mask.shape
    records: list[dict[str, object]] = []
    labels, count = ndimage.label(
        mask > 0, structure=np.ones((3, 3), dtype=np.uint8)
    )
    for component_id in range(1, count + 1):
        region = labels == component_id
        area = int(region.sum())
        if area < minimum_pixels:
            continue
        class_counts = np.bincount(mask[region], minlength=len(classes))
        class_id = int(class_counts[1:].argmax()) + 1
        ys, xs = np.where(region)
        records.append(
            {
                "class_name": classes[class_id],
                "score": float(score[region].mean()),
                "bbox": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ],
                "area_pixels": area,
                "area_frac": area / (height * width),
                "kind": kind,
            }
        )
    return sorted(
        records, key=lambda row: float(row["score"]), reverse=True
    )[:maximum_components]


def _annotated_jpeg(image: Image.Image, mask: np.ndarray, review_mask: np.ndarray) -> str:
    canvas = image.convert("RGBA")
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    for class_id in np.unique(mask):
        if not class_id:
            continue
        color = CLASS_COLORS[(int(class_id) - 1) % len(CLASS_COLORS)]
        overlay[mask == class_id] = (*color, 122)
    overlay[(review_mask > 0) & (mask == 0)] = (245, 158, 11, 105)
    rendered = Image.alpha_composite(canvas, Image.fromarray(overlay, mode="RGBA"))
    rendered.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    rendered.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class DamagePredictor:
    def __init__(self, *, device: str | None = None) -> None:
        artifacts = selected_artifacts()
        self.selection = artifacts["selection"]
        self.checkpoint_path = Path(artifacts["checkpoint"])
        self.profile_path = Path(artifacts["development_profile"])
        self.device = torch.device(
            device or os.environ.get("VEHICLE_DAMAGE_DEVICE") or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        self.calibration = Calibration.load(self.profile_path)
        if self.calibration.mixed_precision and self.device.type != "cuda":
            raise RuntimeError(
                "the selected profile requires CUDA mixed-precision; no calibrated CPU "
                "profile is available"
            )
        self.model, checkpoint = load_checkpoint(str(self.checkpoint_path), self.device)
        self.classes = list(checkpoint["classes"])
        self.lock = threading.Lock()

    def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "model_loaded": True,
            "weights_exist": True,
            "cuda": self.device.type == "cuda",
            "device": str(self.device),
            "model_status": self.selection["status"],
            "production_approved": False,
            "checkpoint_sha256": self.selection["artifacts"]["checkpoint"]["sha256"],
            "profile_sha256": self.selection["artifacts"]["development_profile"]["sha256"],
            "triage_threshold": self.calibration.any_damage_threshold,
            "segmentation_threshold": self.calibration.effective_segmentation_threshold,
        }

    def predict_bytes(self, payload: bytes) -> dict[str, object]:
        started = time.perf_counter()
        if not payload:
            raise ValueError("uploaded image is empty")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise ValueError(f"uploaded image exceeds {MAX_UPLOAD_BYTES} bytes")
        try:
            with Image.open(io.BytesIO(payload)) as source:
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError(f"image exceeds {MAX_IMAGE_PIXELS} pixels")
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError("upload is not a supported safe image") from exc
        quality = assess_quality(image)
        with self.lock:
            (triage_damage, triage_exterior), (mask_damage, mask_exterior) = (
                predict_profile_probabilities(
                    self.model,
                    image,
                    device=self.device,
                    triage_scales=self.calibration.triage_scales,
                    segmentation_scales=self.calibration.effective_segmentation_scales,
                    tile_size=self.calibration.tile_size,
                    overlap=self.calibration.overlap,
                    horizontal_flip_tta=self.calibration.horizontal_flip_tta,
                    mixed_precision=self.calibration.mixed_precision,
                )
            )
        triage_mask, uncertainty = probabilities_to_mask(
            triage_damage,
            triage_exterior,
            self.calibration.any_damage_threshold,
            self.calibration.exterior_floor,
            self.calibration.minimum_component_pixels,
            self.calibration.type_probability_multipliers,
        )
        mask, _ = probabilities_to_mask(
            mask_damage,
            mask_exterior,
            self.calibration.effective_segmentation_threshold,
            self.calibration.exterior_floor,
            self.calibration.effective_segmentation_minimum_component_pixels,
            self.calibration.type_probability_multipliers,
        )
        review_mask = np.where((triage_mask > 0) & (mask == 0), triage_mask, 0).astype(
            np.uint8
        )
        near_threshold = float((uncertainty < 0.03).mean())
        decision = route_inference_decision(
            quality,
            triage_mask,
            mask,
            near_threshold_fraction=near_threshold,
            max_uncertain_fraction=0.10,
        )
        triage_score = _score_map(
            triage_damage, triage_exterior, self.calibration.exterior_floor
        )
        mask_score = _score_map(
            mask_damage, mask_exterior, self.calibration.exterior_floor
        )
        detections = _components(
            mask,
            mask_score,
            self.classes,
            kind="segmentation",
            minimum_pixels=max(
                1, self.calibration.effective_segmentation_minimum_component_pixels
            ),
        )
        review_candidates = _components(
            review_mask,
            triage_score,
            self.classes,
            kind="triage_only",
            minimum_pixels=max(16, self.calibration.minimum_component_pixels),
        )
        reason = {
            "damage_detected": "Damage evidence passed both calibrated branches.",
            "no_damage_detected": "No damage evidence passed the frozen thresholds.",
            "manual_review_required": "The result is uncertain and requires human review.",
            "recapture_required": "Image quality is insufficient; capture another photograph.",
        }[decision.decision]
        annotated_jpeg_b64 = _annotated_jpeg(image, mask, review_mask)
        return {
            "model_status": self.selection["status"],
            "production_approved": False,
            "decision": decision.decision,
            "automated_decision_allowed": decision.automated_decision_allowed,
            "manual_review_required": decision.manual_review_required,
            "recapture_required": decision.recapture_required,
            "decision_reasons": list(decision.decision_reasons),
            "recapture_reasons": list(decision.recapture_reasons),
            "reason": reason,
            "damage_present": bool(np.any(triage_mask)),
            "triage_alert": bool(np.any(triage_mask)),
            "high_confidence_alert": bool(np.any(mask)),
            "quality": asdict(quality),
            "detections": detections,
            "review_candidates": review_candidates,
            "n_detections": len(detections),
            "annotated_jpeg_b64": annotated_jpeg_b64,
            "image": {"width": image.width, "height": image.height},
            "runtime": {
                "device": str(self.device),
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            },
            "thresholds": {
                "triage": self.calibration.any_damage_threshold,
                "segmentation": self.calibration.effective_segmentation_threshold,
            },
        }


def create_app(
    predictor_factory: Callable[[], DamagePredictor] = DamagePredictor,
) -> FastAPI:
    state: dict[str, object] = {"predictor": None, "error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            state["predictor"] = predictor_factory()
        except Exception as exc:  # health must explain startup failures
            state["error"] = str(exc)
        yield

    app = FastAPI(title="Vehicle damage development API", lifespan=lifespan)

    @app.get("/api/health")
    def health() -> dict[str, object]:
        predictor = state["predictor"]
        if predictor is None:
            return {
                "ok": False,
                "damage": {"ok": False, "model_loaded": False, "error": state["error"]},
                "telemetry": {"ok": False},
            }
        return {
            "ok": True,
            "damage": predictor.health(),
            "telemetry": {"ok": False},
        }

    @app.get("/api/damage/samples")
    def samples() -> list[object]:
        return []

    @app.post("/api/damage/predict")
    async def predict(file: UploadFile = File(...)) -> dict[str, object]:
        predictor = state["predictor"]
        if predictor is None:
            raise HTTPException(status_code=503, detail=str(state["error"] or "model unavailable"))
        if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=415, detail="upload a JPEG, PNG, or WebP image")
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            return predictor.predict_bytes(payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "vehicle_damage.api:app",
        host=os.environ.get("VEHICLE_DAMAGE_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("VEHICLE_DAMAGE_API_PORT", "8001")),
        reload=False,
    )


if __name__ == "__main__":
    main()
