"""Fetch and convert the CC0 Humans in the Loop vehicle dataset.

The Kaggle archive is about 3.1 GB. This script reads its ZIP central directory
with HTTP range requests and fetches only annotation/image entries needed for
damage segmentation. This avoids downloading duplicate rendered masks and the
separate car-part subset.
"""

from __future__ import annotations

import argparse
import binascii
import io
import json
import random
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image, ImageDraw

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/humansintheloop/car-parts-and-car-damages"
LICENSE_ID = "CC0-1.0-HITL-car-parts-and-car-damages"

LABEL_MAP = {
    "dent": 1,
    "scratch": 2,
    "cracked": 3,
    "crack": 3,
    "broken part": 3,
    "broken-part": 3,
    "flaking": 4,
    "paint chip": 4,
    "paint-chip": 4,
    "corrosion": 4,
    "missing part": 5,
    "missing-part": 5,
}


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    crc32: int
    compressed_size: int
    size: int
    offset: int


class RemoteZip:
    def __init__(self, url: str) -> None:
        self._local = threading.local()
        response = self._session().get(url, stream=True, timeout=(20, 60))
        response.raise_for_status()
        self.url = response.url
        self.length = int(response.headers["Content-Length"])
        response.close()
        self.entries = self._directory()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=2)
            session.mount("https://", adapter)
            self._local.session = session
        return session

    def _range(self, start: int, end: int) -> bytes:
        for attempt in range(4):
            try:
                response = self._session().get(
                    self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 120)
                )
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"server ignored byte range: HTTP {response.status_code}")
                return response.content
            except (requests.RequestException, RuntimeError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    @staticmethod
    def _zip64_values(extra: bytes, header: tuple) -> tuple[int, int, int]:
        size, compressed, offset = header[9], header[8], header[16]
        position = 0
        while position + 4 <= len(extra):
            field_id, length = struct.unpack("<HH", extra[position : position + 4])
            payload = extra[position + 4 : position + 4 + length]
            position += 4 + length
            if field_id != 0x0001:
                continue
            cursor = 0
            if size == 0xFFFFFFFF:
                size = struct.unpack_from("<Q", payload, cursor)[0]
                cursor += 8
            if compressed == 0xFFFFFFFF:
                compressed = struct.unpack_from("<Q", payload, cursor)[0]
                cursor += 8
            if offset == 0xFFFFFFFF:
                offset = struct.unpack_from("<Q", payload, cursor)[0]
            break
        return size, compressed, offset

    def _directory(self) -> dict[str, ZipEntry]:
        tail_start = max(0, self.length - 65_557)
        tail = self._range(tail_start, self.length - 1)
        position = tail.rfind(b"PK\x05\x06")
        if position < 0:
            raise RuntimeError("ZIP end-of-central-directory record not found")
        eocd = struct.unpack("<4s4H2LH", tail[position : position + 22])
        central_size, central_offset = eocd[5], eocd[6]
        central = self._range(central_offset, central_offset + central_size - 1)
        entries: dict[str, ZipEntry] = {}
        position = 0
        while position + 46 <= len(central) and central[position : position + 4] == b"PK\x01\x02":
            header = struct.unpack("<4s6H3L5H2L", central[position : position + 46])
            filename_length, extra_length, comment_length = header[10], header[11], header[12]
            filename_start = position + 46
            name = central[filename_start : filename_start + filename_length].decode("utf-8")
            extra_start = filename_start + filename_length
            extra = central[extra_start : extra_start + extra_length]
            size, compressed, offset = self._zip64_values(extra, header)
            entries[name] = ZipEntry(name, header[4], header[7], compressed, size, offset)
            position += 46 + filename_length + extra_length + comment_length
        return entries

    def read(self, name: str) -> bytes:
        entry = self.entries[name]
        local = self._range(entry.offset, entry.offset + 29)
        header = struct.unpack("<4s5H3L2H", local)
        if header[0] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {name}")
        data_start = entry.offset + 30 + header[9] + header[10]
        compressed = self._range(data_start, data_start + entry.compressed_size - 1)
        if entry.method == 0:
            content = compressed
        elif entry.method == 8:
            content = zlib.decompress(compressed, -15)
        else:
            raise RuntimeError(f"unsupported ZIP compression method {entry.method}: {name}")
        if len(content) != entry.size or (binascii.crc32(content) & 0xFFFFFFFF) != entry.crc32:
            raise RuntimeError(f"integrity check failed: {name}")
        return content


def _meta_classes(meta: dict) -> set[str]:
    values: set[str] = set()
    for item in meta.get("classes", []):
        title = item.get("title") or item.get("name")
        if title:
            values.add(str(title).strip().lower())
    return values


def find_damage_root(remote: RemoteZip) -> tuple[str, dict]:
    meta_names = [name for name in remote.entries if name.endswith("/meta.json")]
    diagnostics = []
    for name in meta_names:
        meta = json.loads(remote.read(name))
        classes = _meta_classes(meta)
        overlap = classes & set(LABEL_MAP)
        diagnostics.append((name, sorted(classes), sorted(overlap)))
        if overlap:
            return name.rsplit("/", 1)[0], meta
    raise RuntimeError(f"could not identify damage subset from metadata: {diagnostics}")


def _paired_image(entries: dict[str, ZipEntry], root: str, annotation_name: str) -> str:
    filename = annotation_name.rsplit("/", 1)[-1]
    if filename.endswith(".json"):
        filename = filename[:-5]
    candidate = f"{root}/File1/img/{filename}"
    if candidate not in entries:
        raise KeyError(f"image missing for annotation: {annotation_name}")
    return candidate


def _draw_mask(annotation: dict, size: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    unknown: set[str] = set()
    for obj in annotation.get("objects", []):
        title = str(obj.get("classTitle", "")).strip().lower()
        class_id = LABEL_MAP.get(title)
        if class_id is None:
            unknown.add(title)
            continue
        points = obj.get("points", {}).get("exterior", [])
        if len(points) >= 3:
            draw.polygon([(float(x), float(y)) for x, y in points], fill=class_id)
        for interior in obj.get("points", {}).get("interior", []):
            if len(interior) >= 3:
                draw.polygon([(float(x), float(y)) for x, y in interior], fill=0)
    if unknown:
        raise ValueError(f"unmapped damage labels: {sorted(unknown)}")
    return mask


def _resize_pair(image: Image.Image, mask: Image.Image, max_side: int) -> tuple[Image.Image, Image.Image]:
    scale = min(1.0, max_side / max(image.size))
    if scale == 1.0:
        return image, mask
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS), mask.resize(size, Image.Resampling.NEAREST)


def prepare(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    images_dir, masks_dir = output / "images", output / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    remote = RemoteZip(DATASET_URL)
    root, meta = find_damage_root(remote)
    annotations = sorted(
        name for name in remote.entries if name.startswith(f"{root}/File1/ann/") and name.endswith(".json")
    )
    available = len(annotations)
    rng = random.Random(args.seed)
    rng.shuffle(annotations)
    if args.limit:
        annotations = annotations[: args.limit]
    print(json.dumps({"damage_root": root, "available": available, "selected": len(annotations), "classes": sorted(_meta_classes(meta))}))

    def convert(index: int, annotation_name: str) -> tuple[int, dict]:
        image_name = _paired_image(remote.entries, root, annotation_name)
        annotation = json.loads(remote.read(annotation_name))
        image = Image.open(io.BytesIO(remote.read(image_name))).convert("RGB")
        mask = _draw_mask(annotation, image.size)
        image, mask = _resize_pair(image, mask, args.max_side)
        stem = f"hitl_{index:05d}"
        image_path, mask_path = images_dir / f"{stem}.jpg", masks_dir / f"{stem}.png"
        image.save(image_path, format="JPEG", quality=92, optimize=True)
        mask.save(mask_path, format="PNG", optimize=True)
        damage_tags = sorted(
            {
                "damage:" + str(obj.get("classTitle", "")).strip().lower().replace(" ", "_")
                for obj in annotation.get("objects", [])
                if str(obj.get("classTitle", "")).strip()
            }
        )
        fraction = index / max(1, len(annotations))
        split = "train" if fraction < 0.70 else "validation" if fraction < 0.82 else "calibration" if fraction < 0.90 else "test"
        return index, {
                "image": image_path.relative_to(output).as_posix(),
                "mask": mask_path.relative_to(output).as_posix(),
                "split": split,
                "group_id": stem,
                "source": "Humans in the Loop car parts and car damages",
                "source_url": "https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/",
                "license_id": LICENSE_ID,
                "commercial_use": True,
                "tags": ["public_cc0", "damage_enriched", *damage_tags],
            }

    rows_by_index: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert, index, annotation_name): annotation_name
            for index, annotation_name in enumerate(annotations)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            index, row = future.result()
            rows_by_index[index] = row
            print(json.dumps({"prepared": completed, "total": len(annotations), "image": futures[future]}))
    rows = [rows_by_index[index] for index in range(len(annotations))]
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (output / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "dataset": "Car Parts and Car Damages",
                "publisher": "Humans in the Loop",
                "license": "CC0 1.0",
                "source_url": "https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/",
                "download_url": DATASET_URL,
                "archive_bytes": remote.length,
                "damage_root": root,
                "prepared_images": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/hitl_cc0")
    parser.add_argument("--limit", type=int, default=0, help="0 fetches every damage image")
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=8)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
