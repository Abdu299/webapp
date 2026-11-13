
"""Live inference for the OSV image gallery."""

import base64
import csv
import io
import json
import os
import re
import sys
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image, ImageFilter, ImageOps

from .country_points import point


REPO = Path(__file__).resolve().parents[2]
IMAGES = REPO / "osv_image_galary"

WEIGHTS = (
    REPO
    / "model/vit_countries_last3_geographic_lr0.0001.pth"
)

LABELS = REPO / "privacyData/country_to_label.json"

PRIVACY_CKPT = REPO / os.getenv(
    "OSV_PRIVACY_CHECKPOINT",
    "output/geovit/privacy_network/geovit_seed0/best.ckpt"
)

SEGMENT_CKPT = Path(
    os.getenv(
        "OSV_SEGMENT_CHECKPOINT",
        str(REPO / "segmenter_896_seed0.pt")
    )
)

ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}

router = APIRouter(
    prefix="/api/osv",
    tags=["OSV live"]
)

lock = threading.Lock()

_model = None
_transform = None
_labels = None
_privacy_model = None
_segment_model = None

# Live heatmap and segmentation results per photo, reused by /importance
_heat_grids = {}
_clue_masks = {}


def file_for(photo_id):
    if not re.fullmatch(
        r"[A-Za-z0-9_-]{1,140}",
        photo_id
    ):
        raise HTTPException(400, "Invalid photo ID")

    for suffix in ALLOWED:
        path = IMAGES / (photo_id + suffix)

        if path.is_file():
            return path

    raise HTTPException(404, "OSV image not found")


def raw_image(photo_id):
    try:
        with Image.open(file_for(photo_id)) as image:
            return ImageOps.exif_transpose(
                image
            ).convert("RGB")
    except (OSError, ValueError):
        raise HTTPException(422, "Unreadable image")


def jpeg_bytes(image):
    output = io.BytesIO()
    image.save(
        output,
        format="JPEG",
        quality=88
    )
    return output.getvalue()


def jpeg_data(image):
    encoded = base64.b64encode(
        jpeg_bytes(image)
    ).decode("ascii")

    return "data:image/jpeg;base64," + encoded



def labels_for(photo_id):
    csv_path = IMAGES / "gallery.csv"

    if not csv_path.is_file():
        return {}

    with csv_path.open(
        newline="",
        encoding="utf-8-sig"
    ) as handle:

        reader = csv.DictReader(handle)

        for row in reader:
            if str(row.get("id", "")).strip() != photo_id:
                continue

            try:
                lat = float(row["latitude"])
                lng = float(row["longitude"])

                if not (-90 <= lat <= 90):
                    return {}

                if not (-180 <= lng <= 180):
                    return {}

                return {
                    "lat": lat,
                    "lng": lng,
                    "country": row.get("country", "")
                }

            except (ValueError, TypeError, KeyError):
                return {}

    return {}



def get_geovit():
    global _model, _transform, _labels

    if _model is not None:
        return _model, _transform, _labels

    with lock:
        if _model is not None:
            return _model, _transform, _labels

        if not WEIGHTS.is_file():
            raise RuntimeError(
                "GeoViT checkpoint not found"
            )

        if not LABELS.is_file():
            raise RuntimeError(
                "Country labels not found"
            )

        import torch
        from torchvision import transforms

        sys.path.insert(0, str(REPO))

        from modules.vitModel import GeoVit

        mapping = json.loads(LABELS.read_text())

        model = GeoVit(
            len(mapping),
            fine_tune="last3"
        )

        state = torch.load(
            WEIGHTS,
            map_location="cpu",
            weights_only=True
        )

        if "model_state_dict" in state:
            state = state["model_state_dict"]

        rename = {
            ".attention.attention.query.": ".attention.q_proj.",
            ".attention.attention.key.": ".attention.k_proj.",
            ".attention.attention.value.": ".attention.v_proj.",
            ".attention.output.dense.": ".attention.o_proj."
        }

        if any(
            ".q_proj." in name
            for name in model.state_dict()
        ):
            renamed = {}

            for key, value in state.items():
                for old, new in rename.items():
                    key = key.replace(old, new)

                renamed[key] = value

            state = renamed

        model.load_state_dict(
            state,
            strict=True
        )

        # CPU avoids the MPS bicubic error on Mac.
        _model = model.to("cpu").eval()

        _transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        _labels = {
            int(label): country
            for country, label in mapping.items()
        }

    return _model, _transform, _labels


def predict(image):
    import torch

    model, transform, labels = get_geovit()

    image_tensor = transform(
        image
    ).unsqueeze(0)

    with torch.inference_mode():
        probabilities = torch.softmax(
            model(image_tensor),
            dim=-1
        )[0]

        values, indices = probabilities.topk(3)

    top3 = []

    for value, index in zip(values, indices):
        top3.append({
            "country": labels[int(index)],
            "probability_pct": round(
                float(value) * 100,
                2
            )
        })

    country = top3[0]["country"]

    return {
        "model": point(country) or {
            "country": country
        },
        "top3": top3,
        "prediction_type": "country",
        "pin_type": "representative country location"
    }


def add_true_country(truth):
    if (
        truth.get("lat") is not None
        and not truth.get("country")
    ):
        try:
            import reverse_geocoder

            result = reverse_geocoder.search(
                (truth["lat"], truth["lng"]),
                mode=1
            )

            truth["country"] = result[0]["cc"]
        except Exception:
            pass

    return truth


@router.get("/images")
def list_images():
    IMAGES.mkdir(exist_ok=True)

    photos = []

    for path in sorted(IMAGES.iterdir()):
        if path.suffix.lower() not in ALLOWED:
            continue

        if not re.fullmatch(
            r"[A-Za-z0-9_-]{1,140}",
            path.stem
        ):
            continue

        photos.append({
            "id": path.stem,
            "title": f"OSV image {len(photos) + 1}",
            "thumbnail": (
                f"/api/osv/{path.stem}/image"
            )
        })

    return {
        "photos": photos,
        "count": len(photos)
    }


@router.get("/{photo_id}/image")
def get_image(photo_id: str):
    image = raw_image(photo_id)

    return Response(
        jpeg_bytes(image),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"}
    )


@router.post("/{photo_id}/predict")
def predict_original(photo_id: str):
    try:
        image = raw_image(photo_id)

        # Actual GeoViT inference happens now.
        result = predict(image)

        truth = labels_for(photo_id)

        result["truth"] = (
            add_true_country(truth)
            if truth
            else None
        )

        return result

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            503,
            (
                "GeoViT failed: "
                f"{type(error).__name__}: "
                f"{str(error)[:240]}"
            )
        )


@router.post("/{photo_id}/heatmap")
def calculate_heatmap(photo_id: str):
    """
    Live 7x7 occlusion heatmap.
    Measures change in top-country probability.
    """

    try:
        import torch
        from PIL import ImageDraw

        model, transform, _ = get_geovit()

        original = raw_image(photo_id)

        small = original.resize(
            (224, 224),
            Image.Resampling.BICUBIC
        )

        image_tensor = transform(
            small
        ).unsqueeze(0)

        blurred = small.filter(
            ImageFilter.GaussianBlur(radius=15)
        )

        blurred_tensor = transform(
            blurred
        ).unsqueeze(0)

        values = []

        with torch.inference_mode():
            original_probs = model(
                image_tensor
            ).softmax(-1)[0]

            top_idx = int(
                original_probs.argmax()
            )

            baseline = float(
                original_probs[top_idx]
            )

            for row in range(7):
                for col in range(7):
                    y0 = row * 32
                    x0 = col * 32

                    changed = image_tensor.clone()

                    changed[
                        :,
                        :,
                        y0:y0 + 32,
                        x0:x0 + 32
                    ] = blurred_tensor[
                        :,
                        :,
                        y0:y0 + 32,
                        x0:x0 + 32
                    ]

                    new_probs = model(
                        changed
                    ).softmax(-1)[0]

                    change = max(
                        0.0,
                        baseline - float(
                            new_probs[top_idx]
                        )
                    )

                    values.append(change)

        _heat_grids[photo_id] = values

        strongest = max(values) or 1.0

        canvas = Image.new(
            "RGBA",
            (224, 224),
            (0, 0, 0, 0)
        )

        draw = ImageDraw.Draw(canvas)

        for index, value in enumerate(values):
            strength = value / strongest

            if strength <= 0:
                continue

            col = index % 7
            row = index // 7

            draw.rectangle(
                (
                    col * 32,
                    row * 32,
                    col * 32 + 31,
                    row * 32 + 31
                ),
                fill=(
                    255,
                    int(180 * (1 - strength)),
                    0,
                    int(185 * strength)
                )
            )

        canvas = canvas.resize(
            original.size,
            Image.Resampling.BILINEAR
        )

        output = io.BytesIO()
        canvas.save(output, "PNG")

        encoded = base64.b64encode(
            output.getvalue()
        ).decode("ascii")

        return {
            "heatmap_data": (
                "data:image/png;base64," + encoded
            ),
            "method": (
                "Live 7x7 occlusion; "
                "change in GeoViT confidence"
            )
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            503,
            (
                "Live heatmap failed: "
                f"{type(error).__name__}: "
                f"{str(error)[:240]}"
            )
        )


@router.post("/{photo_id}/clues")
def calculate_clues(photo_id: str):
    global _segment_model

    if not SEGMENT_CKPT.is_file():
        raise HTTPException(
            503,
            (
                "Segmenter checkpoint missing. "
                "Set OSV_SEGMENT_CHECKPOINT."
            )
        )

    try:
        import numpy as np
        import torch
        import torch.nn.functional as F

        segment_dir = REPO / "segmenter"

        if str(segment_dir) not in sys.path:
            sys.path.insert(
                0,
                str(segment_dir)
            )

        from model import MaskSegmenter
        from clue_classes import CLUE_CLASSES
        from dataset import IMAGENET_MEAN, IMAGENET_STD

        if _segment_model is None:
            with lock:
                if _segment_model is None:
                    model = MaskSegmenter(
                        num_classes=10
                    )

                    state = torch.load(
                        SEGMENT_CKPT,
                        map_location="cpu",
                        weights_only=True
                    )

                    model.decoder.load_state_dict(
                        state
                    )

                    _segment_model = model.cpu().eval()

        original = raw_image(photo_id)

        resized = original.resize(
            (896, 672),
            Image.Resampling.BILINEAR
        )

        data = torch.from_numpy(
            np.array(resized)
        ).permute(2, 0, 1).float() / 255

        mean = torch.tensor(
            IMAGENET_MEAN
        ).view(3, 1, 1)

        std = torch.tensor(
            IMAGENET_STD
        ).view(3, 1, 1)

        with torch.inference_mode():
            scores = _segment_model(
                ((data - mean) / std).unsqueeze(0)
            )

            scores = F.interpolate(
                scores,
                size=(
                    original.height,
                    original.width
                ),
                mode="bilinear",
                align_corners=False
            )

            mask = scores.argmax(
                dim=1
            )[0].cpu().numpy()

        _clue_masks[photo_id] = mask.astype(np.uint8)

        rgba = np.zeros(
            (
                original.height,
                original.width,
                4
            ),
            dtype=np.uint8
        )

        breakdown = []

        for class_id, name, color in CLUE_CLASSES:
            pixels = mask == class_id

            rgba[pixels] = (*color, 120)

            breakdown.append({
                "name": name.replace(
                    "_", " "
                ).title(),
                "share_pct": round(
                    float(pixels.mean() * 100),
                    1
                ),
                "color": (
                    "#%02x%02x%02x" % color
                )
            })

        output = io.BytesIO()

        Image.fromarray(
            rgba,
            "RGBA"
        ).save(output, "PNG")

        encoded = base64.b64encode(
            output.getvalue()
        ).decode("ascii")

        return {
            "overlay_data": (
                "data:image/png;base64," + encoded
            ),
            "clues": breakdown
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            503,
            (
                "Live segmenter failed: "
                f"{type(error).__name__}: "
                f"{str(error)[:240]}"
            )
        )


@router.post("/{photo_id}/privacy")
def protect_image(photo_id: str):
    global _privacy_model

    if not PRIVACY_CKPT.is_file():
        raise HTTPException(
            503,
            (
                "Privacy checkpoint missing: "
                f"{PRIVACY_CKPT}"
            )
        )

    try:
        import numpy as np
        import torch

        sys.path.insert(0, str(REPO))

        from src.privacy_network import (
            geovit as privacy_geovit
        )

        # The original privacy code expects the
        # checkpoint in cached_models/geovit.
        privacy_geovit.weights_path = WEIGHTS

        from src.privacy_network.model import Model
        from src.blur.extremal import BlurPyramid
        from src.utils.common import unnormalize
        from torchvision.transforms.functional import (
            to_pil_image
        )

        if _privacy_model is None:
            with lock:
                if _privacy_model is None:
                    _privacy_model = (
                        Model.load_from_checkpoint(
                            str(PRIVACY_CKPT),
                            map_location="cpu",
                            strict=False
                        ).cpu().eval()
                    )

        original = raw_image(photo_id)

        transform = (
            _privacy_model.model.encoder.transform
        )

        image_tensor = transform(
            original
        ).unsqueeze(0).cpu()

        with torch.inference_mode():
            # Real trained privacy network.
            mask = _privacy_model(image_tensor)

            # Original privacy blur algorithm.
            blurred = BlurPyramid(
                image_tensor
            ).apply(mask)

            clean = unnormalize(
                blurred.cpu(),
                transform
            )[0].clamp(0, 1)

            original_tensor = unnormalize(
                image_tensor.cpu(),
                transform
            )[0].clamp(0, 1)

        protected = to_pil_image(
            clean
        ).convert("RGB")

        ssim = None

        try:
            from skimage.metrics import (
                structural_similarity
            )

            original_array = (
                original_tensor
                .permute(1, 2, 0)
                .numpy() * 255
            ).astype(np.uint8)

            protected_array = np.asarray(
                protected
            )

            ssim = round(
                float(
                    structural_similarity(
                        original_array,
                        protected_array,
                        channel_axis=2,
                        data_range=255
                    )
                ),
                3
            )
        except ImportError:
            pass

        return {
            "image_data": jpeg_data(protected),
            "blurred_area_pct": round(
                float(mask.mean()) * 100,
                1
            ),
            "ssim": ssim,
            "algorithm": (
                "Trained privacy network + BlurPyramid"
            )
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            503,
            (
                "Privacy model failed: "
                f"{type(error).__name__}: "
                f"{str(error)[:240]}"
            )
        )


class ProtectedPayload(BaseModel):
    image_data: str


@router.post("/{photo_id}/predict-protected")
def test_protected_image(
    photo_id: str,
    payload: ProtectedPayload
):
    # Verify that the selected OSV image exists.
    file_for(photo_id)

    try:
        header, encoded = payload.image_data.split(
            ",", 1
        )

        if header != "data:image/jpeg;base64":
            raise ValueError(
                "Invalid protected image"
            )

        if len(encoded) > 8_000_000:
            raise ValueError(
                "Protected image too large"
            )

        raw = base64.b64decode(
            encoded,
            validate=True
        )

        with Image.open(io.BytesIO(raw)) as image:
            protected = image.convert("RGB")

        # Second real GeoViT inference,
        # triggered by the Test button.
        return predict(protected)

    except (ValueError, OSError) as error:
        raise HTTPException(400, str(error))
    except Exception as error:
        raise HTTPException(
            503,
            (
                "Protected inference failed: "
                f"{type(error).__name__}: "
                f"{str(error)[:240]}"
            )
        )


CONTINENT_CLUES = Path(__file__).resolve().parent / "continent_clues.json"


@router.post("/{photo_id}/importance")
def calculate_importance(photo_id: str):
    """
    Heat ratio per clue class (mean heat on the class / mean heat of the
    photo, as in segmenter/heatmap_ranking.py) and the clue ranking for
    the photo's continent.
    """
    import numpy as np

    segment_dir = REPO / "segmenter"
    if str(segment_dir) not in sys.path:
        sys.path.insert(0, str(segment_dir))
    from clue_classes import CLUE_CLASSES
    from test_sets import CONTINENT

    # Only runs the models if the heatmap or clues button wasn't pressed
    if photo_id not in _heat_grids:
        calculate_heatmap(photo_id)
    if photo_id not in _clue_masks:
        calculate_clues(photo_id)

    # The heatmap cells are on the photo squashed to 224x224,
    # so the mask is squashed the same way
    mask = np.array(
        Image.fromarray(_clue_masks[photo_id]).resize(
            (224, 224), Image.Resampling.NEAREST
        )
    )
    grid = np.array(_heat_grids[photo_id], dtype=np.float32).reshape(7, 7)
    heat = np.repeat(np.repeat(grid, 32, axis=0), 32, axis=1)

    # A flat heatmap means no cell changed the guess, so there is nothing to rank
    importance = None
    if heat.max() >= 0.02 and heat.mean() > 0:
        importance = []
        for class_id, name, color in CLUE_CLASSES:
            in_class = mask == class_id
            if in_class.mean() < 0.01:
                continue
            importance.append({
                "name": name.replace("_", " ").title(),
                "color": "#%02x%02x%02x" % color,
                "heat_ratio": round(float(heat[in_class].mean() / heat.mean()), 2),
            })
        importance.sort(key=lambda c: c["heat_ratio"], reverse=True)

    # Same country -> continent table as the experiments
    country = labels_for(photo_id).get("country", "")
    continent = CONTINENT.get(country, "")
    continent_clues = None
    if continent and CONTINENT_CLUES.is_file():
        continent_clues = json.loads(CONTINENT_CLUES.read_text()).get(continent)

    return {
        "clue_importance": importance,
        "continent": continent,
        "continent_clues": continent_clues,
    }
