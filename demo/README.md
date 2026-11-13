# GeoGuess AI – demo adapted to Abdu299/webapp

This folder is intended to be placed INSIDE your existing [Abdu299/webapp](https://github.com/Abdu299/webapp) clone as `demo/`. It **does not replace** `modules`, `segmenter`, `src`, `model`, or `privacyData`.

## What is connected

- Live uploaded photo: FastAPI strips EXIF/GPS in memory and runs `modules.vitModel.GeoVit` using the existing checkpoint `model/vit_countries_last3_geographic_lr0.0001.pth` and label map `privacyData/country_to_label.json`. Uses `Resize((224,224))`, ImageNet normalization, `softmax` and top three countries like `modules/vitTest.py`.
- Country-based predictions: map pins show a **representative country coordinate**. Your model predicts a country, not precise latitude/longitude.
- Gallery: from `segmenter/test_sets.py` CSVs + `resultsCountary/country_test_predictions.csv`, class ID masks from `segmenter/predict.py`, `.npy` heatmaps from `segmenter/heatmaps.py`, and blurred images from `src/privacy_network/blur_one.py` or its evaluation folder.
- Original answers are not returned in gallery browsing; only when the visitor presses **Lock in my guess**.
- Uploaded photos are never saved and have no true GPS label. Consequently they cannot receive a distance-to-truth score.

## Setup

```bash
# from your existing repository root
cp -R /path/to/unzipped/geoguess-ai-demo ./demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r demo/backend/requirements.txt
# For live inference, install torch/torchvision/transformers from your existing project environment.
python -m uvicorn demo.backend.main:app --reload --port 8000
```

In another terminal:

```bash
cd demo/frontend
npm install
npm run dev
```

Visit http://localhost:5173 and inspect http://localhost:8000/api/health.

**Important:** set `GEOVIT_REPO=/absolute/path/to/webapp` if running the demo outside the GitHub checkout. DINOv2-small must already be cached or downloadable from Hugging Face on first use. `mps` on Apple Silicon may be slower than CUDA and works only where supported.

## Build the precomputed gallery

In your repository root (on an environment that can access the images and metadata):

```bash
python demo/scripts/build_gallery.py \
  --csv /path/to/test_sets/osv5m.csv \
  --predictions /path/to/resultsCountary/country_test_predictions.csv \
  --masks /path/to/masks/osv5m \
  --heatmaps /path/to/heatmaps/geovit/osv5m \
  --blurred /path/to/blurred \
  --limit 20
```

**Requirements:** the CSV needs a `path` to each original JPG and GPS `lat`/`lon`; the prediction CSV needs `id`, `predicted_country`, and optional `top5_countries`, `top5_scores`. If your paths/IDs differ, adjust the CSV paths first. The exporter skips photos without verified ground truth, image files, or GeoViT prediction. It does not fabricate gallery entries.

Run without `--masks`, `--heatmaps`, `--blurred` to test just the GeoViT part. The script requires `numpy` in addition to Pillow.

To enable the privacy view for a particular photo, first run **the protected image through the same GeoViT model** and then set `privacy` in that photo's `result.json`, e.g.

```json
"privacy": {
  "available": true,
  "model": {"country":"NO", "lat":62, "lng":10, "kind":"country_representative"},
  "quality_metric":"SSIM",
  "quality_score":0.92
}
```

The numbers above are **schema examples only**, NOT actual results. Use the real country prediction and real SSIM/LPIPS from your privacy experiments. The displayed change in kilometers is between country representative points, **not an exact GPS change**.

The 0..9 class ID meanings/colors follow `segmenter/clue_classes.py`, not arbitrary labels. `.npy` heatmaps are normalized for visualization only; they must come from the matching photo and crop. The gallery exporter center-crops originals to a square as in `segmenter/heatmaps.py`: predictions originally run on un-cropped images may not exactly correspond to this displayed crop. For rigorous demonstration, rerun GeoViT predictions on the exported crops.

## Notes

- Do not claim privacy succeeded just because the country-point moved. Test whether the true country changed and show appropriate validated quality scores.
- No live segmentation/privacy models are loaded by default. Gallery mode is a reliable, fast fallback.
- For large Git repos, do not copy model checkpoints or entire `output` directories into your front-end bundle.
- Coordinate data in `backend/country_centres.json` is representative and approximate, not a scientific ground-truth geocoder.
