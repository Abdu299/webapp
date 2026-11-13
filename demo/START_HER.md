# GeoGuess AI – start here

This `demo/` folder is meant to be placed **inside** the GitHub repository
`https://github.com/Abdu299/webapp`, next to `modules/`, `privacyData/`,
`output/`, `model/`, and `segmenter/`.

## One-time setup (macOS)

From the cloned **webapp repository root**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r demo/backend/requirements.txt
python -m pip install torch torchvision transformers numpy==1.26.4 reverse_geocoder
cd demo/frontend && npm install && cd ../..
```

The `facebook/dinov2-small` backbone is downloaded on first model load.
Internet is required for that one-time download. The geolocation checkpoint
must be at `model/vit_countries_last3_geographic_lr0.0001.pth`.

## Build your real gallery

With `.venv` activated, run this **from webapp root**:

```bash
python demo/scripts/build_from_repo.py --limit 20
```

This command creates `demo/gallery/index.json` and 20 photo folders by using:

- `privacyData/im2gps/*.jpg` (237 original images)
- `privacyData/im2gps.csv` (ground-truth coordinates)
- `model/vit_countries_last3_geographic_lr0.0001.pth` plus country-to-label JSON (GeoViT inference)
- `privacyData/heatmaps/geovit/im2gps/*.npy` (real heatmaps)
- `output/geovit/privacy_network/geovit_seed0/best/im2gps/blurred/*.png` (privacy images)
- `output/geovit/privacy_network/geovit_seed0/best/im2gps/geovit_predictions.csv` (real SSIM/LPIPS)

GeoViT is evaluated on **both the original and protected** image. Existing
privacy CSV's 20015 km antipode values are NEVER interpreted as coordinates.

Optional: supply a folder of William's true 0..9 *clue segmentation masks*:

```bash
python demo/scripts/build_from_repo.py --limit 20 --clue-masks /path/to/clue/masks/im2gps
```

**Do not pass** `output/.../masks` from Victoria's privacy-network experiment:
those are blur masks, not William's 10-class clue masks.

To export all 237 photos:

```bash
python demo/scripts/build_from_repo.py --limit 0
```

## Run

First terminal, from `webapp/`:

```bash
source .venv/bin/activate
python -m uvicorn demo.backend.main:app --reload --port 8000
```

Second terminal, from `webapp/`:

```bash
cd demo/frontend
npm run dev
```

Open http://localhost:5173

Check http://localhost:8000/api/gallery – it must contain photo cards.

## Important research interpretation

GeoViT predicts **countries**. Pins are representative coordinates for
countries, not true continuous GPS predictions, so displayed km errors are
**approximate proxies**. The genuine research privacy results use a separate
0/20015-km country-correctness convention; do not conflate those with displayed
map distances. The segmenter outputs 0–9 clue IDs; a privacy blur mask is not
segmentation. Uploads do not have true GPS labels, so no user-vs-model score
can be computed for those images.

For a live poster, precompute the gallery on your laptop once, then play
without running a heavy model at every gallery click. The map's base tiles
and country border GeoJSON still require internet access.
