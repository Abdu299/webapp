"""
Groups the 66 Mapillary Vistas v1.2 classes into 10 clue classes.

The segmenter labels every pixel with one of these clue classes. In week 3 we
blur one clue class at a time and measure how much worse the geolocation model
gets, to find out which kinds of content give the location away.

Why 10 groups instead of all 66 classes:
- 10 groups means 10 blur experiments per photo instead of 66.
- Rare classes (e.g. pothole, phone booth) have too few pixels to learn alone,
  but are learnable when merged with similar classes.
- The answer we want is about kinds of clues ("signs give location away"),
  not single Mapillary classes.

The label files on disk keep all 66 classes. This mapping is applied when the
data is loaded, so the grouping can be changed without touching the dataset.
"""

import json

import numpy as np

# Pixels with this value are left out of the loss and the metrics, so the model
# is neither rewarded nor punished for them (e.g. unlabeled pixels, the camera car).
# 255 is the usual choice: it fits in uint8 and is far from any real class id.
IGNORE_INDEX = 255

# The classes the model predicts, as (id, name, RGB color).
# Ids must be 0..9 with no gaps, because the decoder has exactly one output
# channel per clue class. Colors are only used to draw predicted masks.
CLUE_CLASSES = [
    # Road surface material and quality (asphalt, concrete, dirt) varies by region.
    (0, "road", (128, 64, 128)),
    # Line colour and style differs by country (e.g. yellow vs white centre lines).
    (1, "road_markings", (255, 255, 255)),
    # Language, alphabet and sign design. Small and rare, so weighted more in week 2.
    # Bright yellow so they are easy to spot in the masks.
    (2, "signs_and_text", (255, 220, 0)),
    # Utility poles, street lights and bollards look different per country.
    # Small and rare like signs, so also weighted more. Bright orange.
    (3, "poles_and_street_furniture", (255, 120, 0)),
    # Architecture and building materials.
    (4, "buildings_and_structures", (70, 70, 70)),
    # Kerbs, guard rails, fences and pavements: roadside build quality and style.
    (5, "barriers_and_sidewalk", (190, 153, 153)),
    # Climate and landscape: palm trees vs pine forest, desert vs snow.
    (6, "vegetation_and_terrain", (107, 142, 35)),
    # Probably a weak clue, but it covers a large part of most photos, so it needs
    # its own class. Also a control in week 3: blurring sky should barely matter.
    (7, "sky", (70, 130, 180)),
    # Car models, number plates and which side of the road traffic drives on.
    (8, "vehicles", (0, 0, 142)),
    # Clothing. Also a privacy concern in its own right.
    (9, "people", (220, 20, 60)),
]

# Maps every Mapillary Vistas v1.2 class name (from config_v1.2.json) to a
# clue class id, or IGNORE_INDEX. All 66 names must be here.
# Rule used for each class: if week 3 blurs this clue class, should this thing
# disappear with it?
MAPPING = {
    # --- 0: road ---
    # Surfaces that vehicles drive on.
    "construction--flat--road": 0,
    "construction--flat--service-lane": 0,
    "construction--flat--bike-lane": 0,
    "construction--flat--parking": 0,
    "construction--flat--rail-track": 0,
    # Crosswalk area with no paint, so it is just road surface.
    # The painted stripes are marking--crosswalk-zebra (road_markings).
    "construction--flat--crosswalk-plain": 0,
    # These lie in the road surface, so blurring the road should remove them too.
    "object--manhole": 0,
    "object--catch-basin": 0,
    "object--pothole": 0,

    # --- 1: road_markings ---
    # Paint on the road. Colour and style of lines differ by country.
    "marking--general": 1,
    "marking--crosswalk-zebra": 1,

    # --- 2: signs_and_text ---
    # Anything that carries text or symbols that can reveal language or country.
    "object--traffic-sign--front": 2,
    # The back of a sign has no text, but its shape and mounting still differ by
    # country, and it is part of the same physical sign.
    "object--traffic-sign--back": 2,
    "object--billboard": 2,
    "object--banner": 2,

    # --- 3: poles_and_street_furniture ---
    # Thin vertical structures and small objects along the street.
    "object--support--pole": 3,
    "object--support--utility-pole": 3,
    # The frame holding a sign, not the sign itself, so it groups with poles.
    "object--support--traffic-sign-frame": 3,
    "object--street-light": 3,
    "object--traffic-light": 3,
    "object--bench": 3,
    "object--bike-rack": 3,
    "object--cctv-camera": 3,
    "object--fire-hydrant": 3,
    "object--junction-box": 3,
    "object--mailbox": 3,
    "object--phone-booth": 3,
    "object--trash-can": 3,

    # --- 4: buildings_and_structures ---
    # Large built structures showing architecture and building materials.
    "construction--structure--building": 4,
    "construction--structure--bridge": 4,
    "construction--structure--tunnel": 4,
    # A wall is built like a building (brick, stone, concrete), not a thin
    # roadside barrier like a guard rail, so it goes here.
    "construction--barrier--wall": 4,

    # --- 5: barriers_and_sidewalk ---
    # Roadside build quality and style: kerbs, rails, fences and pavements.
    "construction--barrier--curb": 5,
    "construction--barrier--fence": 5,
    "construction--barrier--guard-rail": 5,
    "construction--barrier--other-barrier": 5,
    "construction--flat--sidewalk": 5,
    "construction--flat--pedestrian-area": 5,
    # The lowered part of a kerb, so it belongs with the kerb and sidewalk.
    "construction--flat--curb-cut": 5,

    # --- 6: vegetation_and_terrain ---
    # Natural landscape, which reveals climate.
    "nature--vegetation": 6,
    "nature--terrain": 6,
    "nature--mountain": 6,
    "nature--sand": 6,
    "nature--snow": 6,
    "nature--water": 6,

    # --- 7: sky ---
    "nature--sky": 7,

    # --- 8: vehicles ---
    # All vehicles, including boats and trains: their models differ by country.
    "object--vehicle--car": 8,
    "object--vehicle--truck": 8,
    "object--vehicle--bus": 8,
    "object--vehicle--caravan": 8,
    "object--vehicle--trailer": 8,
    "object--vehicle--motorcycle": 8,
    "object--vehicle--bicycle": 8,
    "object--vehicle--on-rails": 8,
    "object--vehicle--boat": 8,
    "object--vehicle--wheeled-slow": 8,
    "object--vehicle--other-vehicle": 8,

    # --- 9: people ---
    # Riders are people on bikes or motorcycles; the bike itself is a vehicle.
    "human--person": 9,
    "human--rider--bicyclist": 9,
    "human--rider--motorcyclist": 9,
    "human--rider--other-rider": 9,

    # --- ignore ---
    # Animals are rare and not a useful location clue.
    "animal--bird": IGNORE_INDEX,
    "animal--ground-animal": IGNORE_INDEX,
    # Not part of the scene: the camera mount and the camera car itself.
    "void--car-mount": IGNORE_INDEX,
    "void--ego-vehicle": IGNORE_INDEX,
    # Pixels the annotators did not label, so there is no correct answer.
    "void--unlabeled": IGNORE_INDEX,
}


def build_lookup_table(config_path):
    """Return a uint8 array lut of length 256 so that lut[vistas_id] = clue id.

    A whole label image is then converted in one step with lut[label_array],
    which is far faster than looping over millions of pixels in Python.
    """
    # The Vistas id of a class is its position in this list (0 = animal--bird).
    with open(config_path) as f:
        names = [label["name"] for label in json.load(f)["labels"]]

    # Fail loudly on typos in MAPPING or on the wrong config version (v2.0 has
    # different names), instead of silently mapping those pixels to ignore.
    missing = [name for name in names if name not in MAPPING]
    if missing:
        raise ValueError(f"Vistas classes missing from MAPPING: {missing}")

    # 256 entries so every possible uint8 pixel value has an entry. Values that
    # are not real Vistas classes stay IGNORE_INDEX.
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for vistas_id, name in enumerate(names):
        lut[vistas_id] = MAPPING[name]
    return lut
