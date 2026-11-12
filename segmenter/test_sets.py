"""
Make one CSV per week 3 test set, with the true location and continent of every photo.

- Im2GPS: the 237 Flickr photos. Their coordinates are stored in a JPEG comment
  (same trick as Victoria's src/images.py).
- OSV-5M: street view photos from the first test shard, sampled so every continent
  has photos, because the clue ranking is also made per continent.

The continent comes from the coordinates for both sets (nearest town -> country ->
continent), so both test sets are labelled the same way.

Run on the login node (it needs the internet to download OSV-5M):
    python test_sets.py --im2gps /fp/projects01/ec674/williaan/data/im2gps
Output: /fp/projects01/ec674/williaan/data/test_sets/{im2gps,osv5m}.csv
"""

import argparse
import os
import zipfile
from pathlib import Path

import pandas as pd
import reverse_geocoder
from huggingface_hub import hf_hub_download
from PIL import Image

DATA = "/fp/projects01/ec674/williaan/data"

# ISO country code -> continent. Central America and the Caribbean count as North America.
CONTINENT_CODES = {
    "Africa": "DZ AO BJ BW BF BI CV CM CF TD KM CD CG CI DJ EG GQ ER SZ ET GA GM GH GN GW KE LS LR LY "
              "MG MW ML MR MU YT MA MZ NA NE NG RE RW SH ST SN SC SL SO ZA SS SD TZ TG TN UG EH ZM ZW",
    "Asia": "AF AM AZ BH BD BT BN KH CN CY GE HK IN ID IR IQ IL JP JO KZ KW KG LA LB MO MY MV MN MM NP "
            "KP OM PK PS PH QA SA SG KR LK SY TW TJ TH TL TR TM AE UZ VN YE",
    "Europe": "AX AL AD AT BY BE BA BG HR CZ DK EE FO FI FR DE GI GR GG HU IS IE IM IT JE XK LV LI LT LU "
              "MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SJ SE CH UA GB VA",
    "North America": "AI AG AW BS BB BZ BM BQ VG CA KY CR CU CW DM DO SV GL GD GP GT HT HN JM MQ MX MS NI "
                     "PA PR BL KN LC MF PM VC SX TT TC US VI UM",
    "South America": "AR BO BR CL CO EC FK GF GY PY PE SR UY VE",
    # Christmas Island (CX) and the Cocos Islands (CC) are Australian territories
    "Oceania": "AS AU CC CK CX FJ PF GU KI MH FM NR NC NZ NU NF MP PW PG PN WS SB TK TO TV VU WF",
}
# Turned around for lookup: "NO" -> "Europe"
CONTINENT = {code: continent for continent, codes in CONTINENT_CODES.items() for code in codes.split()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--im2gps", default=os.path.join(DATA, "im2gps"))
    p.add_argument("--osv5m", default=os.path.join(DATA, "osv5m"))
    p.add_argument("--out", default=os.path.join(DATA, "test_sets"))
    # At most this many OSV-5M photos per continent. Equal numbers give every continent
    # a usable sample, instead of mostly Europe and North America like OSV-5M itself.
    p.add_argument("--per-continent", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def add_continent(df):
    """Add country and continent columns, looked up from lat and lon."""
    # mode=1 runs in this process. The default starts several processes, which is not needed here.
    places = reverse_geocoder.search(list(zip(df["lat"], df["lon"])), mode=1)
    df["country"] = [place["cc"] for place in places]
    df["continent"] = df["country"].map(CONTINENT).fillna("Other")
    # Show which countries are missing from CONTINENT_CODES, so they can be added
    missing = df.loc[df["continent"] == "Other", "country"].value_counts()
    if len(missing):
        print("countries with no continent:", missing.to_dict())
    return df


def read_comment_number(image, key):
    """Im2GPS keeps e.g. 'latitude: 59.9' in a JPEG comment. Returns the number or None."""
    for marker, data in image.applist:
        if marker == "COM" and key in data:
            return float(data.partition(key)[2].split(b"\x00")[0])
    return None


def im2gps_table(folder):
    rows = []
    for path in sorted(Path(folder).rglob("*.jpg")):
        with Image.open(path) as image:
            lat = read_comment_number(image, b"latitude:")
            lon = read_comment_number(image, b"longitude:")
        if lat is None or lon is None:
            print("no coordinates, skipped:", path.name)
            continue
        rows.append({"name": path.stem, "path": str(path), "lat": lat, "lon": lon})
    return add_continent(pd.DataFrame(rows))


def osv5m_table(folder, per_continent, seed):
    folder = Path(folder)
    csv_path = folder / "test.csv"
    image_dir = folder / "images" / "test"

    # Download the test metadata and the first of the test image zips (once)
    if not csv_path.exists():
        hf_hub_download("osv5m/osv5m", "test.csv", repo_type="dataset", local_dir=folder)
    if not image_dir.is_dir():
        zip_path = hf_hub_download("osv5m/osv5m", "images/test/00.zip", repo_type="dataset", local_dir=folder)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(image_dir)
        os.remove(zip_path)

    # Only photos from the downloaded shard can be used. Image file names are the ids.
    paths = {path.stem: str(path) for path in image_dir.rglob("*.jpg")}
    df = pd.read_csv(csv_path, dtype={"id": str})
    df = df[df["id"].isin(paths)]
    df = pd.DataFrame({"name": df["id"], "path": df["id"].map(paths),
                       "lat": df["latitude"], "lon": df["longitude"]})
    df = add_continent(df)

    # Up to per_continent random photos from each continent: shuffle, then keep the first ones
    df = df.sample(frac=1, random_state=seed).groupby("continent").head(per_continent)
    return df.reset_index(drop=True)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # A test set that already exists is kept, because masks and results are made from it.
    # Making it again could pick other photos (e.g. after a change to CONTINENT_CODES).
    tables = {}
    if os.path.exists(os.path.join(args.out, "osv5m.csv")):
        print("osv5m.csv exists, kept")
    else:
        tables["osv5m"] = osv5m_table(args.osv5m, args.per_continent, args.seed)
    # The Im2GPS photos come from Victoria, so OSV-5M can be made before they are copied over
    if os.path.exists(os.path.join(args.out, "im2gps.csv")):
        print("im2gps.csv exists, kept")
    elif os.path.isdir(args.im2gps):
        tables["im2gps"] = im2gps_table(args.im2gps)
    else:
        print("no Im2GPS folder yet, skipped:", args.im2gps)

    for name, df in tables.items():
        path = os.path.join(args.out, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"{name}: {len(df)} photos -> {path}")
        print(df["continent"].value_counts().to_string(), "\n")


if __name__ == "__main__":
    main()
