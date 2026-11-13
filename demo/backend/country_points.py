"""Representative country coordinates (not exact image predictions)."""
import json
from pathlib import Path

POINTS_FILE = Path(__file__).with_name('country_centres.json')
POINTS = json.loads(POINTS_FILE.read_text())

def point(code):
    p = POINTS.get(str(code).upper())
    if p is None:
        return None
    return {'lat': float(p[0]), 'lng': float(p[1]), 'country': str(code).upper(), 'kind': 'country_representative'}
