from __future__ import annotations

import io
import base64
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from . import live
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
GALLERY = Path(os.getenv('GALLERY_DIR', str(ROOT / 'gallery'))).resolve()
MAX_UPLOAD = 12 * 1024 * 1024
app = FastAPI(title='GeoGuess AI Demo', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'], allow_methods=['GET','POST'], allow_headers=['*'])


def safe_id(photo_id: str) -> str:
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}', photo_id):
        raise HTTPException(400, 'Invalid photo ID')
    return photo_id


def photo_dir(photo_id: str) -> Path:
    return GALLERY / safe_id(photo_id)


def read_result(photo_id: str) -> dict[str, Any]:
    path = photo_dir(photo_id) / 'result.json'
    if not path.is_file():
        raise HTTPException(404, 'Photo result missing')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(500, 'Invalid result.json')
    for k in ('truth', 'model'):
        point = data.get(k)
        if not isinstance(point, dict) or not all(isinstance(point.get(c), (int, float)) for c in ('lat', 'lng')):
            raise HTTPException(500, f'Missing {k} coordinates')
    return data


def public_result(data: dict[str, Any]) -> dict[str, Any]:
    # The frontend must not receive the answer before the guess is locked.
    return {'title': data.get('title', 'Mystery photo'), 'status': 'ready'}


@app.get('/api/health')
def health():
    return {'status': 'ok', 'live_model_available': live.available()}


@app.get('/api/gallery')
def gallery():
    index = GALLERY / 'index.json'
    if not index.is_file():
        return {'photos': [], 'message': 'Add gallery/index.json to load precomputed photos.'}
    try:
        photos = json.loads(index.read_text(encoding='utf-8'))
        if isinstance(photos, dict):
            photos = photos.get('photos', [])
        clean = []
        for photo in photos:
            try:
                pid = safe_id(str(photo['id']))
            except (KeyError, HTTPException):
                continue
            if (photo_dir(pid) / 'original.jpg').is_file() and (photo_dir(pid) / 'result.json').is_file():
                clean.append({'id': pid, 'title': photo.get('title', 'Mystery photo'), 'thumbnail': f'/api/photos/{pid}/original.jpg'})
        return {'photos': clean}
    except (OSError, ValueError, TypeError):
        raise HTTPException(500, 'Invalid gallery index.json')


@app.get('/api/photos/{photo_id}')
def photo_info(photo_id: str):
    return public_result(read_result(photo_id))


@app.get('/api/photos/{photo_id}/{filename}')
def photo_asset(photo_id: str, filename: str):
    if filename not in {'original.jpg', 'clues.png', 'heatmap.png', 'blurred.jpg'}:
        raise HTTPException(404, 'Unknown asset')
    path = photo_dir(photo_id) / filename
    if not path.is_file():
        raise HTTPException(404, 'Asset not available')
    return FileResponse(path, headers={'Cache-Control': 'public, max-age=3600'})


@app.post('/api/photos/{photo_id}/reveal')
def reveal(photo_id: str):
    # Reveals only after deliberate client interaction; this is a kiosk game, not secure anti-cheat.
    return read_result(photo_id)


@app.post('/api/upload')
async def upload(file: UploadFile = File(...)):
    if file.content_type not in {'image/jpeg', 'image/png', 'image/webp'}:
        raise HTTPException(415, 'Only JPEG, PNG, and WebP supported')
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, 'Image too large (max 12 MB)')
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            cleaned = ImageOps.exif_transpose(source).convert('RGB')
            cleaned.thumbnail((1600, 1600))
            buffer = io.BytesIO()
            cleaned.save(buffer, format='JPEG', quality=90)  # Removes GPS/EXIF.
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, 'Invalid image')
    # Inference uses only cleaned bytes in RAM. No upload saved to disk.
    try:
        inference = await run_in_threadpool(live.predict, buffer.getvalue())
    except Exception as exc:
        raise HTTPException(503, f'Live model unavailable: {type(exc).__name__}: {str(exc)[:200]}') from exc
    return {'id': 'upload', 'title': 'Your uploaded photo', 'image_data': 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii'), 'result': inference}


from .osv_live import router as osv_router

app.include_router(osv_router)