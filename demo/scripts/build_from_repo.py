"""Build a fully scored offline poster gallery from Abdu299/webapp's existing files.

Run from the *repository root*:
  python demo/scripts/build_from_repo.py --limit 20

Uses actual GeoViT country predictions for original AND protected photos.
Never treats the privacy experiment's artificial antipodal coordinates as GPS.
Clue segmentation is optional: pass --clue-masks PATH containing indexed 0..9 PNGs.
"""
import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DEMO = Path(__file__).resolve().parents[1]
REPO = DEMO.parent
sys.path.insert(0, str(DEMO))
from backend import live
from backend.country_points import point

CLASSES = [
    ('road', (128,64,128)), ('road_markings',(255,255,255)),
    ('signs_and_text',(255,220,0)), ('poles_and_street_furniture',(255,120,0)),
    ('buildings_and_structures',(70,70,70)), ('barriers_and_sidewalk',(190,153,153)),
    ('vegetation_and_terrain',(107,142,35)), ('sky',(70,130,180)),
    ('vehicles',(0,0,142)), ('people',(220,20,60)),
]

def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def predict_image(image):
    buf = io.BytesIO()
    image.convert('RGB').save(buf, 'JPEG', quality=92, exif=b'')
    return live.predict(buf.getvalue())

def export_heatmap(path, out):
    if not path.is_file():
        return False
    h = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if h.ndim != 2:
        return False
    h = np.nan_to_num(h, nan=0, posinf=0, neginf=0)
    # Strongest visible areas, not raw probability. The input heatmap is centred crop.
    high = float(np.percentile(h, 99))
    low = float(np.min(h))
    if high <= low:
        return False
    h = np.clip((h-low)/(high-low), 0, 1)
    rgba = np.zeros((*h.shape,4),dtype=np.uint8)
    rgba[...,0] = 255
    rgba[...,1] = (190*(1-h)).astype(np.uint8)
    rgba[...,2] = (30*(1-h)).astype(np.uint8)
    rgba[...,3] = (190*h).astype(np.uint8)
    Image.fromarray(rgba,'RGBA').resize((768,768),Image.Resampling.BILINEAR).save(out)
    return True

def export_clues(path, out):
    if not path or not path.is_file():
        return []
    with Image.open(path) as im:
        values = np.array(im.convert('L').resize((768,768), Image.Resampling.NEAREST))
    rgba = np.zeros((768,768,4),dtype=np.uint8)
    info=[]
    for idx,(name,color) in enumerate(CLASSES):
        selector = values == idx
        pct=float(selector.mean()*100)
        if pct > 0:
            info.append({'name':name.replace('_',' ').title(), 'color':'#%02x%02x%02x'%color, 'share_pct':round(pct,2)})
        rgba[selector,:3] = color
        rgba[selector,3] = 112
    Image.fromarray(rgba,'RGBA').save(out)
    return info

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=REPO,help='Root of cloned GitHub repository')
    p.add_argument('--limit',type=int,default=20,help='0 exports all 237 images')
    p.add_argument('--out',type=Path,default=DEMO/'gallery')
    p.add_argument('--clue-masks',type=Path,default=None,help='Optional folder of 0..9 segmentation mask PNGs')
    a=p.parse_args()
    repo=a.repo.resolve()
    live.REPO=repo
    live.CHECKPOINT=repo/'model/vit_countries_last3_geographic_lr0.0001.pth'
    live.LABELS=repo/'privacyData/country_to_label.json'
    if not live.available():
        p.error('Missing GeoViT files. Run from the cloned webapp repository and keep demo/ inside it.')
    coords=repo/'privacyData/im2gps.csv'
    images=repo/'privacyData/im2gps'
    heatmaps=repo/'privacyData/heatmaps/geovit/im2gps'
    exp=repo/'output/geovit/privacy_network/geovit_seed0/best/im2gps'
    blurred=exp/'blurred'
    exp_csv=exp/'geovit_predictions.csv'
    if not coords.is_file() or not images.is_dir():
        p.error('Missing privacyData/im2gps.csv or privacyData/im2gps images. Check clone/LFS files.')
    metrics={r['name']:r for r in read_csv(exp_csv)} if exp_csv.is_file() else {}
    try:
        import reverse_geocoder as rg
        location_rows=read_csv(coords)
        countries=rg.search([(float(r['lat']),float(r['lon'])) for r in location_rows],mode=1,verbose=False)
        country_by_stem={Path(r['filename']).stem:c['cc'] for r,c in zip(location_rows,countries)}
    except ImportError:
        country_by_stem={}
        print('WARNING: install reverse_geocoder to show true country codes.')
    rows=read_csv(coords)
    a.out.mkdir(parents=True,exist_ok=True)
    index=[]
    for row in rows:
        if a.limit and len(index)>=a.limit:
            break
        filename=row['filename']
        stem=Path(filename).stem
        original_path=images/filename
        if not original_path.is_file():
            print('SKIP missing original:',filename)
            continue
        try:
            lat,lon=float(row['lat']),float(row['lon'])
            if not (-90<=lat<=90 and -180<=lon<=180):
                raise ValueError('out-of-range coordinates')
            with Image.open(original_path) as im:
                image=ImageOps.exif_transpose(im).convert('RGB')
            prediction=predict_image(image)
            model=prediction['model']
            if not point(model.get('country')):
                raise ValueError('No representative point for '+str(model.get('country')))
            num=len(index)+1
            pid=f'photo_{num:03d}'
            dest=a.out/pid
            dest.mkdir(parents=True,exist_ok=True)
            # Generate a centrally cropped square to align with the published heatmap work.
            normalized=ImageOps.fit(image,(768,768),method=Image.Resampling.LANCZOS)
            normalized.save(dest/'original.jpg',quality=90,exif=b'')
            mask_path=a.clue_masks/(stem+'.png') if a.clue_masks else None
            clue_list=export_clues(mask_path,dest/'clues.png')
            heat=export_heatmap(heatmaps/(stem+'.npy'),dest/'heatmap.png')
            if not heat:
                print('WARNING no heatmap:',stem)
            privacy={'available':False}
            blur_path=blurred/(stem+'.png')
            if blur_path.is_file():
                with Image.open(blur_path) as im:
                    protected=ImageOps.exif_transpose(im).convert('RGB')
                ImageOps.fit(protected,(768,768),Image.Resampling.LANCZOS).save(dest/'blurred.jpg',quality=90,exif=b'')
                new_prediction=predict_image(protected)
                protected_model=new_prediction['model']
                # The predicted-country representatives are for display only, NOT actual GPS predictions.
                if point(protected_model.get('country')):
                    privacy={'available':True,'model':protected_model,
                             'new_top3':new_prediction['top3'], 'estimate_method':'country_representative'}
                    vals=metrics.get(stem,{})
                    for key in ('ssim','lpips','blurred_area'):
                        try:
                            value=float(vals[key])
                            if math.isfinite(value):
                                privacy[key]=round(value,5)
                        except (KeyError,TypeError,ValueError):
                            pass
                    if 'ssim' in privacy:
                        privacy['quality_metric']='SSIM'
                        privacy['quality_score']=privacy['ssim']
            data={'title':f'Mystery photo {num}',
                  'truth':{'lat':lat,'lng':lon,'country':country_by_stem.get(stem,'')},
                  'model':model,'top3':prediction['top3'],
                  'clues':clue_list,'privacy':privacy,
                  'prediction_type':'country','source_id':stem,
                  'heatmap_available':heat,'clues_available':bool(clue_list)}
            (dest/'result.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
            index.append({'id':pid,'title':data['title']})
            print(f'{len(index):3} {stem} | GeoViT {model["country"]} | privacy {privacy["available"]}',flush=True)
        except Exception as e:
            print('SKIP FAILED',stem,repr(e),flush=True)
    (a.out/'index.json').write_text(json.dumps({'photos':index},indent=2),encoding='utf-8')
    print(f'\nDONE: {len(index)} rounds, {a.out.resolve()}')
    print('Start backend: python -m uvicorn demo.backend.main:app --reload --port 8000')
    if not index:
        raise SystemExit('No rounds exported. Check errors above.')

if __name__=='__main__':
    main()
