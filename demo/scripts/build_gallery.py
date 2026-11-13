"""Export an actual research gallery using existing repo outputs.

CSV requires path,lat,lon; optional name/id,country. GeoVit predictions CSV from
modules.vitTest.py can supply top5_countries and top5_scores. Optional mask path
points to single-channel 0..9 PNG; heatmap path points to .npy; blur path is image.
Never fabricates predictions, GPS or metrics.

Example:
 python scripts/build_gallery.py --csv /path/test_sets/osv5m.csv --predictions /path/resultsCountary/country_test_predictions.csv --masks /path/masks/osv5m --heatmaps /path/privacyData/heatmaps/geovit/osv5m --blurred /path/output/privacy/blurred --limit 20
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from country_points import point

CLASSES = [
 ('road',(128,64,128)), ('road_markings',(255,255,255)), ('signs_and_text',(255,220,0)),
 ('poles_and_street_furniture',(255,120,0)), ('buildings_and_structures',(70,70,70)),
 ('barriers_and_sidewalk',(190,153,153)), ('vegetation_and_terrain',(107,142,35)),
 ('sky',(70,130,180)), ('vehicles',(0,0,142)), ('people',(220,20,60)),
]

def parse_list(v):
 return [i.strip() for i in str(v or '').split(',') if i.strip()]

def locate(folder, stem, extensions):
 if not folder:return None
 for ext in extensions:
  p=folder/(stem+ext)
  if p.is_file():return p
 return None

def read_predictions(path):
 if not path:return {}
 with open(path,newline='') as f:
  rows=list(csv.DictReader(f))
 return {str(r.get('id') or r.get('name') or Path(r.get('image_path','')).stem):r for r in rows}

def find_photo(row, base):
 p=Path(row.get('path') or row.get('image_path') or '')
 if p.is_file():return p
 p=base/p
 if p.is_file():return p
 return None

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--csv',required=True,type=Path,help='CSV with path,lat,lon (or latitude,longitude)')
 parser.add_argument('--predictions',type=Path,help='Output of modules.vitTest.py')
 parser.add_argument('--masks',type=Path,help='Grayscale class-id PNG files')
 parser.add_argument('--heatmaps',type=Path,help='NumPy .npy heatmaps')
 parser.add_argument('--blurred',type=Path,help='Privacy-network blurred PNG files')
 parser.add_argument('--limit',type=int,default=30)
 parser.add_argument('--out',type=Path,default=ROOT/'gallery')
 args=parser.parse_args()
 preds=read_predictions(args.predictions)
 with args.csv.open(newline='') as f:rows=list(csv.DictReader(f))
 args.out.mkdir(parents=True,exist_ok=True)
 index=[]
 for r in rows:
  if len(index)>=args.limit:break
  stem=str(r.get('name') or r.get('id') or Path(r.get('path') or r.get('image_path') or '').stem)
  photo=find_photo(r,args.csv.parent)
  if photo is None:
   print('skip missing file:',stem);continue
  try:
   lat=float(r.get('lat') or r.get('latitude'))
   lon=float(r.get('lon') or r.get('longitude'))
   if not (math.isfinite(lat) and math.isfinite(lon) and -90<=lat<=90 and -180<=lon<=180):continue
  except (ValueError,TypeError):
   print('skip missing GPS label:',stem);continue
  p=preds.get(stem)
  if not p:
   print('skip no GeoViT prediction:',stem);continue
  countries=parse_list(p.get('top5_countries'))
  scores=parse_list(p.get('top5_scores'))
  if not countries:
   countries=[p.get('predicted_country','')]
   scores=[p.get('prediction_score','')]
  top3=[]
  for country,score in zip(countries[:3],scores[:3]):
   try:top3.append({'country':country.upper(),'probability_pct':round(float(score)*100,2)})
   except (ValueError,TypeError):pass
  if not top3:continue
  estimated=point(top3[0]['country'])
  if estimated is None:
   print('skip country without point:',stem,top3[0]['country']);continue
  slug='photo_'+str(len(index)+1).zfill(3)
  dest=args.out/slug
  dest.mkdir(exist_ok=True)
  try:
   with Image.open(photo) as image:
    original=ImageOps.exif_transpose(image).convert('RGB')
    # Original square matches the centre crop of segmenter/heatmaps.py.
    square=ImageOps.fit(original,(768,768),method=Image.Resampling.LANCZOS)
    square.save(dest/'original.jpg',quality=88,exif=b'')
  except Exception as exc:
   print('bad image',stem,exc);continue
  clues=[]
  mask_path=locate(args.masks,stem,['.png'])
  if mask_path:
   with Image.open(mask_path) as image:
    arr=np.asarray(image.convert('L').resize((768,768),Image.Resampling.NEAREST))
   rgba=np.zeros((768,768,4),dtype=np.uint8)
   for i,(name,color) in enumerate(CLASSES):
    matching=arr==i
    share=round(float(matching.mean()*100),2)
    if share>0:clues.append({'name':name.replace('_',' ').title(),'color':'#%02x%02x%02x'%color,'share_pct':share})
    rgba[matching,:3]=color
    rgba[matching,3]=105
   Image.fromarray(rgba,'RGBA').save(dest/'clues.png')
  heat_path=locate(args.heatmaps,stem,['.npy'])
  if heat_path:
   heat=np.load(heat_path,allow_pickle=False).astype('float32')
   if heat.ndim==2:
    heat=np.nan_to_num(heat,nan=0,posinf=0,neginf=0)
    mx=float(np.max(heat))
    if mx>0:
     heat=np.clip(heat/mx,0,1)
     from PIL import ImageColor
     rgba=np.zeros((*heat.shape,4),dtype=np.uint8)
     rgba[...,0]=255
     rgba[...,1]=np.uint8(200*(1-heat))
     rgba[...,2]=np.uint8(40*(1-heat))
     rgba[...,3]=np.uint8(180*heat)
     Image.fromarray(rgba,'RGBA').resize((768,768),Image.Resampling.BILINEAR).save(dest/'heatmap.png')
  privacy={'available':False}
  blur=locate(args.blurred,stem,['_blurred.png','.png','_blurred.jpg','.jpg'])
  if blur:
   with Image.open(blur) as image:
    ImageOps.fit(image.convert('RGB'),(768,768),method=Image.Resampling.LANCZOS).save(dest/'blurred.jpg',quality=88)
   # We do not invent the privacy model's new prediction or image metrics.
   # Provide them later using a validated result.json generated by your experiment.
  result={'title':'Mystery photo '+str(len(index)+1),'truth':{'lat':lat,'lng':lon,'country':r.get('country') or p.get('actual_country') or ''},'model':estimated,'top3':top3,'clues':clues,'privacy':privacy,'prediction_type':'country','source_id':stem}
  (dest/'result.json').write_text(json.dumps(result,indent=2))
  index.append({'id':slug,'title':result['title']})
 (args.out/'index.json').write_text(json.dumps({'photos':index},indent=2))
 print(f'Exported {len(index)} real prediction rounds to {args.out}')
 print('Privacy view needs the new GeoViT prediction + SSIM/LPIPS in each result.json before it becomes available.')

if __name__=='__main__':main()
