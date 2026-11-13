"""Lazy adapter for the EXISTING repo's modules.vitModel.GeoVit.

Requires running this webapp in the GitHub repository root with the existing
model/ checkpoint and privacyData/country_to_label.json available.
"""
import io
import json
import os
import threading
from pathlib import Path
from PIL import Image
from .country_points import point

REPO = Path(os.environ.get('GEOVIT_REPO', Path(__file__).resolve().parents[2])).resolve()
CHECKPOINT = REPO / 'model/vit_countries_last3_geographic_lr0.0001.pth'
LABELS = REPO / 'privacyData/country_to_label.json'
_lock = threading.Lock()
_model = None
_transform = None
_labels = None
_device = None

def available():
    return CHECKPOINT.is_file() and LABELS.is_file() and (REPO/'modules/vitModel.py').is_file()

def _load():
    global _model, _transform, _labels, _device
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        if not available():
            raise RuntimeError('GeoViT checkpoint or labels missing. Set GEOVIT_REPO to the cloned repository.')
        import sys
        import torch
        from torchvision import transforms
        sys.path.insert(0, str(REPO))
        from modules.vitModel import GeoVit
        mapping = json.loads(LABELS.read_text())
        _labels = {int(v):k for k,v in mapping.items()}
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        model = GeoVit(len(mapping), fine_tune='last3')
        state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        # Checkpoint may predate the transformers attention key rename.
        renames = {'.attention.attention.query.':'.attention.q_proj.', '.attention.attention.key.':'.attention.k_proj.', '.attention.attention.value.':'.attention.v_proj.', '.attention.output.dense.':'.attention.o_proj.'}
        if any('.q_proj.' in k for k in model.state_dict()):
            updated = {}
            for key,val in state.items():
                for old,new in renames.items():
                    key = key.replace(old,new)
                updated[key]=val
            state=updated
        model.load_state_dict(state, strict=True)
        _device=device
        _model=model.to(device).eval()
        # Exact transform used by modules/vitTest.py: resize both dimensions to 224.
        _transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def predict(clean_jpeg):
    _load()
    import torch
    image=Image.open(io.BytesIO(clean_jpeg)).convert('RGB')
    tensor=_transform(image).unsqueeze(0).to(_device)
    with torch.inference_mode():
        probs=torch.softmax(_model(tensor),dim=1)[0]
        values, indices=probs.topk(min(3,len(_labels)))
    top3=[{'country':_labels[int(i)],'probability_pct':round(float(v)*100,2)} for v,i in zip(values.cpu(),indices.cpu())]
    prediction=point(top3[0]['country']) or {'country':top3[0]['country']}
    return {'model':prediction,'top3':top3,'truth':None,'clues':[],'privacy':{'available':False},'prediction_type':'country'}
