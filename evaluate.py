"""Evaluate the final checkpoint with the Ignite tutorial's probability features."""
import json
import os
import sys
from train import ROOT, generator

sys.path.insert(0,str(ROOT/'eval_lib'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'torch'))
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel

def resize(batch):
    return torch.stack([to_tensor(to_pil_image(im).resize((299,299),Image.Resampling.BILINEAR)) for im in batch])

@torch.no_grad()
def evaluate():
    torch.set_num_threads(4)
    g = generator().eval()
    g.load_state_dict(torch.load(ROOT/'generator.pt',map_location='cpu',weights_only=True))
    torch.manual_seed(286)
    fake = ((g(torch.randn(512,64,1,1))+1)/2).clamp(0,1)
    real = torch.from_numpy(np.load(ROOT/'wikiart.npy')[10000:]).permute(0,3,1,2).float()/255
    classifier = InceptionModel(return_features=False)
    fid = FID(num_features=1000)
    fake_is, real_is = InceptionScore(num_features=1000), InceptionScore(num_features=1000)
    for f,r in zip(fake.split(8),real.split(8)):
        pf,pr = classifier(resize(f)),classifier(resize(r))
        fid.update((pf,pr))
        fake_is.update(pf)
        real_is.update(pr)
    scores = dict(fid=fid.compute(),fake_is=fake_is.compute(),real_is=real_is.compute())
    print(scores)
    (ROOT/'metrics.json').write_text(json.dumps(scores))

if __name__ == '__main__':
    evaluate()
