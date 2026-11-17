"""Train the 32-pixel WikiArt DCGAN. Run fetch.py first."""
from pathlib import Path
import csv
import numpy as np
import torch
from torch import nn
from PIL import Image

ROOT = Path(__file__).resolve().parent

def generator():
    layers = []
    for inp, out, stride, pad in [(64,128,1,0),(128,64,2,1),(64,32,2,1)]:
        layers.extend([nn.ConvTranspose2d(inp,out,4,stride,pad,bias=False),nn.BatchNorm2d(out),nn.ReLU()])
    return nn.Sequential(*layers,nn.ConvTranspose2d(32,3,4,2,1),nn.Tanh())

def discriminator():
    layers = [nn.Conv2d(3,32,4,2,1),nn.LeakyReLU(.2)]
    for inp,out in [(32,64),(64,128)]:
        layers.extend([nn.Conv2d(inp,out,4,2,1,bias=False),nn.BatchNorm2d(out),nn.LeakyReLU(.2)])
    return nn.Sequential(*layers,nn.Conv2d(128,1,4),nn.Flatten())

def train():
    torch.manual_seed(285)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = np.load(ROOT/'wikiart.npy')
    if dataset.shape != (10512,32,32,3):
        raise ValueError(f'Unexpected dataset shape {dataset.shape}')
    images = torch.from_numpy(dataset[:10000]).permute(0,3,1,2).float().to(device)/127.5-1
    gen,disc = generator().to(device),discriminator().to(device)
    for model in (gen,disc):
        for layer in model.modules():
            if isinstance(layer,(nn.Conv2d,nn.ConvTranspose2d)):
                nn.init.normal_(layer.weight,0,.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    go,do = [torch.optim.Adam(m.parameters(),lr=.0002,betas=(.5,.999)) for m in (gen,disc)]
    noise = torch.randn(25,64,1,1,device=device)
    loss_fn = nn.BCEWithLogitsLoss()
    with (ROOT/'loss.csv').open('w',newline='') as log:
        writer = csv.writer(log)
        writer.writerow(['epoch','discriminator','generator'])
        for epoch in range(1,41):
            losses = []
            for idx in torch.randperm(len(images),device=device).split(32):
                real = images[idx]
                fake = gen(torch.randn(len(idx),64,1,1,device=device))
                ones = torch.ones(len(idx),1,device=device)
                do.zero_grad()
                dl = loss_fn(disc(real),ones)+loss_fn(disc(fake.detach()),torch.zeros_like(ones))
                dl.backward()
                do.step()
                disc.requires_grad_(False)
                go.zero_grad()
                gl = loss_fn(disc(fake),ones)
                gl.backward()
                go.step()
                disc.requires_grad_(True)
                losses.append([dl.item(),gl.item()])
            writer.writerow([epoch,*np.mean(losses,axis=0)])
            log.flush()
            if epoch in (1,10,20,40):
                with torch.no_grad():
                    pixels = ((gen.eval()(noise)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
                canvas = Image.new('RGB',(5*34-2,5*34-2),'white')
                for i,pixel in enumerate(pixels):
                    canvas.paste(Image.fromarray(pixel),(i%5*34,i//5*34))
                canvas.save(ROOT/f'epoch-{epoch}.png')
                gen.train()
                print(epoch,np.mean(losses,axis=0),flush=True)
    torch.save(gen.cpu().state_dict(),ROOT/'generator.pt')

if __name__ == '__main__':
    train()
