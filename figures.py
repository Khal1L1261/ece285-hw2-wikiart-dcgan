import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
names = json.loads((root/'manifest.json').read_text(encoding='utf8'))
assert len(names) == len(set(names)) == 10512
paintings = np.load(root/'wikiart.npy',mmap_mode='r')
assert paintings.shape == (10512,32,32,3)
grid = Image.new('RGB',(168,168),'white')
for index, image in enumerate(paintings[10000:10025]):
    grid.paste(Image.fromarray(image),(index%5*34,index//5*34))
grid.save(root/'heldout.png')
with (root/'loss.csv').open() as file:
    rows = list(csv.DictReader(file))
assert [int(row['epoch']) for row in rows] == list(range(1,41))
fig,ax = plt.subplots(figsize=(5.0,1.27),layout='constrained')
for key, color in [('discriminator','#1b597e'),('generator','#a64a22')]:
    ax.plot(range(1,41),[float(r[key]) for r in rows],lw=1.35,label=key.capitalize(),color=color)
ax.set(xlabel='Epoch',ylabel='Mean loss',xlim=(1,40),xticks=[1,10,20,30,40])
ax.set_ylim(bottom=0)
ax.legend(loc='upper center',ncol=2,bbox_to_anchor=(.5,1.34),frameon=False,fontsize=8)
ax.spines[['right','top']].set_visible(False)
ax.tick_params(labelsize=8)
fig.savefig(root/'lossplot.pdf')
scores=json.loads((root/'metrics.json').read_text())
values = {'fidScore':f"{scores['fid']:.4f}",'fakeIS':f"{scores['fake_is']:.2f}",'realIS':f"{scores['real_is']:.2f}"}
for label,key in [('gen','generator'),('disc','discriminator')]:
    for end,row in [('First',rows[0]),('Last',rows[-1])]:
        values[label+end]=f'{float(row[key]):.3f}'
values['visualFinding']=(root/'observation.txt').read_text(encoding='utf8').strip()
(root/'results.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in values.items()),encoding='utf8')
