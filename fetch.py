"""Fetch a deterministic sample from Kaggle's WikiArt v1 ZIP via HTTP ranges."""
import concurrent.futures
import io
import json
import random
import struct
import threading
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

URL = 'https://www.kaggle.com/api/v1/datasets/download/steubk/wikiart'
root = Path(__file__).resolve().parent
session = threading.local()
head = requests.get(URL, headers={'Range':'bytes=-100'}, timeout=45)
head.raise_for_status()
total = int(head.headers['Content-Range'].split('/')[-1])

def read_range(start, end):
    if not hasattr(session, 'http'):
        session.http = requests.Session()
    for attempt in range(5):
        try:
            response = session.http.get(head.url, headers={'Range':f'bytes={start}-{end}'}, timeout=40)
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != end-start+1:
                raise IOError('unexpected byte range')
            return response.content
        except (requests.RequestException, IOError):
            if attempt == 4:
                raise
            time.sleep(1+attempt)

class RemoteZip(io.RawIOBase):
    def seek(self, offset, whence=0):
        self.pos = (0 if whence == 0 else self.pos if whence == 1 else total) + offset
        return self.pos
    def tell(self):
        return self.pos
    def read(self, size=-1):
        end = total if size < 0 else min(total,self.pos+size)
        data = read_range(self.pos,end-1) if end > self.pos else b''
        self.pos = end
        return data

archive = zipfile.ZipFile(RemoteZip())
files = sorted((i for i in archive.infolist() if i.filename.lower().endswith(('.jpg','.jpeg','.png'))), key=lambda i:i.filename)
sample = random.Random(285).sample(files,10512)
def get_image(info):
    start = info.header_offset
    data = read_range(start,min(total-1,start+info.compress_size+2047))
    name_len, extra_len = struct.unpack_from('<HH', data, 26)
    offset = 30+name_len+extra_len
    raw = data[offset:offset+info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(raw,-15)
    assert zlib.crc32(raw) == info.CRC
    with Image.open(io.BytesIO(raw)) as im:
        return np.asarray(ImageOps.fit(im.convert('RGB'),(32,32),method=Image.Resampling.LANCZOS))

cache = root/'image_cache'
cache.mkdir(exist_ok=True)
def cached_image(info):
    path = cache/f'{info.header_offset}.npy'
    if path.exists():
        return np.load(path)
    image = get_image(info)
    np.save(path,image)
    return image

images = []
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
    for i,img in enumerate(pool.map(cached_image,sample),1):
        images.append(img)
        if i%1000 == 0:
            print(i,'images prepared',flush=True)
np.save(root/'wikiart.npy',np.stack(images))
(root/'manifest.json').write_text(json.dumps([i.filename for i in sample],indent=2),encoding='utf8')
print('Archive images:',len(files),'sample:',len(images))
