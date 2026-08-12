#!/usr/bin/env python3
"""Turn paper figures in ../papers/ into base64 thumbnails under thumbs/.

Input : <stem>.png | .jpg | .pdf  (optionally <stem>.pN.pdf to pick page N)
Output: thumbs/<stem>.b64  (raw base64 of a 560px-wide JPEG)

Usage: python3 make_thumbs.py [papers_dir]
"""
import base64, glob, io, os, re, sys
from PIL import Image, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'papers')
OUT  = os.path.join(HERE, 'thumbs'); os.makedirs(OUT, exist_ok=True)
MAXW = 560

def trim(im):
    bg = Image.new('RGB', im.size, (255, 255, 255))
    bb = ImageChops.difference(im.convert('RGB'), bg).getbbox()
    return im.crop(bb) if bb else im

def load(path):
    if path.lower().endswith('.pdf'):
        import pymupdf
        m = re.search(r'\.p(\d+)\.pdf$', path, re.I)
        page = int(m.group(1)) - 1 if m else 0
        d = pymupdf.open(path); p = d[page]
        zoom = max(1.0, 1400 / p.rect.width)
        pix = p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        return Image.open(io.BytesIO(pix.tobytes('png')))
    return Image.open(path)

for path in sorted(glob.glob(os.path.join(SRC, '*'))):
    if not re.match(r'^\d\d-', os.path.basename(path)):
        continue
    stem = re.sub(r'(\.p\d+)?\.(png|jpg|jpeg|pdf)$', '', os.path.basename(path), flags=re.I)
    im = trim(load(path)).convert('RGB')
    if im.width > MAXW:
        im = im.resize((MAXW, round(im.height * MAXW / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=80, optimize=True, progressive=True)
    open(os.path.join(OUT, stem + '.b64'), 'w').write(base64.b64encode(buf.getvalue()).decode())
    print('  %-36s %dx%d  %.1fKB' % (stem, im.width, im.height, buf.tell() / 1024))
