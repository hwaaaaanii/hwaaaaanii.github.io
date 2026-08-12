#!/usr/bin/env python3
"""Replace __THUMB_<stem>__ tokens in template.html with inline base64 <img> tags.

Reads thumbs/<stem>.b64 (raw base64 of a JPEG). If the file is missing, the
token is replaced with the neutral placeholder box instead, so a paper without
a figure still renders cleanly.

Run this BEFORE inline_icons.py so the placeholder's Font Awesome <i> tag gets
converted to an SVG sprite reference like every other icon.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, 'template.html')
THUMBS = os.path.join(HERE, 'thumbs')

PLACEHOLDER = ('<div class="pub-thumb-placeholder">'
               '<i class="fas fa-image"></i>thumbnail</div>')

html = open(TPL, encoding='utf-8').read()
tokens = re.findall(r'__THUMB_([0-9A-Za-z\-]+)__', html)
if not tokens:
    print('  no thumbnail tokens found')
    sys.exit(0)

filled, empty = 0, []
for stem in tokens:
    path = os.path.join(THUMBS, stem + '.b64')
    if os.path.exists(path):
        b64 = open(path, encoding='utf-8').read().strip()
        repl = ('<img class="pub-thumb-img" loading="lazy" alt="" '
                'src="data:image/jpeg;base64,%s">' % b64)
        filled += 1
    else:
        repl = PLACEHOLDER
        empty.append(stem)
    html = html.replace('__THUMB_%s__' % stem, repl)

assert '__THUMB_' not in html, 'unsubstituted thumbnail token remains'
open(TPL, 'w', encoding='utf-8').write(html)

print('  thumbnails inlined: %d / %d' % (filled, len(tokens)))
if empty:
    print('  still placeholder : %s' % ', '.join(empty))
