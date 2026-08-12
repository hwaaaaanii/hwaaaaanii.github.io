#!/usr/bin/env bash
# Build the single-file homepage preview from head.part + body.part.
# Usage:  ./build.sh  [output.html]
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:-../hwaaaaanii-homepage-preview.html}"

if [ ! -d node_modules/@fortawesome/fontawesome-free ]; then
  echo "==> installing Font Awesome SVG source"
  npm install --silent @fortawesome/fontawesome-free
fi

echo "==> assembling template.html"
python3 -c "open('template.html','w',encoding='utf-8').write(
    open('head.part',encoding='utf-8').read()+open('body.part',encoding='utf-8').read())"

echo "==> inlining paper thumbnails"
python3 inline_thumbs.py

echo "==> inlining icons"
python3 inline_icons.py

echo "==> injecting profile image"
python3 - "$OUT" <<'PY'
import sys
b64 = open('profile.b64').read().strip()
html = open('template_svg.html', encoding='utf-8').read()
assert '__PROFILE_B64__' in html, 'profile placeholder missing'
open(sys.argv[1], 'w', encoding='utf-8').write(html.replace('__PROFILE_B64__', b64))
PY

echo "==> sanity check"
python3 - "$OUT" <<'PY'
import sys
s = open(sys.argv[1], encoding='utf-8').read()
print('  em-dash :', s.count('—'))
print('  en-dash :', s.count('–'))
print('  size    :', len(s), 'bytes')
assert s.count('—') == 0 and s.count('–') == 0, 'dashes present'
PY

echo "==> done: $OUT"
