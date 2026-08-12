#!/usr/bin/env bash
# Drop a new CV_JeonghwanChoi.pdf in this folder, then run this script.
# It commits and pushes, and GitHub Pages republishes the site within a minute or two.
#   ./update-cv.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f CV_JeonghwanChoi.pdf ]; then
  echo "ERROR: CV_JeonghwanChoi.pdf가 이 폴더에 없음."
  exit 1
fi

git add CV_JeonghwanChoi.pdf

if git diff --cached --quiet -- CV_JeonghwanChoi.pdf; then
  echo "변경 없음. 이미 최신 CV가 올라가 있음."
  exit 0
fi

git commit -m "Update CV ($(date +%Y-%m-%d))"
git push

echo
echo "푸시 완료. 빌드가 끝나면 (보통 1~2분) 아래에서 확인 가능:"
echo "  https://hwaaaaanii.github.io/CV_JeonghwanChoi.pdf"
echo "브라우저 캐시 때문에 옛날 게 보이면 새로고침(Cmd+Shift+R) 한 번."
