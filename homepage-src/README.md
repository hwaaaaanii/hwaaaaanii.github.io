# hwaaaaanii.github.io 리뉴얼 - 인수인계

Jeonghwan Choi (hwani.choi@kaist.ac.kr) 개인 홈페이지 리뉴얼 작업.

## 1. 지금 상태

`hwaaaaanii-homepage-preview.html` = 완성된 새 홈페이지 미리보기 (단일 파일, 이미지·아이콘 전부 인라인).
브라우저로 바로 열어볼 수 있음.

2026-08-11 세션에서 프로필 사진 교체와 최신 CV 반영을 완료했음. 아래 4·5절 참고.

## 2. 블로그 저장소

원격은 멀쩡함. 로컬 복구:

```bash
git clone --recurse-submodules https://github.com/hwaaaaanii/hwaaaaanii.github.io.git
cd hwaaaaanii.github.io && bundle install && bundle exec jekyll serve
```

`--recurse-submodules` 필수. `assets/lib`이 cotes2020/chirpy-static-assets 서브모듈이라
빼먹으면 로컬에서 CSS가 깨짐.

- 테마: jekyll-theme-chirpy 포크 (저장소에 테마 소스 전부 포함)
- 브랜치: master, 배포는 .github/workflows
- 기존 포스트: `_posts/` 13편 (논문 리뷰)
- 기존 탭: `_tabs/` (about, Publications, paper-reviews, archives, Paper Review Videos)

## 3. 디자인 방향 (사용자 확정)

**하이브리드**: 메인 페이지는 AcaNova-X 스타일 학술 홈페이지,
기존 Chirpy 블로그는 `/blog` 경로로 유지.

참고 템플릿: https://github.com/yihangtao/AcaNova-X (MIT)
원본 참고 사이트: https://tiancheng-gu.github.io

AcaNova-X는 순수 HTML/CSS/JS라 Jekyll이 아님. 그래서 통째로 쓰지 않고
**디자인만 이식**했음. Tailwind CDN 의존성 제거하고 순수 CSS로 재작성.
Font Awesome도 CDN 대신 SVG 스프라이트로 인라인 (`inline_icons.py`).

디자인 토큰 (원본과 동일):
- accent `#d4a562`, accent-dark `#b7791f`, primary `#1e293b`, bg `#f8fafc`
- 폰트: Crimson Text (제목), Inter (본문), Kalam (News 손글씨체)
- 레이아웃: 좌측 320px sticky 프로필 컬럼 + 우측 콘텐츠 컬럼

## 4. 프로필 사진 (2026-08-11 교체 완료)

원본: `/Users/hwaaaaanii/Desktop/Source./images./IMG_3897.HEIC`
pillow-heif로 JPEG 변환 후 크롭 (원본 4284x5712 → crop box
(560, 1290, 3290, 4814) → 620x800 리사이즈, brightness 1.05 / contrast 1.04).
처음엔 (900, 1150, 3100, 3980)으로 얼굴 위주로 잡았으나 너무 타이트하다는
지적을 받아, 셔츠·넥타이·코트가 보이는 상반신 구도로 다시 잡았음.
`src/profile.jpg`가 그 결과물이고, base64가 `src/profile.b64`에 들어있음.

주의: 이 사진은 화장실 거울 셀카이고 휴대폰이 오른쪽 눈을 가림. 사용자가
알고서 선택한 것이지만, 학술 홈페이지 용도로는 이전의 정장 증명사진이
더 나았음. 나중에 바꿀 생각이면 아래처럼 하면 됨:

```bash
# 새 사진을 620x800 안팎으로 만든 뒤
base64 -i profile.jpg -o src/profile.b64   # macOS
base64 -w0 profile.jpg > src/profile.b64   # Linux
cd src && ./build.sh
```

## 5. 콘텐츠 출처 (2026-08-11 CV 반영 완료)

`/Users/hwaaaaanii/Desktop/CV_JeonghwanChoi.pdf` 최신본 기준.
이전 세션이 쓰던 저장소 커밋본(2026-04-24)보다 훨씬 최신이라 아래를 반영했음.

이번에 CV 대조로 바뀐 것:
- COLM 2026 억셉 3편. 전에 "Under Review"였던 What Makes a Sale?,
  Towards Query-Agnostic RAG Evaluation, RAQE가 전부 COLM 2026으로 이동
- Findings of EMNLP 2026 신규 (Towards Robust RAG via Reliability-Aware
  Adaptive Evidence Selection, Minkyu Che 1저자)
- Under Review 2편으로 교체: Toward Agentic RAG for Enterprise Retail
  Workflows (Jeonghwan Choi 1저자), SoCRATES (Taewon Yun 1저자)
- KCC 2026 국내 논문 2편 추가 (Simulating Seller-Buyer Retail Interactions,
  Toward Low-Cost Query Expansion)
- Academic Service 섹션 신설: LASS 2026 PC 위원 (CIKM 2026 병설),
  DocInsights 2026 리뷰어 (EMNLP 2026 병설). 네비게이션에 Service 추가
- GS Retail 프로젝트 기간을 "Jan 2026 - Present"에서 "Jan 2026 - Jun 2026"으로 수정
- Physical AI 문구를 CV 원문 기준으로 교체. 이전 세션의 임시 초안이었는데
  CV에 "motivated by my work on agent simulation, I am recently expanding my
  research toward Physical AI, particularly world model-based approaches for
  robot learning and sim-to-real transfer"라고 명시돼 있어서 그대로 반영
- News 항목 4개 추가 (EMNLP, COLM x3, LASS PC, KCC)

이전 세션에서 이미 반영돼 있던 것:
- ICLR 2026 억셉, Findings of ACL 2026 억셉, 특허 2건, 수상 4건 + 장학금 2건
- IITP 갈등중재 멀티모달 에이전트 프로젝트, IE540 조교
- Ext2Gen -> "Aligning Extraction and Generation..." 제목 변경
- BRIDGE는 RDGENAI@CIKM 2025
- Google Scholar: https://scholar.google.com/citations?user=0DwYO7IAAAAJ

## 6. 사용자 지시사항 (반영 완료)

- Paper Reviews 카드 그리드 섹션 삭제 (네비게이션의 /blog 링크는 유지)
- Contact에서 사무실 주소 삭제
- 프로필의 한글 이름 '최정환' 삭제
- em대시(U+2014) / en대시(U+2013) 전부 제거. 문맥에 따라 쉼표·마침표·하이픈으로 대체
- About Me / Research Overview 카드 분량 축소. 콜론(:) 사용 자제.
  단, 문장이 뚝뚝 끊기지 않게 접속 표현으로 연결할 것 (where, and then, after 등)
- Latest News의 Kalam 손글씨체 제거 → 본문 Inter로 통일, 날짜는 tabular-nums
- 네비게이션 햄버거 메뉴 폐지. 좁은 화면에서는 이름 아래 가로 스크롤 스트립으로
  전체 항목이 항상 보이게 함 (`@media(max-width:1100px)`). Research 항목 추가
- 섹션 순서를 중요도 순으로 재배치:
  About > Research > News > Education > Publications > Patents > Honors >
  Service > Projects > Contact
  (Education은 사용자 요청으로 Publications 바로 앞에 배치)

## 7. 남은 작업

1. **News 날짜 검증** (중요). 억셉 날짜를 학회 일정으로 추정해서 채웠음.
   CV에는 연도만 있고 월이 없어서 확인이 안 됨. 실제 날짜로 교체할 것:
   - 2026.08 Findings of EMNLP 2026 (추정)
   - 2026.07 COLM 2026 3편 (추정)
   - 2026.07 LASS 2026 PC 위촉 (추정)
   - 2026.06 KCC 2026 (추정)
   - 2026.05 Findings of ACL 2026 (이전 세션 추정치, 미검증)
   - 2026.04 특허 2건차 출원 (CV 기준 Apr 2026, 확실)
   - 2026.01 ICLR 2026 (추정), GS Retail 시작 (CV 기준 Jan 2026, 확실)
   - 2025.12 KSC 2025 Best Paper (추정)
   - 2025.10 WSDM 2026 (추정)
   - 2025.06 Findings of ACL 2025 (추정)

2. **논문 링크 보강**. 아직 공개 링크가 없는 것은 RAQE (COLM 2026), BRIDGE,
   Agentic RAG for Retail, Reliability-Aware 네 편.
   나오는 대로 `pub-line-4`에 `<a class="pub-link-btn">` 추가.

3. **논문 썸네일 2건 미비**. 12편 중 10편은 채웠고 `02-socrates`, `10-bridge`는
   figure 대신 논문 약칭을 세리프로 조판한 타이틀 카드가 들어감. 나중에
   `papers/`에 그림을 넣고 다시 빌드하면 자동으로 그림으로 바뀜. 아래 10절 참고.

4. **저장소의 CV PDF 갱신**. CV 버튼은 상대경로 `CV_JeonghwanChoi.pdf`로 링크함.
   처음엔 루트 절대경로 `/CV_JeonghwanChoi.pdf`였는데, 그러면 미리보기 HTML을
   로컬이나 다른 호스트에서 열었을 때 그 호스트의 루트를 찾아가서 깨짐.
   상대경로로 바꾸고 PDF를 HTML과 같은 폴더에 복사해둠. 홈페이지가 사이트 루트
   (`permalink: /`)에 있으므로 배포 후에도 동일하게 `/CV_JeonghwanChoi.pdf`로 풀림.
   데스크톱 최신본을 저장소 루트에 덮어쓸 것.
   Jekyll로 이식할 때는 `{{ '/CV_JeonghwanChoi.pdf' | relative_url }}`로 쓰는 게 안전함.
   같은 이유로 Blog 링크도 `blog/` 상대경로로 바꿔둠.

5. **Chirpy 저장소 이식**. 아직 안 함. 계획:
   - `_layouts/homepage.html` 새로 만들고 현재 HTML의 구조를 Liquid로 옮김
   - CSS는 `assets/css/homepage.css`로 분리
   - 콘텐츠는 `_data/news.yml`, `_data/publications.yml`, `_data/honors.yml`,
     `_data/projects.yml`로 빼서 유지보수 쉽게
   - `_tabs/about.md`가 현재 `permalink: /`를 잡고 있으므로 이걸 새 레이아웃으로 교체
   - 기존 블로그 목록(`layout: home`)을 `/blog/`로 이동, 네비게이션 링크 연결

## 8. 빌드 방법

```bash
cd src && ./build.sh
```

`head.part`(HTML head + 전체 CSS) + `body.part`(본문) 를 합쳐 `template.html`을 만들고,
`annotate_counts.py`가 연도 헤더에 논문 수를 붙이고 각 논문 제목 앞에 CV와 같은 역순 번호 `[N]`..`[1]`을 넣은 뒤(둘 다 마크업에서 자동 계산, body.part에는 번호가 없음),
`inline_thumbs.py`가 `__THUMB_*__` 토큰을 논문 썸네일로 치환하고,
`inline_icons.py`가 Font Awesome `<i>` 태그를 SVG 스프라이트로 치환한 뒤,
`profile.b64`를 `__PROFILE_B64__` 자리에 넣어 최종 단일 HTML을 뱉음.
`inline_thumbs.py`가 `inline_icons.py`보다 먼저 돌아야 함. 썸네일이 없는 논문에
들어가는 플레이스홀더가 Font Awesome `<i>` 태그를 쓰기 때문.

내용을 고칠 때는 `body.part`를, 스타일을 고칠 때는 `head.part`를 수정할 것.
생성물(`template.html`, `template_svg.html`, 최종 html)은 직접 수정하지 말 것.

의존성: python3, `npm install @fortawesome/fontawesome-free` (SVG 원본용).
build.sh가 node_modules 없으면 알아서 설치함.

## 9. 검수 방법

Playwright로 확인. 2026-08-11 기준 통과 항목:
- 깨진 아이콘 참조 0개 (인라인 아이콘 28종)
- 가로 오버플로 0px (데스크톱 1440px / 모바일 390px)
- 네비게이션 바 오버플로 0px (Service 항목 추가 후에도 여유 있음)
- 프로필 이미지 로드 확인 (620x800)
- 논문 항목 21개 = CV의 publication 21편과 일치
- 유니코드 대시 0개 (build.sh가 assert로 강제)
- 썸네일 10개 로드 + 플레이스홀더 2개, 깨진 이미지 0개
  (1440 / 1100 / 900 / 700 / 390px 전부 오버플로 0px, JS 에러 0건)

## 10. 방문자 통계 (비공개)

`analytics.conf`의 `GOATCOUNTER_CODE=`가 비어 있으면 아무 스크립트도 안 나감.
지금은 꺼진 상태고, 페이지의 외부 요청은 0개.

켜는 법:

1. https://www.goatcounter.com 가입 (무료, 카드 불필요). 사이트 코드를 정함
2. `analytics.conf`의 `GOATCOUNTER_CODE=`에 그 코드를 적음
3. `cd homepage-src && ./build.sh ../index.html` 후 커밋, 푸시

대시보드는 `https://<코드>.goatcounter.com`. 로그인해야 보이고 페이지에는
숫자가 전혀 표시되지 않음. 쿠키를 안 쓰고 스크립트는 3.5KB.

본인 방문을 빼려면 브라우저 콘솔에서 한 번만
`localStorage.setItem('skipgc', 't')` 실행. 쓰는 브라우저마다 따로 해야 함.

## 10. 논문 썸네일 파이프라인 (2026-08-11 구축)

원본은 `~/Desktop/gitblog/papers/`. 파일명 규칙과 논문 매핑은 그 폴더의
`FILENAMES.md`에 있음. 확장자는 png/jpg/pdf 아무거나 되고, PDF면 1페이지를 렌더링함.
(실제로 받은 8개 PDF는 전부 이미 잘라낸 1페이지짜리 figure였음.)

변환은 pymupdf로 페이지를 가로 1400px로 렌더 → Pillow `ImageChops.difference`로
흰 여백 트림 → 가로 560px JPEG(q80)로 리사이즈 → base64로 `src/thumbs/<stem>.b64` 저장.
`build.sh`가 이 b64를 읽어서 HTML에 인라인함. `body.part`에는 `__THUMB_<stem>__`
토큰만 남아 있으므로 본문 편집이 base64에 파묻히지 않음.

새 논문을 추가할 때는 `src/thumbs/`에 같은 형식으로 b64를 넣고 `body.part`에
`<div class="pub-thumbnail-box">__THUMB_<stem>__</div>`를 쓴 뒤 `./build.sh`.
b64 파일이 없으면 자동으로 플레이스홀더가 들어가므로 빌드는 안 깨짐.

**썸네일 프레임 결정.** framework figure의 가로세로비가 1.6:1부터 5:1까지
제각각이고 `12-learn-to-verify`만 세로로 김. 그래서 박스는 250px 폭에 2:1로
고정하고 이미지는 `object-fit:contain`으로 통째로 보여줌. 넓은 그림은 위아래
여백이 생기지만 파이프라인 전체가 잘리지 않음.
꽉 찬 카드가 더 좋으면 `head.part`의 `.pub-thumb-img`를
`width:100%;height:100%;object-fit:cover`로 바꾸면 됨 (대신 그림 좌우가 잘림).
