import re, os, sys
def _find_fa():
    here=os.path.abspath(os.path.dirname(__file__))
    for _ in range(5):
        p=os.path.join(here,'node_modules','@fortawesome','fontawesome-free','svgs')
        if os.path.isdir(p): return p
        here=os.path.dirname(here)
    sys.exit("ERROR: Font Awesome SVGs not found.\n"
             "  run: npm install @fortawesome/fontawesome-free")
BASE=_find_fa()
html=open('template.html').read()

# find all <i class="..."></i>
used={}
def repl(m):
    cls=m.group(1)
    style=m.group(2) or ''
    parts=cls.split()
    fam='solid'
    name=None
    for p in parts:
        if p=='fab': fam='brands'
        elif p=='far': fam='regular'
        elif p.startswith('fa-'): name=p[3:]
    key=(fam,name)
    used[key]=True
    st=(' style="%s"'%style) if style else ''
    return '<svg class="ic"%s aria-hidden="true"><use href="#%s-%s"/></svg>'%(st,fam,name)

pat=re.compile(r'<i class="([^"]*fa-[^"]*)"(?:\s+style="([^"]*)")?\s*></i>')
html=pat.sub(repl,html)

# aliases FA6 renamed
ALIAS={'map-marker-alt':'location-dot','magnifying-glass':'magnifying-glass'}
sprite=['<svg xmlns="http://www.w3.org/2000/svg" style="display:none">']
missing=[]
for fam,name in sorted(used):
    fn=os.path.join(BASE,fam,name+'.svg')
    if not os.path.exists(fn):
        alt=ALIAS.get(name)
        if alt: fn=os.path.join(BASE,fam,alt+'.svg')
    if not os.path.exists(fn):
        missing.append((fam,name)); continue
    s=open(fn).read()
    vb=re.search(r'viewBox="([^"]+)"',s).group(1)
    inner=re.search(r'<svg[^>]*>(.*)</svg>',s,re.S).group(1)
    inner=re.sub(r'<!--.*?-->','',inner,flags=re.S).strip()
    sprite.append('<symbol id="%s-%s" viewBox="%s">%s</symbol>'%(fam,name,vb,inner))
sprite.append('</svg>')
if missing:
    sys.exit('ERROR: icons not found in Font Awesome: %r' % (missing,))

CSS='''
.ic{width:1em;height:1em;fill:currentColor;display:inline-block;vertical-align:-.125em;flex-shrink:0}
'''
html=html.replace('</style>', CSS+'</style>')
html=html.replace('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">\n','')
html=html.replace('<body>','<body>\n'+''.join(sprite)+'\n')
open('template_svg.html','w').write(html)
print('icons inlined:',len(used)-len(missing))
