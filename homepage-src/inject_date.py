#!/usr/bin/env python3
"""Replace __LAST_UPDATED__ in template.html with the build date.

Uses today rather than the last git commit date. The build always runs just
before the commit, so reading git here would report the previous commit and
leave the page a day behind. The page is rebuilt only when something changes,
so the build date is the publication date a visitor cares about.
"""
import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, 'template.html')

stamp = date.today().isoformat()
html = open(TPL, encoding='utf-8').read()
assert '__LAST_UPDATED__' in html, 'date placeholder missing'
open(TPL, 'w', encoding='utf-8').write(html.replace('__LAST_UPDATED__', stamp))
print('  last updated: %s' % stamp)
