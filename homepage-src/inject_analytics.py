#!/usr/bin/env python3
"""Replace __ANALYTICS__ in template.html with the GoatCounter beacon.

Reads GOATCOUNTER_CODE from analytics.conf. If it is empty the placeholder is
replaced with nothing, so an unconfigured build emits no script and the page
keeps making zero external requests.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, 'template.html')
CONF = os.path.join(HERE, 'analytics.conf')

code = ''
if os.path.exists(CONF):
    for line in open(CONF, encoding='utf-8'):
        m = re.match(r'\s*GOATCOUNTER_CODE\s*=\s*(\S*)\s*$', line)
        if m:
            code = m.group(1).strip().strip('"\'')

if code:
    assert re.fullmatch(r'[a-z0-9-]+', code), 'bad site code: %r' % code
    tag = ('<!-- private analytics: nothing is rendered, the numbers are only\n'
           '     visible on the logged-in dashboard -->\n'
           '<script data-goatcounter="https://%s.goatcounter.com/count"\n'
           '        async src="//gc.zgo.at/count.js"></script>' % code)
    print('  analytics: on (%s.goatcounter.com)' % code)
else:
    tag = ''
    print('  analytics: off, no script emitted')

html = open(TPL, encoding='utf-8').read()
assert '__ANALYTICS__' in html, 'analytics placeholder missing'
open(TPL, 'w', encoding='utf-8').write(html.replace('__ANALYTICS__\n', tag, 1).replace('__ANALYTICS__', tag))
