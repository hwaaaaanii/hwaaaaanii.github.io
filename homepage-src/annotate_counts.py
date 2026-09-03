#!/usr/bin/env python3
"""Append a paper count to every publication year header in template.html.

"2026" becomes "2026 (7)". The count is derived from the markup at build time
rather than typed by hand, so adding a paper to body.part keeps the headers
honest with no second edit.

Run before inline_icons.py; it only touches text nodes inside the headers.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, 'template.html')

html = open(TPL, encoding='utf-8').read()

GROUP = re.compile(
    r'(<div class="pub-year-group">\s*<div class="pub-year-header">)([^<]+)(</div>)(.*?)(?=<div class="pub-year-group">|</section>)',
    re.S)


def annotate(m):
    head_open, label, head_close, body = m.groups()
    n = body.count('class="pub-list-item"')
    if not n:
        return m.group(0)
    label = re.sub(r'\s*<span class="pub-year-count">.*', '', label).strip()
    return '%s%s<span class="pub-year-count">(%d)</span>%s%s' % (
        head_open, label, n, head_close, body)


html, count = GROUP.subn(annotate, html)

# Reverse numbering [N] .. [1] across every publication, matching the CV's
# etaremune list: newest paper gets the largest number, oldest gets [1].
# The section markup is the single source of truth, so the count is derived
# here at build time instead of being typed into body.part.
SECTION = re.compile(r'(<section id="publications">.*?</section>)', re.S)
ITEM = re.compile(r'(<li class="pub-list-item">.*?<span class="pub-title-text">)(\s*)', re.S)


def number_items(m):
    sec = m.group(1)
    total = sec.count('class="pub-list-item"')
    state = {'n': total}

    def sub(mm):
        n = state['n']
        state['n'] -= 1
        return '%s<span class="pub-num">[%d]</span> %s' % (mm.group(1), n, mm.group(2))
    sec, done = ITEM.subn(sub, sec)
    assert done == total, 'numbered %d of %d publications' % (done, total)
    print('  publications numbered: [%d] .. [1]' % total)
    return sec


html, nsec = SECTION.subn(number_items, html)
assert nsec == 1, 'publications section not found'
open(TPL, 'w', encoding='utf-8').write(html)

pairs = re.findall(r'<div class="pub-year-header">([^<]*)<span class="pub-year-count">\((\d+)\)', html)
print('  year headers counted: %d' % count)
for label, n in pairs:
    print('    %-14s %s' % (label, n))
