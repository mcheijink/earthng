"""Power lines and wind turbines for the country pack (mod/lua/ge/extensions/e2r/power.lua).

OSM power=line ways become 'power_line' lines, power=generator with generator:source=wind nodes
'wind_turbine' points. power=tower / pole / portal nodes become no points of their own: each line
run carries its vertices' tower facts in the tag TOWERS ('kind,design,height' per vertex, '|'
between vertices), so every tile that draws a span knows both towers the same way.

A line is not clipped to the cell like a fence: a span is 200-450 m and the cell's halo 100 m, so
a clipped span ended in mid-air. A cell keeps runs of whole spans instead, the spans whose box
reaches the padded cell plus one span past each end (the tower at a run's end then has both of
its neighbours, which set its crossarms). Run ids carry the way's vertex index of their first
vertex (':p<k>'), so the generator can tell the same span or tower in two cells apart from two
towers on one spot.
"""

TOWERS = 'e2r:towers'
# the tags the generator reads; the rest (names, operators, references) stays out of the pack
LINE_TAGS = ('power', 'voltage', 'cables', 'circuits', 'wires', 'design', 'line_arrangement')
TURBINE_TAGS = ('power', 'generator:source', 'generator:method', 'generator:type', 'height', 'rotor:diameter')
TOWER_KINDS = {'tower': 't', 'pole': 'p', 'portal': 'o'}
# lines inside a substation are no overhead line
NOT_LINES = {'busbar', 'bay'}
# metres past the padded cell a run's tower facts are looked up: a span and a half of the longest
# common spans (~450 m), so a run's extra end vertex finds its tower
TOWER_REACH = 800.0


def is_line(tags):
    return (tags.get('power') == 'line' and tags.get('line') not in NOT_LINES
            and tags.get('location') not in ('underground', 'underwater'))


def is_tower(tags):
    return tags.get('power') in TOWER_KINDS


def is_turbine(tags):
    return tags.get('power') == 'generator' and tags.get('generator:source') == 'wind'


def keep(tags, keys):
    return {k: tags[k] for k in keys if k in tags}


def _clean(value):
    return str(value or '').replace('|', ' ').replace(',', ' ').strip()


def tower_attr(tags):
    """'kind,design,height' of a tower node's tags ('' for a vertex without one)."""
    if not tags:
        return ''
    design = tags.get('design') or tags.get('line_arrangement') or ''
    return ','.join((TOWER_KINDS.get(tags.get('power'), ''), _clean(design), _clean(tags.get('height'))))


def key(x, y):
    """A vertex's place to the centimetre: a way vertex and its tower node share one location."""
    return (round(x * 100), round(y * 100))


def runs(coords, rect):
    """[(first, last)] vertex ranges (inclusive) of coords whose spans' boxes reach rect, each
    grown by one span at both ends (clamped to the way)."""
    x0, y0, x1, y1 = rect
    kept = []
    for k in range(len(coords) - 1):
        (ax, ay), (bx, by) = coords[k][:2], coords[k + 1][:2]
        if min(ax, bx) <= x1 and max(ax, bx) >= x0 and min(ay, by) <= y1 and max(ay, by) >= y0:
            kept.append(k)
    out = []
    for k in kept:
        first, last = max(0, k - 1), min(len(coords) - 1, k + 2)
        if out and first <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], last))
        else:
            out.append((first, last))
    return out


def line_runs(fid, n, coords, tags, rect, towers):
    """Line features (without z) of one way part: {'id', 'kind', 'tags', 'xy'}. towers maps key(x, y)
    to a tower node's tags."""
    base = keep(tags, LINE_TAGS)
    out = []
    for first, last in runs(coords, rect):
        part = [tuple(p[:2]) for p in coords[first:last + 1]]
        attrs = '|'.join(tower_attr(towers.get(key(*p))) for p in part)
        out.append({'id': f'{fid}:{n}:p{first}', 'kind': 'power_line', 'xy': part,
                    'tags': {**base, TOWERS: attrs}})
    return out
