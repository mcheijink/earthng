"""Traffic signals and speed cameras for the scenery reader (fast_sources).

The pack carries OSM nodes as points with all their tags; the runtime (mod/.../e2r/signals.lua) stands
the shapes. Two things only the exporter can know, because the pack's roads are merged and may run
against their ways (fast_roads orients strokes by travel or node order):

* `traffic_signals:direction` / `direction` = forward or backward is relative to the way the node is
  on. The tracker turns it into an absolute heading of the controlled traffic.
* An `enforcement=maxspeed` relation names its camera (`device`) and the stretch it checks
  (`from` -> `to`). The tracker makes a camera point of a device that is not a speed_camera node
  and gives every device the heading from -> to.

The heading goes into the point's tags as HEADING: degrees clockwise from north, the direction the
controlled or checked traffic travels. The runtime snaps it to the road it finds.
"""
import math

HEADING = 'e2r:heading'
SIGNAL, CAMERA = 'signal', 'speed_camera'
# a numeric or cardinal direction=* on a speed camera is the heading of the traffic it checks
CARDINAL = {'N': 0., 'NNE': 22.5, 'NE': 45., 'ENE': 67.5, 'E': 90., 'ESE': 112.5, 'SE': 135., 'SSE': 157.5,
            'S': 180., 'SSW': 202.5, 'SW': 225., 'WSW': 247.5, 'W': 270., 'WNW': 292.5, 'NW': 315., 'NNW': 337.5}
# ways a signal or camera node controls traffic on (a footway's own signal node is not a road's)
ROADS = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential',
         'living_street', 'service', 'road', 'motorway_link', 'trunk_link', 'primary_link',
         'secondary_link', 'tertiary_link'}


def point_kind(tags):
    """SIGNAL, CAMERA or None for an OSM node's tags."""
    highway = tags.get('highway')
    if highway == 'traffic_signals':
        return SIGNAL
    if highway == 'speed_camera':
        return CAMERA
    return None


def relative(tags, kind):
    """'forward' or 'backward' when the node's direction is relative to its way, else None."""
    keys = ('traffic_signals:direction', 'direction') if kind == SIGNAL else ('direction',)
    for key in keys:
        value = str(tags.get(key, '')).strip().lower()
        if value in ('forward', 'backward'):
            return value
    return None


def heading(tags):
    """The absolute heading in a node's tags (HEADING, or a camera's degrees or cardinal direction), else None."""
    for key in (HEADING, 'direction'):
        value = str(tags.get(key, '')).strip()
        if not value:
            continue
        if value.upper() in CARDINAL:
            return CARDINAL[value.upper()]
        try:
            degrees = float(value)
        except ValueError:
            continue
        if math.isfinite(degrees):
            return degrees % 360.
    return None


def bearing(a, b):
    """Degrees clockwise from north from (lon, lat) a to b (a local flat approximation: metres apart)."""
    dx = (b[0] - a[0]) * math.cos(math.radians((a[1] + b[1]) / 2))
    dy = b[1] - a[1]
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dx, dy)) % 360.


class Tracker:
    """Collects what fast_sources' osmium reader sees (nodes, then ways, then relations) and resolves it.

    node(nid, tags, kind)        a point node (point_kind already decided)
    way(tags, nodes)             a way: nodes are (ref, (lon, lat) or None) in way order
    relation(tags, members)      a relation: members are (type, ref, role)
    wanted()                     node ids whose locations the reader still has to look up
    finish(points, located)      sets HEADING in points {nid: tags} and returns new camera points
                                 [(nid, tags, (lon, lat))]; located is {nid: (lon, lat)}
    """

    def __init__(self):
        self.pending = {}      # nid -> 'forward' / 'backward', waiting for its way
        self.headings = {}     # nid -> degrees
        self.checks = []       # (relation tags, devices, froms, tos)

    def node(self, nid, tags, kind):
        if kind in (SIGNAL, CAMERA):
            side = relative(tags, kind)
            if side:
                self.pending[nid] = side

    def way(self, tags, nodes):
        if not self.pending or tags.get('highway') not in ROADS:
            return
        for k, (ref, at) in enumerate(nodes):
            side = self.pending.get(ref)
            if side is None or ref in self.headings or at is None:
                continue
            # the way's direction through the node: from the node before to the node after it
            before = nodes[k - 1][1] if k > 0 else at
            after = nodes[k + 1][1] if k + 1 < len(nodes) else at
            if before is None or after is None:
                continue
            b = bearing(before, after)
            if b is not None:
                self.headings[ref] = b if side == 'forward' else (b + 180.) % 360.

    def relation(self, tags, members):
        if tags.get('type') != 'enforcement' or tags.get('enforcement') != 'maxspeed':
            return
        roles = {'device': [], 'from': [], 'to': []}
        for kind, ref, role in members:
            if kind == 'n' and role in roles:
                roles[role].append(ref)
        if roles['device']:
            self.checks.append((dict(tags), roles['device'], roles['from'], roles['to']))

    def wanted(self, known=()):
        ids = set()
        for _tags, devices, froms, tos in self.checks:
            ids.update(devices, froms, tos)
        return ids - set(known)

    def finish(self, points, located):
        for nid, degrees in self.headings.items():
            if nid in points and heading(points[nid]) is None:
                points[nid][HEADING] = f'{degrees:.1f}'
        made = []
        for tags, devices, froms, tos in self.checks:
            start = next((located[n] for n in froms if n in located), None)
            end = next((located[n] for n in tos if n in located), None)
            for device in devices:
                at = located.get(device)
                # the checked stretch from -> to; without one end the device stands at it
                a, z = start or at, end or at
                b = bearing(a, z) if a and z else None
                if device in points:
                    if b is not None and heading(points[device]) is None:
                        points[device][HEADING] = f'{b:.1f}'
                    continue
                if at is None:
                    continue
                camera = {'highway': CAMERA, 'enforcement': 'maxspeed'}
                if 'maxspeed' in tags:
                    camera['maxspeed'] = tags['maxspeed']
                if b is not None:
                    camera[HEADING] = f'{b:.1f}'
                points[device] = camera
                made.append((device, camera, at))
        return made


POINT = 10 ** 12   # fast_sources keys a point node -(POINT + id), apart from areas and ways


def locate(path, ids):
    """{id: (lon, lat)} of the given nodes in an OSM file (a second pass: the reader's node filter
    keeps untagged nodes such as an enforcement's from and to out of Python)."""
    import osmium
    found = {}

    class Nodes(osmium.SimpleHandler):
        def node(self, n):
            if n.id in ids and n.location.valid():
                found[n.id] = (n.location.lon, n.location.lat)

    Nodes().apply_file(path)
    return found


def finish(tracker, records, path):
    """Resolve one reader pass over the OSM file at path. records: fast_sources' {key: (kind, tags, geometry)};
    the points' tags take HEADING in place, and devices that were no point become camera points."""
    from shapely.geometry import Point
    points, located = {}, {}
    for key, (kind, tags, geometry) in records.items():
        if key <= -POINT and kind in (SIGNAL, CAMERA):
            nid = -key - POINT
            points[nid], located[nid] = tags, (geometry.x, geometry.y)
    wanted = tracker.wanted(located)
    if wanted:
        located.update(locate(path, wanted))
    for nid, tags, at in tracker.finish(points, located):
        records[-(POINT + nid)] = (CAMERA, tags, Point(*at))
