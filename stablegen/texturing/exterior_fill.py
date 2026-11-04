"""Which faces of a mesh can be seen from outside, by flood-filling empty voxels from the bounding box's border.

TRELLIS.2 meshes are double-walled: under the outer skin runs a parallel copy of the surface a few voxels inside, and
closed objects keep sealed interiors (a car's seats and inner panels). A face is exterior when the empty space next
to it connects to the border of the grid. Unlike rays cast from a ring of viewpoints, this reaches every concave
area wider than about two voxels (armpits, wheel wells, the gap between legs), and it never reaches a cavity
that only a thin slit opens into.

Generated surfaces are rarely sealed: hairline holes let the flood into most TRELLIS.2 interiors. Closing the
surface by a few voxels keeps it out, but also fills narrow visible gaps (between roof shingles, petals, slats), so
an exact visibility pass (a rasterized ID buffer from many directions) keeps every face that can be seen.

Pure numpy (no scipy), so it runs inside Blender.
"""

import numpy as np

# Largest number of voxels along the longest side. Memory is about 6 bytes per voxel.
DEFAULT_RESOLUTION = 320
_MAX_PASSES = 64


def _grid(vertices, resolution, pad=2):
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    pitch = max(float((hi - lo).max()), 1.0e-9) / (resolution - 2 * pad)
    origin = lo - pad * pitch
    shape = tuple(int(n) for n in np.ceil((hi - origin) / pitch).astype(np.int64) + pad + 1)
    return origin, pitch, shape


def _surface_occupancy(vertices, triangles, origin, pitch, shape):
    """Voxels the surface passes through. Each triangle is sampled on a barycentric grid finer than half a voxel, so
    a closed surface leaves no gap a flood fill could leak through."""
    occupied = np.zeros(shape, dtype=bool)
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    longest = np.maximum(np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1)),
                         np.linalg.norm(a - c, axis=1))
    steps = np.maximum(1, np.ceil(longest / (0.5 * pitch))).astype(np.int64)
    limit = np.array(shape) - 1
    for step in np.unique(steps):
        chosen = np.nonzero(steps == step)[0]
        i, j = np.meshgrid(np.arange(step + 1), np.arange(step + 1), indexing='ij')
        keep = (i + j) <= step
        u = (i[keep] / step).astype(np.float32)
        v = (j[keep] / step).astype(np.float32)
        # Bound the temporary point array to a few million points per batch.
        batch = max(1, 4_000_000 // len(u))
        for start in range(0, len(chosen), batch):
            sel = chosen[start:start + batch]
            pa, pb, pc = a[sel], b[sel], c[sel]
            points = (pa[:, None, :] * (1.0 - u - v)[None, :, None] + pb[:, None, :] * u[None, :, None]
                      + pc[:, None, :] * v[None, :, None]).reshape(-1, 3)
            index = np.floor((points - origin) / pitch).astype(np.int64)
            np.clip(index, 0, limit, out=index)
            occupied[index[:, 0], index[:, 1], index[:, 2]] = True
    return occupied


def _flood_exterior(occupied):
    """Empty voxels connected (6-neighbourhood) to the grid border. Alternating sweeps along each axis spread the
    exterior through whole runs of empty voxels at once, so a few passes usually converge."""
    free = ~occupied
    exterior = np.zeros_like(occupied)
    exterior[0] = exterior[-1] = True
    exterior[:, 0] = exterior[:, -1] = True
    exterior[:, :, 0] = exterior[:, :, -1] = True
    exterior &= free
    total = int(exterior.sum())
    for _ in range(_MAX_PASSES):
        for axis in range(3):
            occ_view = np.moveaxis(occupied, axis, -1)
            ext_view = np.moveaxis(exterior, axis, -1)
            free_view = np.moveaxis(free, axis, -1)
            rows, cols, length = occ_view.shape
            chunk = max(1, 8_000_000 // (cols * length))
            for start in range(0, rows, chunk):
                stop = min(rows, start + chunk)
                occ = occ_view[start:stop]
                # Runs of empty voxels along each line: a new run starts after every occupied voxel.
                run = np.cumsum(occ, axis=-1, dtype=np.int32)
                line = np.arange((stop - start) * cols, dtype=np.int64).reshape(stop - start, cols, 1)
                ids = line * (length + 1) + run
                touched = np.zeros((stop - start) * cols * (length + 1), dtype=bool)
                touched[ids[ext_view[start:stop]]] = True
                ext_view[start:stop] = free_view[start:stop] & touched[ids]
        new_total = int(exterior.sum())
        if new_total == total:
            break
        total = new_total
    return exterior


def _dilate(mask):
    """One voxel of 6-neighbourhood dilation."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def _face_adjacency(triangles):
    """Pairs of triangles that share an edge (manifold edges only)."""
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(triangles)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    # An edge shared by exactly two triangles: equal to its successor but not to the entry before it or after next.
    start = np.nonzero(same)[0]
    if len(start) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    before = np.zeros(len(start), dtype=bool)
    before[start > 0] = same[start[start > 0] - 1]
    after = np.zeros(len(start), dtype=bool)
    inside = start + 1 < len(same)
    after[inside] = same[start[inside] + 1]
    pairs = start[~before & ~after]
    return np.stack([owner[pairs], owner[pairs + 1]], axis=1)


def _smooth_labels(labels, adjacency, iterations=2):
    """Majority vote over each triangle and its edge neighbours, which removes speckles of one or two triangles."""
    if len(adjacency) == 0:
        return labels
    labels = labels.copy()
    neighbours = np.bincount(adjacency.ravel(), minlength=len(labels)).astype(np.int32)
    for _ in range(iterations):
        votes = labels.astype(np.int32)
        np.add.at(votes, adjacency[:, 0], labels[adjacency[:, 1]])
        np.add.at(votes, adjacency[:, 1], labels[adjacency[:, 0]])
        total = neighbours + 1
        flipped = np.where(votes * 2 > total, True, np.where(votes * 2 < total, False, labels))
        if np.array_equal(flipped, labels):
            break
        labels = flipped
    return labels


def _components(count, pairs):
    """Connected-component root per element for undirected pairs (vectorized union-find: hook every pair's larger
    root under the smaller, then compress paths, until no pair spans two roots)."""
    parent = np.arange(count)
    if len(pairs) == 0:
        return parent
    while True:
        a = parent[pairs[:, 0]]
        b = parent[pairs[:, 1]]
        spanning = a != b
        if not spanning.any():
            return parent
        np.minimum.at(parent, np.maximum(a, b)[spanning], np.minimum(a, b)[spanning])
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped


def _flip_small_regions(labels, adjacency, areas, max_area, flip_value):
    """Flip regions labelled flip_value whose area is at most max_area and that border the other label."""
    same = labels[adjacency[:, 0]] == labels[adjacency[:, 1]]
    root = _components(len(labels), adjacency[same])
    region_area = np.bincount(root, weights=areas, minlength=len(labels))
    borders = np.zeros(len(labels), dtype=bool)
    cross = adjacency[~same]
    borders[root[cross[:, 0]]] = True
    borders[root[cross[:, 1]]] = True
    small = (labels == flip_value) & borders[root] & (region_area[root] <= max_area)
    return np.where(small, not flip_value, labels)


def _clean_regions(labels, adjacency, areas, max_area):
    """First remove small kept patches inside removed geometry (specks of the inner wall), so the inner wall becomes
    one region again; then keep small removed patches inside the kept skin (they would become holes to cap). A
    region that touches only its own label, such as a separate sealed part, keeps its label."""
    if len(adjacency) == 0:
        return labels
    labels = _flip_small_regions(labels, adjacency, areas, max_area, True)
    return _flip_small_regions(labels, adjacency, areas, max_area, False)


def _reach_back(exterior, free, steps):
    """Grow the exterior through free voxels for a few steps (a geodesic dilation)."""
    for _ in range(steps):
        exterior = _dilate(exterior) & free
    return exterior


def _sphere_directions(count):
    i = np.arange(count) + 0.5
    phi = np.arccos(1 - 2 * i / count)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


RASTER_IMAGE_SIZE = 512
_RASTER_CHUNK = 4_000_000


def _nearest_triangles(px, py, depth, triangles, size):
    """Orthographic ID buffer: for each pixel centre the nearest triangle covering it (-1 where none does).
    px, py are vertex positions in pixels, depth grows away from the viewer."""
    best_depth = np.full(size * size, np.inf)
    best_id = np.full(size * size, -1, dtype=np.int64)
    x, y = px[triangles], py[triangles]
    # Pixel centres sit at i + 0.5: the index range each triangle's bounding box covers.
    x_lo = np.clip(np.ceil(x.min(axis=1) - 0.5), 0, size).astype(np.int64)
    x_hi = np.clip(np.floor(x.max(axis=1) - 0.5), -1, size - 1).astype(np.int64)
    y_lo = np.clip(np.ceil(y.min(axis=1) - 0.5), 0, size).astype(np.int64)
    y_hi = np.clip(np.floor(y.max(axis=1) - 0.5), -1, size - 1).astype(np.int64)
    nx = np.maximum(x_hi - x_lo + 1, 0)
    ny = np.maximum(y_hi - y_lo + 1, 0)
    area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    count = np.where(np.abs(area) > 1e-12, nx * ny, 0)
    candidates = np.nonzero(count)[0]
    # Barycentric weights and depth are affine in the pixel position: w = a + b * cx + c * cy per triangle.
    tri = candidates
    inv = 1.0 / area[tri]
    x0, y0 = x[tri, 0], y[tri, 0]
    ex1, ey1 = x[tri, 1] - x0, y[tri, 1] - y0
    ex2, ey2 = x[tri, 2] - x0, y[tri, 2] - y0
    w1 = np.stack([(-x0 * ey2 + ex2 * y0) * inv, ey2 * inv, -ex2 * inv], axis=1)
    w2 = np.stack([(x0 * ey1 - ex1 * y0) * inv, -ey1 * inv, ex1 * inv], axis=1)
    d = depth[triangles[tri]]
    dz = w1 * (d[:, 1] - d[:, 0])[:, None] + w2 * (d[:, 2] - d[:, 0])[:, None]
    dz[:, 0] += d[:, 0]
    per_all = count[tri]
    ends = np.cumsum(per_all)
    start = 0
    while start < len(tri):
        # Triangles in chunks of about _RASTER_CHUNK candidate pixels.
        stop = max(start + 1, int(np.searchsorted(ends, (ends[start - 1] if start else 0) + _RASTER_CHUNK)))
        per = per_all[start:stop]
        owner = np.repeat(np.arange(start, stop), per)
        local = np.arange(len(owner)) - np.repeat(np.cumsum(per) - per, per)
        t = tri[owner]
        ix = x_lo[t] + local % nx[t]
        iy = y_lo[t] + local // nx[t]
        cx, cy = ix + 0.5, iy + 0.5
        a = w1[owner]
        b = w2[owner]
        u = a[:, 0] + a[:, 1] * cx + a[:, 2] * cy
        v = b[:, 0] + b[:, 1] * cx + b[:, 2] * cy
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
        c = dz[owner[inside]]
        z = c[:, 0] + c[:, 1] * cx[inside] + c[:, 2] * cy[inside]
        pixel = iy[inside] * size + ix[inside]
        np.minimum.at(best_depth, pixel, z)
        won = z == best_depth[pixel]
        best_id[pixel[won]] = t[inside][won]
        start = stop
    return best_id


def visible_triangles(vertices, triangles, directions=64, image_size=RASTER_IMAGE_SIZE, floor_axis=None):
    """Boolean per triangle: nearest to the viewer at some pixel centre in at least one of `directions`
    orthographic views spread over the sphere (a rasterized ID buffer, so thin strips and large low-poly
    triangles count exactly, whatever way their normals point). floor_axis stands the object on an opaque floor
    that covers its footprint a hair below its lowest point along that axis: nothing is seen through an opening in
    the underside, while low views past the footprint's edge still see overhangs from below."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    count = len(triangles)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-12)
    if floor_axis is not None:
        a, b = [axis for axis in range(3) if axis != floor_axis]
        quad = np.zeros((4, 3))
        quad[:, floor_axis] = lo[floor_axis] - 1e-3 * radius
        quad[:, a] = [lo[a], hi[a], hi[a], lo[a]]
        quad[:, b] = [lo[b], lo[b], hi[b], hi[b]]
        triangles = np.concatenate([triangles, len(vertices) + np.array([[0, 1, 2], [0, 2, 3]])])
        vertices = np.concatenate([vertices, quad])
    points = (vertices - (lo + hi) / 2) / radius
    size = int(image_size)
    visible = np.zeros(len(triangles), dtype=bool)
    for direction in _sphere_directions(int(directions)):
        helper = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(direction, helper)
        u /= np.linalg.norm(u)
        v = np.cross(direction, u)
        ids = _nearest_triangles((points @ u + 1) * 0.5 * size, (points @ v + 1) * 0.5 * size,
                                 -(points @ direction), triangles, size)
        visible[ids[ids >= 0]] = True
    return visible[:count]


def _grow(labels, adjacency, rings, allowed):
    """Add the edge neighbours of labelled triangles that are `allowed`, `rings` times."""
    for _ in range(int(rings)):
        touched = labels[adjacency[:, 0]] | labels[adjacency[:, 1]]
        grown = np.zeros_like(labels)
        grown[adjacency[touched].ravel()] = True
        labels = labels | (grown & allowed)
    return labels


# Rings of neighbours added to the triangles the view pass sees, among triangles under VISIBLE_GROW_PIXELS pixels
# of area: one that small can miss every pixel centre although its neighbours are seen.
VISIBLE_GROW = 2
VISIBLE_GROW_PIXELS = 4.0


def exterior_triangles(vertices, triangles, resolution=DEFAULT_RESOLUTION, smooth=True, clean_area=0.002,
                       seal_bottom=False, up_axis=2, close_holes=0, view_directions=0, visible=None):
    """Boolean per triangle: whether empty space next to it connects to the outside.

    vertices: (N, 3) float array; triangles: (F, 3) int array. clean_area is the largest region, as a fraction of
    the total surface area, that _clean_regions may flip (0 disables it). seal_bottom puts a floor under the object
    (the plane of its lowest point, along up_axis) before the flood, so an opening in the underside does not let
    outside space into a hollow object: its inside surface is then interior. close_holes (voxels) thickens the
    surface before the flood, so holes up to about twice that wide do not let it in, then lets outside space back
    to the real surface through the same number of free voxels. view_directions also keeps every triangle visible
    from that many directions (visible_triangles, standing on the floor when seal_bottom is set) and its small
    neighbours, so gap closing cannot cut into surfaces you can see; `visible` passes that mask in precomputed.
    Returns (mask, info) where info reports the grid."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if len(triangles) == 0:
        return np.zeros(0, dtype=bool), {}
    origin, pitch, shape = _grid(vertices, int(resolution), pad=2 + int(close_holes))
    occupied = _surface_occupancy(vertices, triangles, origin, pitch, shape)
    if visible is None and view_directions:
        visible = visible_triangles(vertices, triangles, directions=view_directions,
                                    floor_axis=up_axis if seal_bottom else None)
    if seal_bottom:
        floor = int(np.floor((vertices[:, up_axis].min() - origin[up_axis]) / pitch))
        index = [slice(None)] * 3
        index[up_axis] = floor
        occupied[tuple(index)] = True
    blocked = occupied
    for _ in range(int(close_holes)):
        blocked = _dilate(blocked)
    exterior = _flood_exterior(blocked)
    if close_holes:
        exterior = _reach_back(exterior, ~occupied, int(close_holes))
    near_exterior = _dilate(exterior)
    corners = vertices[triangles]
    limit = np.array(shape) - 1
    mask = np.zeros(len(triangles), dtype=bool)
    # A triangle is exterior when its centre or a point toward any corner sits next to an exterior voxel.
    centre = corners.mean(axis=1)
    for point in (centre, *(0.5 * (centre + corners[:, k]) for k in range(3))):
        index = np.clip(np.floor((point - origin) / pitch).astype(np.int64), 0, limit)
        mask |= near_exterior[index[:, 0], index[:, 1], index[:, 2]]
    # Neighbours by position: a mesh split at UV seams or into separate triangles still has its shared edges.
    welded = np.unique(vertices, axis=0, return_inverse=True)[1].reshape(-1)[triangles]
    adjacency = _face_adjacency(welded) if smooth or clean_area or visible is not None else None
    areas = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    max_area = clean_area * float(areas.sum())
    if smooth:
        mask = _smooth_labels(mask, adjacency)
    if clean_area:
        mask = _clean_regions(mask, adjacency, areas, max_area)
    if visible is not None:
        # After the cleanup, so neither the majority vote nor the speck removal takes a seen triangle away.
        pixel = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) / RASTER_IMAGE_SIZE
        small = areas < VISIBLE_GROW_PIXELS * pixel * pixel
        mask |= _grow(np.asarray(visible, dtype=bool), adjacency, VISIBLE_GROW, small)
        if clean_area:
            mask = _flip_small_regions(mask, adjacency, areas, max_area, False)
    info = {'pitch': pitch, 'shape': shape, 'occupied': int(occupied.sum()), 'exterior': int(exterior.sum())}
    return mask, info
