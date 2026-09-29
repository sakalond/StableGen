"""Joint surface weights. Pure NumPy solver plus a Blender POINT-field adapter.

The field is continuous over mesh triangles, rather than a succession of
independently normalized camera subsets. Sampling accuracy is mesh dependent.
"""
import heapq
import json
import math
import numpy as np


def normalize(values):
    return np.divide(values, values.sum(axis=1, keepdims=True),
                     out=np.zeros_like(values), where=values.sum(axis=1, keepdims=True) > 0)


def simplex(values, allowed):
    """Euclidean projection onto each vertex's valid-camera probability simplex."""
    count = allowed.sum(axis=1)
    ordered = np.sort(np.where(allowed, values, -1e30), axis=1)[:, ::-1]
    sums = np.cumsum(ordered, axis=1) - 1
    ranks = np.arange(1, values.shape[1] + 1)
    positive = (ordered - sums / ranks > 0) & (ranks <= count[:, None])
    rho = np.maximum(positive.sum(axis=1), 1)
    theta = sums[np.arange(len(values)), rho - 1] / rho
    result = np.where(allowed, np.maximum(values - theta[:, None], 0), 0)
    result[count == 0] = 0
    return result


def transition_band(positions, edges, allowed, radius):
    """Geodesic distance from a camera-support boundary, in feather widths."""
    n = len(positions)
    if len(edges) == 0:
        return np.zeros(n, bool)
    a, b = edges.T
    lengths = np.linalg.norm(positions[a] - positions[b], axis=1)
    cost = lengths / np.maximum((radius[a] + radius[b]) * .5, 1e-12)
    boundary = np.any(allowed[a] != allowed[b], axis=1)
    distance = np.full(n, np.inf)
    adjacency = [[] for _ in range(n)]
    for i, j, c in zip(a, b, cost):
        adjacency[i].append((j, c)); adjacency[j].append((i, c))
    seeds = np.unique(edges[boundary])
    distance[seeds] = 0
    queue = [(0., int(i)) for i in seeds]
    heapq.heapify(queue)
    while queue:
        value, i = heapq.heappop(queue)
        if value != distance[i] or value >= 1:
            continue
        for j, c in adjacency[i]:
            candidate = value + c
            if candidate < min(distance[j], 1.):
                distance[j] = candidate
                heapq.heappush(queue, (candidate, int(j)))
    return distance < 1


def solve(positions, edges, quality, guard, radius, exponent, rescue=.25,
          tolerance=1e-7, max_iterations=2000):
    """Minimize neighbor disagreement subject to visibility and sum-to-one.

    Preferences are fixed outside a geodesic band. Within it all camera
    contributions are solved together, with invalid components fixed at zero.
    Conservative guards may be relaxed for coverage; genuine visibility never is.
    """
    quality = np.maximum(np.asarray(quality, dtype=np.float64), 0)
    visible = quality > 0
    trusted = quality * np.clip(guard, 0, 1)
    best = quality.max(axis=1)
    recover = trusted.max(axis=1) < best * max(rescue, 0)
    allowed = visible & ((guard > .5) | recover[:, None])
    # No trusted source: explicitly retain genuine coverage. Never manufacture
    # an occluded source, and never normalize a zero vector into a color.
    empty = ~allowed.any(axis=1) & visible.any(axis=1)
    allowed[empty] = visible[empty]
    a, b = np.asarray(edges, dtype=np.int64).reshape(-1, 2).T
    # A camera must already be zero on the visible side of a sampled true
    # visibility boundary; runtime raycast must not cut an interpolated weight.
    safe = allowed.copy()
    for camera in range(quality.shape[1]):
        crossing = visible[a, camera] != visible[b, camera]
        safe[np.unique(np.concatenate((a[crossing], b[crossing]))), camera] = False
    narrow = ~safe.any(axis=1) & allowed.any(axis=1)
    safe[narrow] = allowed[narrow]
    allowed = safe
    preferred = np.where(allowed, quality, 0)
    maximum = preferred.max(axis=1, keepdims=True)
    preferred = np.divide(preferred, maximum, out=np.zeros_like(preferred), where=maximum > 0)
    preferred = normalize(preferred ** max(float(exponent), .001))
    band = transition_band(positions, edges, allowed, radius)
    active = band & (allowed.sum(axis=1) > 1)
    weights = preferred.copy()
    # Symmetric positive edge weights yield a convex Dirichlet energy. Do not
    # link separate objects or disconnected mesh components by spatial proximity.
    length = np.linalg.norm(positions[a] - positions[b], axis=1)
    conductance = 1 / np.maximum(length, 1e-9)
    src = np.concatenate((a, b)); dst = np.concatenate((b, a))
    conductance = np.concatenate((conductance, conductance))
    # Uncovered vertices cannot act as artificial black boundary conditions.
    covered = allowed.any(axis=1)
    conductance *= covered[src] & covered[dst]
    degree = np.bincount(src, weights=conductance, minlength=len(positions))
    active &= degree > 0
    change = 0.; iteration = 0
    if active.any():
        extrapolated = weights.copy(); momentum = 1.; settled = 0
        for iteration in range(1, max_iterations + 1):
            average = np.column_stack([
                np.bincount(src, weights=conductance * extrapolated[dst, c], minlength=len(positions))
                for c in range(quality.shape[1])])
            average /= np.maximum(degree[:, None], 1e-12)
            candidate = simplex(.5*(extrapolated[active]+average[active]), allowed[active])
            change = float(np.max(np.abs(candidate - weights[active])))
            # Accelerated projected gradient in the degree-weighted metric.
            # A half step bounds the normalized Laplacian spectrum by one.
            next_momentum = (1+math.sqrt(1+4*momentum*momentum))*.5
            delta = candidate-weights[active]
            restart = np.sum(degree[active,None]*(extrapolated[active]-candidate)*delta) > 0
            if restart:
                next_momentum = 1.
                extrapolated[active] = candidate
            else:
                extrapolated[active] = candidate+(momentum-1)/next_momentum*delta
            weights[active] = candidate; momentum = next_momentum
            settled = settled+1 if change < tolerance else 0
            if settled >= 8:
                break
    return weights, {'iterations': iteration, 'residual': change,
                     'converged': change < tolerance, 'band_vertices': int(active.sum()),
                     'uncovered_vertices': int((~covered).sum()),
                     'narrow_coverage_vertices': int(narrow.sum()),
                     'recovered_vertices': int((recover | empty).sum())}


def _attribute(mesh, name, values=None, vector=False):
    attr = mesh.attributes.get(name)
    if values is None:
        if attr is None:
            raise RuntimeError('Surface Blend inputs are missing; Reproject to rebuild them')
        data = np.empty((len(mesh.vertices), 3) if vector else len(mesh.vertices), np.float32)
        attr.data.foreach_get('vector' if vector else 'value', data.ravel())
        return data
    if attr is None:
        attr = mesh.attributes.new(name, 'FLOAT_VECTOR' if vector else 'FLOAT', 'POINT')
    attr.data.foreach_set('vector' if vector else 'value', np.asarray(values, np.float32).ravel())


def validate(context, objects, cameras):
    """Reject unsupported inputs before replacing any projection material."""
    if any(c.data.type not in {'PERSP','ORTHO'} for c in cameras):
        raise ValueError('Surface Blend preview supports perspective and orthographic cameras')
    depsgraph = context.evaluated_depsgraph_get(); seen = set()
    for obj in objects:
        identity = obj.data.as_pointer()
        if identity in seen:
            raise ValueError('Surface Blend preview requires single-user mesh data; make linked meshes single-user first')
        seen.add(identity)
        evaluated = obj.evaluated_get(depsgraph); mesh = evaluated.to_mesh()
        try:
            if len(mesh.vertices) != len(obj.data.vertices):
                raise ValueError(f'Surface Blend: apply topology-changing modifiers on {obj.name} before Reproject')
        finally:
            evaluated.to_mesh_clear()


def update(material, angle, exponent):
    """Update cached preference/angle controls without repeating raycasts."""
    import bpy
    metadata = json.loads(material['sg_surface_blend'])
    if metadata.get('angle') == angle and metadata.get('exponent') == exponent:
        return
    reports = []
    for name in metadata['objects']:
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise RuntimeError('Surface Blend mesh changed; Reproject to rebuild its field')
        mesh = obj.data; key = metadata['key']; count = metadata['cameras']
        positions = _attribute(mesh, key + 'Position', vector=True)
        edges = np.empty((len(mesh.edges), 2), np.int32)
        mesh.edges.foreach_get('vertices', edges.ravel())
        quality = np.column_stack([_attribute(mesh, key + f'Quality{i}') for i in range(count)])
        cosines = np.column_stack([_attribute(mesh, key + f'Cosine{i}') for i in range(count)])
        quality[cosines <= math.cos(math.radians(min(float(angle), 90.)))] = 0
        guards = np.column_stack([_attribute(mesh, key + f'Guard{i}') for i in range(count)])
        weights, report = solve(positions, edges, quality, guards,
                                _attribute(mesh, key + 'Radius'), exponent, metadata['rescue'])
        for i in range(count):
            _attribute(mesh, key + f'Weight{i}', weights[:, i])
        mesh.update(); reports.append({'object': name, **report})
    metadata.update(angle=angle, exponent=exponent, reports=reports)
    material['sg_surface_blend'] = json.dumps(metadata)
    print('[StableGen] Surface Blend:', reports)


def build(context, objects, cameras, images, material, mat_id, stop_index):
    """Sample evaluated geometry and projection masks; cache a surface field."""
    import bpy
    import cv2
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from .projection import _is_projection_mesh, _scene_ray_epsilon, _scene_projection_scale
    from ..utils import get_dir_path
    from ..cameras.geometry import _get_camera_resolution
    from pathlib import Path
    scene = context.scene; depsgraph = context.evaluated_depsgraph_get()
    vertices = []; polygons = []
    for ob in context.view_layer.objects:
        if not _is_projection_mesh(ob):
            continue
        evaluated = ob.evaluated_get(depsgraph); mesh = evaluated.to_mesh()
        try:
            offset = len(vertices)
            vertices.extend(evaluated.matrix_world @ v.co for v in mesh.vertices)
            polygons.extend(tuple(offset + i for i in p.vertices) for p in mesh.polygons)
        finally:
            evaluated.to_mesh_clear()
    tree = BVHTree.FromPolygons(vertices, polygons)
    epsilon = _scene_ray_epsilon(context); scale = _scene_projection_scale(context)
    key = f'_SG_Surface{mat_id}_'; paths = Path(get_dir_path(context, 'inpaint')['visibility'])
    cache = {}
    def sample(path, uv, default=1.):
        if path not in cache:
            cache[path] = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if Path(path).is_file() else None
        image = cache[path]
        if image is None:
            return np.full(len(uv), default)
        x = np.clip(uv[:, 0] * image.shape[1] - .5, 0, image.shape[1]-1)
        y = np.clip((1-uv[:, 1]) * image.shape[0] - .5, 0, image.shape[0]-1)
        ix = x.astype(int); iy = y.astype(int); fx = x-ix; fy = y-iy
        if image.ndim == 3:
            # This adapter samples alpha from source images, scalar from masks.
            image = image[:, :, 3] if image.shape[2] == 4 else np.full(image.shape[:2], 255, np.uint8)
        value = ((1-fy)*((1-fx)*image[iy, ix]+fx*image[iy, np.minimum(ix+1,image.shape[1]-1)])
                 +fy*((1-fx)*image[np.minimum(iy+1,image.shape[0]-1),ix]
                       +fx*image[np.minimum(iy+1,image.shape[0]-1),np.minimum(ix+1,image.shape[1]-1)]))
        return value / (65535 if image.dtype == np.uint16 else 255)
    names = []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph); mesh = evaluated.to_mesh()
        try:
            if len(mesh.vertices) != len(obj.data.vertices):
                raise RuntimeError(f'Surface Blend: {obj.name} has topology-changing modifiers; apply them before Reproject')
            positions = np.array([evaluated.matrix_world @ v.co for v in mesh.vertices])
            normal_matrix = evaluated.matrix_world.to_3x3().inverted().transposed()
            normals = np.array([(normal_matrix @ v.normal).normalized() for v in mesh.vertices])
        finally:
            evaluated.to_mesh_clear()
        mesh = obj.data; n = len(mesh.vertices)
        if n == 0:
            continue
        names.append(obj.name); _attribute(mesh, key+'Position', positions, vector=True)
        widths = []
        loops = np.empty(len(mesh.loops), np.int32); mesh.loops.foreach_get('vertex_index', loops)
        loop_counts = np.maximum(np.bincount(loops, minlength=n), 1)
        for i, camera in enumerate(cameras):
            uv_attr = mesh.attributes.get(f'ProjectionUV_{i}_{mat_id}')
            if uv_attr is None:
                raise RuntimeError('Surface Blend requires projection coordinates; run Reproject')
            coords = np.empty((len(uv_attr.data), 3), np.float32); uv_attr.data.foreach_get('vector', coords.ravel())
            uv = np.column_stack([np.bincount(loops, weights=coords[:, c], minlength=n)/loop_counts for c in range(2)]) if uv_attr.domain == 'CORNER' else coords[:, :2]
            location = np.array(camera.matrix_world.translation)
            delta = positions-location; distance = np.linalg.norm(delta, axis=1)
            forward = np.array(camera.matrix_world.to_quaternion() @ Vector((0,0,-1)))
            depth = delta @ forward
            direction = delta / np.maximum(distance[:, None], 1e-12)
            if camera.data.type == 'ORTHO':
                direction[:] = forward; distance = depth
            dot = -np.sum(normals*direction, axis=1)
            cosine = np.abs(dot) if scene.sg_allow_backface_projection else np.maximum(dot, 0)
            visible = (depth > 0) & np.all((uv >= 0) & (uv <= 1), axis=1)
            if i > stop_index or images[i].image is None or images[i].image.size[0] == 0:
                visible[:] = False
            for index in np.flatnonzero(visible):
                origin = location if camera.data.type != 'ORTHO' else positions[index]-forward*distance[index]
                hit, _, _, hit_distance = tree.ray_cast(Vector(origin),Vector(direction[index]),float(distance[index])+epsilon*10)
                visible[index] = hit is not None and abs(hit_distance-distance[index]) <= epsilon*2
            x, y = _get_camera_resolution(camera, scene)
            fov = camera.data.angle_x
            if y > x: fov = 2*math.atan(math.tan(fov/2)*x/y)
            world_pixel = (np.full(n,camera.data.ortho_scale/y) if camera.data.type == 'ORTHO'
                           else 2*np.maximum(depth,epsilon)*math.tan(fov/2)/x)
            widths.append(world_pixel * scene.sg_silhouette_feather)
            quality = cosine * visible
            if scene.sg_rework_acquisition_quality:
                quality *= np.clip(scale/np.maximum(world_pixel,epsilon)/(max(.5*min(x,y),1)),0,1)
            if scene.visibility_vignette:
                w = max(min(scene.visibility_vignette_width,.49),1e-9)
                edge = np.max(np.abs(uv*2-1),axis=1)
                t = np.clip((edge-(1-w))/w,0,1)
                quality *= (1-t*t*(3-2*t)) ** max(scene.visibility_vignette_softness,.01)
            if images[i].image is not None:
                quality *= sample(bpy.path.abspath(images[i].image.filepath),uv)
            _attribute(mesh,key+f'Quality{i}',quality)
            _attribute(mesh,key+f'Cosine{i}',cosine)
            _attribute(mesh,key+f'Guard{i}',sample(paths/f'projection_guard{i}.png',uv))
        _attribute(mesh,key+'Radius',np.median(widths,axis=0))
    material['sg_surface_blend'] = json.dumps({'version':1,'objects':names,'key':key,
                                              'cameras':len(cameras),'rescue':scene.sg_guard_coverage_rescue})
    update(material,scene.discard_factor,scene.weight_exponent)
    return key
