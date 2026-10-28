"""Estimate Cycles SVM stack use of a material before Cycles renders it black.

Cycles syncs Blender's inlined copy of the tree (depth-first from the output)
and compiles an SVM shader by sweeping those nodes in order; a node compiles
in the first sweep in which all its inputs are ready. A value occupies the
255-float stack until its last consumer has compiled. When the stack
overflows, Cycles only logs "out of SVM stack space" and the material renders
black, so StableGen checks its own projection graphs.

The model mirrors the parts of Cycles that decide the stack use: implicit
socket conversions and expanded clamps (created after all nodes, so their
consumers wait for the next sweep), constant folding, deduplication, the
allocation order inside nodes, and nodes such as Raycast whose compile()
assigns every output even when it is unused. It reproduces the peak usage
Cycles 5.1 logs exactly for StableGen's projection materials.
"""

import bpy


SVM_STACK_SIZE = 255

_SIZE = {'FLOAT': 1, 'INT': 1, 'COLOR': 3, 'VECTOR': 3, 'POINT': 3, 'NORMAL': 3,
         'CLOSURE': 0}
_BLENDER_TYPES = {'VALUE': 'FLOAT', 'INT': 'INT', 'BOOLEAN': 'INT', 'RGBA': 'COLOR',
                  'VECTOR': 'VECTOR', 'SHADER': 'CLOSURE'}

# Cycles socket types that differ from the Blender socket type.
_INPUT_TYPES = {
    'ShaderNodeTexImage': {'Vector': 'POINT'},
    'ShaderNodeTexGradient': {'Vector': 'POINT'},
    'ShaderNodeTexNoise': {'Vector': 'POINT'},
    'ShaderNodeTexVoronoi': {'Vector': 'POINT'},
    'ShaderNodeRaycast': {'Position': 'POINT', 'Direction': 'NORMAL'},
    'ShaderNodeSeparateXYZ': {'Vector': 'COLOR'},
    'ShaderNodeFresnel': {'Normal': 'NORMAL'},
    'ShaderNodeLayerWeight': {'Normal': 'NORMAL'},
    'ShaderNodeNormal': {'Normal': 'NORMAL'},
    'ShaderNodeBump': {'Normal': 'NORMAL'},
    'ShaderNodeMapping': {'Vector': 'POINT', 'Location': 'POINT',
                          'Rotation': 'POINT', 'Scale': 'POINT'},
}
_OUTPUT_TYPES = {
    'ShaderNodeNewGeometry': {'Position': 'POINT', 'Normal': 'NORMAL', 'Tangent': 'NORMAL',
                              'True Normal': 'NORMAL', 'Parametric': 'POINT'},
    'ShaderNodeTexCoord': {'Generated': 'POINT', 'Normal': 'NORMAL', 'UV': 'POINT',
                           'Object': 'POINT', 'Camera': 'POINT', 'Window': 'POINT',
                           'Reflection': 'NORMAL'},
    'ShaderNodeUVMap': {'UV': 'POINT'},
    'ShaderNodeRaycast': {'Hit Position': 'POINT', 'Hit Normal': 'NORMAL'},
    'ShaderNodeMapping': {'Vector': 'POINT'},
    'ShaderNodeNormal': {'Normal': 'NORMAL'},
    'ShaderNodeBump': {'Normal': 'NORMAL'},
    'ShaderNodeNormalMap': {'Normal': 'NORMAL'},
}
# compile() assigns every output, linked or not; unused ones are never freed.
_ALL_OUTPUTS_ASSIGNED = {'ShaderNodeRaycast', 'ShaderNodeSeparateXYZ',
                         'ShaderNodeSeparateColor', 'ShaderNodeNormal'}
# Inputs compile() reads only when linked (no temporary constant slot).
_LINKED_ONLY_INPUTS = {
    'ShaderNodeMapRange': {'From Min', 'From Max', 'To Min', 'To Max'},
    'ShaderNodeEmission': {'Strength'},
    'ShaderNodeFresnel': {'Normal'},
}
_ALWAYS_PRESENT_INPUTS = {
    'ShaderNodeMath': {'Value', 'Value_001', 'Value_002'},
    'ShaderNodeVectorMath': {'Vector', 'Vector_001', 'Vector_002', 'Scale'},
    'ShaderNodeMapRange': {'Value', 'From Min', 'From Max', 'To Min', 'To Max', 'Steps'},
}
_FOLDABLE = {'ShaderNodeMath', 'ShaderNodeVectorMath', 'ShaderNodeCombineXYZ',
             'ShaderNodeSeparateXYZ', 'ShaderNodeMapRange', 'ShaderNodeMixRGB',
             'ShaderNodeMix', 'ShaderNodeSeparateColor', 'ShaderNodeCombineColor',
             'ShaderNodeRGBToBW', 'ShaderNodeClamp', 'CONVERT', 'ShaderNodeValToRGB'}
_IGNORED_PROPS = {'rna_type', 'name', 'label', 'location', 'location_absolute', 'width',
                  'height', 'dimensions', 'select', 'hide', 'show_options', 'show_preview',
                  'show_texture', 'color', 'use_custom_color', 'parent', 'internal_links',
                  'inputs', 'outputs', 'mute', 'warning_propagation', 'color_tag', 'type'}


class _Node:
    __slots__ = ('id', 'kind', 'op', 'label', 'key', 'inputs', 'outputs', 'const', 'proxy')

    def __init__(self, ident, kind, op='', label='', key=()):
        self.id = ident
        self.kind = kind
        self.op = op
        self.label = label
        self.key = key
        self.inputs = []    # [name, type, link (node, output index) | None, value]
        self.outputs = []   # [name, type]
        self.const = None
        self.proxy = False

    def clamped(self):
        return ('use_clamp', True) in self.key


def _prop_key(node):
    """Unlinkable settings; Cycles only merges nodes whose settings match."""
    values = []
    for prop in node.bl_rna.properties:
        name = prop.identifier
        if name in _IGNORED_PROPS or name.startswith('bl_'):
            continue
        try:
            value = getattr(node, name)
        except AttributeError:
            continue
        if isinstance(value, bpy.types.ID):
            value = value.as_pointer()
        elif not isinstance(value, (str, int, float, bool, type(None))):
            continue
        values.append((name, value))
    ramp = getattr(node, 'color_ramp', None)
    if ramp is not None:
        values.append(('ramp', tuple((e.position, tuple(e.color)) for e in ramp.elements)))
    return tuple(values)


def _socket_value(socket):
    value = getattr(socket, 'default_value', None)
    if value is None:
        return None
    try:
        return float(value)
    except TypeError:
        try:
            return tuple(round(v, 7) for v in value)
        except TypeError:
            return None


def _evaluate(node):
    """Folded output values; unknown operations get a value equal to nothing."""
    values = [entry[3] for entry in node.inputs]
    result = {}
    try:
        if node.kind == 'ShaderNodeMath':
            result['Value'] = _math(node.op, *values[:3])
        elif node.kind == 'ShaderNodeCombineXYZ':
            result['Vector'] = tuple(values[:3])
        elif node.kind == 'ShaderNodeSeparateXYZ':
            result.update(zip(('X', 'Y', 'Z'), values[0]))
        elif node.kind == 'ShaderNodeVectorMath':
            result.update(_vector_math(node.op, *values[:4]))
        elif node.kind == 'CONVERT':
            value = values[0]
            if isinstance(value, tuple) and _SIZE[node.op[1]] == 1:
                weights = ((0.2126, 0.7152, 0.0722) if node.op[0] == 'COLOR'
                           else (1 / 3, 1 / 3, 1 / 3))
                value = sum(w * v for w, v in zip(weights, value))
            elif not isinstance(value, tuple) and _SIZE[node.op[1]] == 3:
                value = (value, value, value)
            result['Out'] = value
    except (TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        result = {}
    return [result.get(name, ('folded', node.id, name)) for name, _type in node.outputs]


def _math(op, a, b, c):
    import math
    operations = {
        'ADD': lambda: a + b, 'SUBTRACT': lambda: a - b, 'MULTIPLY': lambda: a * b,
        'DIVIDE': lambda: a / b if b != 0.0 else 0.0, 'POWER': lambda: a ** b,
        'MINIMUM': lambda: min(a, b), 'MAXIMUM': lambda: max(a, b),
        'LESS_THAN': lambda: float(a < b), 'GREATER_THAN': lambda: float(a > b),
        'ABSOLUTE': lambda: abs(a), 'TANGENT': lambda: math.tan(a),
        'ARCCOSINE': lambda: math.acos(max(-1.0, min(1.0, a))),
        'COMPARE': lambda: float(abs(a - b) <= max(c, 1e-5)),
        'MULTIPLY_ADD': lambda: a * b + c, 'SQRT': lambda: math.sqrt(a) if a > 0 else 0.0,
    }
    return float(operations[op]())


def _vector_math(op, a, b, c, scale):
    import math

    def length(v):
        return math.sqrt(sum(x * x for x in v))

    if op == 'NORMALIZE':
        n = length(a)
        return {'Vector': tuple(x / n for x in a) if n else (0.0, 0.0, 0.0)}
    if op == 'LENGTH':
        return {'Value': length(a)}
    if op == 'DOT_PRODUCT':
        return {'Value': sum(x * y for x, y in zip(a, b))}
    if op == 'SCALE':
        return {'Vector': tuple(x * scale for x in a)}
    if op == 'CROSS_PRODUCT':
        return {'Vector': (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                           a[0] * b[1] - a[1] * b[0])}
    if op in {'ADD', 'SUBTRACT', 'MULTIPLY'}:
        function = {'ADD': lambda x, y: x + y, 'SUBTRACT': lambda x, y: x - y,
                    'MULTIPLY': lambda x, y: x * y}[op]
        return {'Vector': tuple(function(x, y) for x, y in zip(a, b))}
    return {}


class _Graph:
    def __init__(self):
        self.nodes = []
        self.output = None
        self.clamped = []
        self.deduplicated = 0

    def expand(self):
        """MathNode/MapRangeNode::expand append a Clamp after every node."""
        for node in self.clamped:
            if node.const is not None:
                continue
            clamp = self.add('CLAMP', '', node.label + ' clamp')
            clamp.inputs = [['Value', 'FLOAT', None, None], ['Min', 'FLOAT', None, 0.0],
                            ['Max', 'FLOAT', None, 1.0]]
            clamp.outputs.append(['Result', 'FLOAT'])
            self._relink(node, (clamp, 0))
            clamp.inputs[0][2] = (node, 0)

    def add(self, kind, op='', label='', key=()):
        node = _Node(len(self.nodes), kind, op, label, key)
        self.nodes.append(node)
        return node

    def connect(self, source, target):
        """ShaderGraph::connect: a type mismatch adds a conversion node."""
        node, index = source
        from_type = node.outputs[index][1]
        to_type = target[1]
        if from_type != to_type and 'CLOSURE' not in (from_type, to_type):
            convert = self.add('CONVERT', (from_type, to_type), 'convert ' + node.label,
                               (from_type, to_type))
            convert.inputs.append(['In', from_type, source, None])
            convert.outputs.append(['Out', to_type])
            source = (convert, 0)
        target[2] = source

    def _proxy(self, ctype, value=None):
        proxy = self.add('PROXY')
        proxy.proxy = True
        proxy.inputs.append(['In', ctype, None, value])
        proxy.outputs.append(['Out', ctype])
        return proxy

    def add_tree(self, tree, input_proxies=None, output_proxies=None):
        """blender/shader.cpp add_nodes(): all nodes first, then all links."""
        outputs = {}
        inputs = {}
        for b_node in tree.nodes:
            kind = b_node.bl_idname
            if kind == 'NodeFrame':
                continue
            if b_node.mute or kind == 'NodeReroute':
                for link in b_node.internal_links:
                    ctype = _BLENDER_TYPES.get(link.to_socket.type)
                    if ctype is None:
                        continue
                    proxy = self._proxy(ctype, _socket_value(link.from_socket))
                    inputs.setdefault(link.from_socket.as_pointer(), []).append(proxy.inputs[0])
                    outputs[link.to_socket.as_pointer()] = (proxy, 0)
                continue
            if kind == 'ShaderNodeGroup':
                group_inputs, group_outputs = {}, {}
                for socket in b_node.inputs:
                    ctype = _BLENDER_TYPES.get(socket.type)
                    if ctype is not None:
                        proxy = self._proxy(ctype, _socket_value(socket))
                        group_inputs[socket.identifier] = proxy
                        inputs.setdefault(socket.as_pointer(), []).append(proxy.inputs[0])
                for socket in b_node.outputs:
                    ctype = _BLENDER_TYPES.get(socket.type)
                    if ctype is not None:
                        proxy = self._proxy(ctype)
                        group_outputs[socket.identifier] = proxy
                        outputs[socket.as_pointer()] = (proxy, 0)
                if b_node.node_tree is not None:
                    self.add_tree(b_node.node_tree, group_inputs, group_outputs)
                continue
            if kind == 'NodeGroupInput':
                for socket in b_node.outputs:
                    proxy = (input_proxies or {}).get(socket.identifier)
                    if proxy is not None:
                        outputs[socket.as_pointer()] = (proxy, 0)
                continue
            if kind == 'NodeGroupOutput':
                if b_node.is_active_output:
                    for socket in b_node.inputs:
                        proxy = (output_proxies or {}).get(socket.identifier)
                        if proxy is not None:
                            inputs.setdefault(socket.as_pointer(), []).append(proxy.inputs[0])
                continue
            if kind == 'ShaderNodeOutputMaterial' and (
                    input_proxies is not None or self.output is not None
                    or not b_node.is_active_output or b_node.target == 'EEVEE'):
                continue
            if kind == 'ShaderNodeValue':
                node = self.add(kind, '', b_node.label or b_node.name)
                node.const = [b_node.outputs[0].default_value]
                node.outputs.append(['Value', 'FLOAT'])
                outputs[b_node.outputs[0].as_pointer()] = (node, 0)
                continue
            op = getattr(b_node, 'operation', '') or getattr(b_node, 'blend_type', '')
            node = self.add(kind, op, b_node.label or b_node.name, _prop_key(b_node))
            if kind == 'ShaderNodeOutputMaterial':
                self.output = node
            if getattr(b_node, 'use_clamp', False) and kind == 'ShaderNodeMath' \
                    or kind == 'ShaderNodeMapRange' and b_node.clamp:
                self.clamped.append(node)
            in_types = _INPUT_TYPES.get(kind, {})
            out_types = _OUTPUT_TYPES.get(kind, {})
            always = (_ALWAYS_PRESENT_INPUTS.get(kind, ())
                      if getattr(b_node, 'data_type', 'FLOAT') == 'FLOAT' else ())
            for socket in b_node.inputs:
                # Cycles keeps inputs that Blender hides for some operations.
                if not socket.enabled and socket.identifier not in always:
                    continue
                ctype = in_types.get(socket.name) or _BLENDER_TYPES.get(socket.type)
                if ctype is None:
                    continue
                entry = [socket.name, ctype, None, _socket_value(socket)]
                node.inputs.append(entry)
                if socket.enabled:
                    inputs.setdefault(socket.as_pointer(), []).append(entry)
            for socket in b_node.outputs:
                if not socket.enabled:
                    continue
                ctype = out_types.get(socket.name) or _BLENDER_TYPES.get(socket.type)
                if ctype is not None:
                    node.outputs.append([socket.name, ctype])
                    outputs[socket.as_pointer()] = (node, len(node.outputs) - 1)
        for link in tree.links:
            # Cycles ignores invalid and muted links and unavailable sockets.
            if (not link.is_valid or link.is_muted
                    or not link.from_socket.enabled or not link.to_socket.enabled):
                continue
            source = outputs.get(link.from_socket.as_pointer())
            if source is None:
                continue
            for target in inputs.get(link.to_socket.as_pointer(), ()):
                if target[2] is None:
                    self.connect(source, target)

    def _relink(self, node, replacement):
        for other in self.nodes:
            for entry in other.inputs:
                if entry[2] is not None and entry[2][0] is node:
                    entry[2] = replacement

    def remove_proxies(self):
        for node in self.nodes:
            if not node.proxy:
                continue
            source = node.inputs[0][2]
            for other in self.nodes:
                for entry in other.inputs:
                    if entry[2] is not None and entry[2][0] is node:
                        entry[2] = source
                        if source is None:
                            entry[3] = node.inputs[0][3]
            node.const = 'removed'

    def fold(self):
        """ShaderGraph::constant_fold for the node types projection graphs use."""
        changed = True
        while changed:
            changed = False
            for node in self.nodes:
                if node.const is not None:
                    continue
                for entry in node.inputs:
                    source = entry[2]
                    if source is not None and isinstance(source[0].const, list):
                        entry[2] = None
                        entry[3] = source[0].const[source[1]]
                        changed = True
                if node.kind in _FOLDABLE and node.inputs \
                        and all(entry[2] is None for entry in node.inputs):
                    node.const = _evaluate(node)
                    changed = True
                    continue
                target = self._fold_identity(node)
                if target == 'zero':
                    node.const = [0.0]
                elif target == 'one':
                    node.const = [1.0]
                elif target is not None:
                    self._relink(node, target)
                    node.const = 'removed'
                if target is not None:
                    changed = True

    @staticmethod
    def _fold_identity(node):
        def zero(entry):
            return entry[2] is None and entry[3] in (0.0, (0.0, 0.0, 0.0))

        def one(entry):
            return entry[2] is None and entry[3] in (1.0, (1.0, 1.0, 1.0))

        def through(entry):
            if entry[2] is None:
                return None
            source, index = entry[2]
            return entry[2] if source.outputs[index][1] == node.outputs[0][1] else None

        if node.kind == 'ShaderNodeMath' and len(node.inputs) >= 2 and not node.clamped():
            a, b = node.inputs[0], node.inputs[1]
            if node.op == 'ADD':
                return through(b) if zero(a) else through(a) if zero(b) else None
            if node.op == 'SUBTRACT':
                return through(a) if zero(b) else None
            if node.op == 'MULTIPLY':
                if one(a):
                    return through(b)
                if one(b):
                    return through(a)
                return 'zero' if zero(a) or zero(b) else None
            if node.op == 'DIVIDE':
                return through(a) if one(b) else 'zero' if zero(a) else None
            if node.op == 'POWER':
                if one(a) or zero(b):
                    return 'one'
                return through(a) if one(b) else None
        if node.kind == 'CONVERT':
            source = node.inputs[0][2]
            if source is not None and source[0].kind == 'CONVERT' \
                    and source[0].op == (node.op[1], node.op[0]) \
                    and _SIZE[node.op[0]] == 3 and source[0].inputs[0][2] is not None:
                return source[0].inputs[0][2]
        return None

    def reachable(self):
        seen = set()
        pending = [self.output] if self.output is not None else []
        while pending:
            node = pending.pop()
            if node.id in seen:
                continue
            seen.add(node.id)
            pending.extend(entry[2][0] for entry in node.inputs if entry[2] is not None)
        return seen

    def deduplicate(self, alive):
        """ShaderGraph::deduplicate_nodes: merge equal settings and inputs."""
        changed = True
        while changed:
            changed = False
            seen = {}
            for node in self.nodes:
                if node.id not in alive or node is self.output:
                    continue
                signature = (node.kind, node.op, node.key, tuple(
                    (entry[0], (entry[2][0].id, entry[2][1]) if entry[2] is not None
                     else ('constant', entry[3])) for entry in node.inputs))
                keep = seen.setdefault(signature, node)
                if keep is node:
                    continue
                for other in self.nodes:
                    for entry in other.inputs:
                        if entry[2] is not None and entry[2][0] is node:
                            entry[2] = (keep, entry[2][1])
                alive.discard(node.id)
                self.deduplicated += 1
                changed = True


def _temporary_inputs(node):
    """Indices of unlinked inputs that compile() loads into temporary slots."""
    linked_only = _LINKED_ONLY_INPUTS.get(node.kind, ())
    temps = []
    for index, entry in enumerate(node.inputs):
        if entry[2] is not None or entry[0] in linked_only:
            continue
        if node.kind == 'ShaderNodeVectorMath' and index == 2 \
                and node.op not in {'WRAP', 'FACEFORWARD', 'MULTIPLY_ADD'}:
            continue
        if node.kind == 'ShaderNodeMapRange' and entry[0] not in {'Value', 'Steps'}:
            continue
        if _SIZE.get(entry[1], 0):
            temps.append(index)
    return temps


# compile() functions that allocate several sockets as arguments of a single
# call. MSVC and GCC evaluate those arguments right to left, which decides
# where each value lands on the stack and therefore its fragmentation.
_ALLOCATION_ORDER = {
    'ShaderNodeRaycast': ('out:Hit Normal', 'out:Hit Position', 'out:Hit Distance',
                          'out:Self Hit', 'out:Is Hit', 'in:Length', 'in:Direction',
                          'in:Position'),
    'ShaderNodeMixRGB': ('in:Color2', 'in:Color1', 'in:Fac', 'out:Color'),
    'ShaderNodeTexImage': ('in:Vector', 'out:Alpha', 'out:Color'),
    'ShaderNodeTexGradient': ('in:Vector', 'out:Color', 'out:Fac'),
    'ShaderNodeFresnel': ('out:Fac', 'in:Normal', 'in:IOR'),
    'ShaderNodeCombineXYZ': ('out:Vector', 'in:X', 'in:Y', 'in:Z'),
    'ShaderNodeSeparateXYZ': ('out:X', 'in:Vector', 'out:Y', 'out:Z'),
}


def _allocation_steps(node):
    """Ordered ('in', index) temporaries and ('out', index) outputs."""
    temps = _temporary_inputs(node)
    order = _ALLOCATION_ORDER.get(node.kind)
    if order is None:
        return ([('in', index) for index in temps]
                + [('out', index) for index in range(len(node.outputs))])
    steps = []
    for item in order:
        side, name = item.split(':', 1)
        if side == 'in':
            steps += [('in', index) for index in temps if node.inputs[index][0] == name]
        else:
            steps += [('out', index) for index, output in enumerate(node.outputs)
                      if output[0] == name]
    return steps


def _simulate(graph, alive, limit):
    """Replay SVMCompiler::generate_svm_nodes with its stack allocator."""
    order = [node for node in graph.nodes if node.id in alive and node is not graph.output]
    users = {}
    for node in order + [graph.output]:
        for entry in node.inputs:
            if entry[2] is not None:
                users.setdefault((entry[2][0].id, entry[2][1]), set()).add(node.id)
    stack = [0] * limit
    state = {'peak': 0, 'overflow': False, 'peak_live': ()}
    offsets = {}
    done = set()

    def allocate(size):
        run = 0
        for index in range(limit):
            run = 0 if stack[index] else run + 1
            if run == size:
                offset = index + 1 - size
                for slot in range(offset, index + 1):
                    stack[slot] = 1
                if index + 1 > state['peak']:
                    state['peak'] = index + 1
                    state['peak_live'] = tuple(offsets.values())
                return offset
        state['overflow'] = True
        return None

    def release(offset, size):
        for slot in range(offset, offset + size):
            stack[slot] -= 1

    remaining = order
    passes = 0
    while remaining and not state['overflow']:
        passes += 1
        deferred = []
        for node in remaining:
            if any(entry[2] is not None and entry[2][0].id not in done
                   for entry in node.inputs):
                deferred.append(node)
                continue
            temporaries = []
            if node.kind == 'CONVERT' and _SIZE[node.op[0]] == 3 == _SIZE[node.op[1]]:
                # float3 -> float3 conversions alias their input (stack_link).
                source = node.inputs[0][2]
                offset, size, _owner = offsets[(source[0].id, source[1])]
                for slot in range(offset, offset + size):
                    stack[slot] += 1
                offsets[(node.id, 0)] = (offset, size, node)
            else:
                for side, index in _allocation_steps(node):
                    if side == 'in':
                        size = _SIZE[node.inputs[index][1]]
                        temporaries.append((allocate(size), size))
                        continue
                    size = _SIZE.get(node.outputs[index][1], 0)
                    if not size:
                        continue
                    if (node.id, index) in users:
                        offset = allocate(size)
                        if offset is not None:
                            offsets[(node.id, index)] = (offset, size, node)
                    elif node.kind in _ALL_OUTPUTS_ASSIGNED:
                        allocate(size)  # Cycles never releases it
            done.add(node.id)
            for entry in node.inputs:
                if entry[2] is None:
                    continue
                key = (entry[2][0].id, entry[2][1])
                if key in offsets and users[key] <= done:
                    offset, size, _owner = offsets.pop(key)
                    release(offset, size)
            for offset, size in temporaries:
                if offset is not None:
                    release(offset, size)
            if state['overflow']:
                break
        if len(deferred) == len(remaining):
            break  # dependency cycle; Cycles breaks it before compiling
        remaining = deferred
    return state, passes


def estimate_svm_stack(material, limit=SVM_STACK_SIZE):
    """Return (peak stack floats, details) for the material's Cycles surface.

    A peak of SVM_STACK_SIZE means Cycles cannot compile the shader and the
    material renders black. ``details`` contains the number of compile
    sweeps and the node labels that hold the most stack at the peak. A larger
    ``limit`` measures how far an oversized graph exceeds the stack.
    """
    if material is None or not material.use_nodes or material.node_tree is None:
        return 0, {}
    graph = _Graph()
    # Blender 5.1+ Cycles compiles an inlined copy of the tree: groups are
    # flattened, constants folded and nodes copied in depth-first order from
    # the output. That order, not creation order, is the SVM compile order.
    inlined = (material.inline_shader_nodes()
               if hasattr(material, 'inline_shader_nodes') else None)
    graph.add_tree(inlined.node_tree if inlined is not None else material.node_tree)
    if graph.output is None:
        return 0, {}
    graph.remove_proxies()
    graph.expand()
    graph.fold()
    alive = graph.reachable()
    graph.deduplicate(alive)
    alive = graph.reachable()
    state, passes = _simulate(graph, alive, max(limit, SVM_STACK_SIZE))
    holders = {}
    for _offset, size, owner in state['peak_live']:
        label = owner.label.split('-', 1)[0] if owner.label else owner.kind
        holders[label] = holders.get(label, 0) + size
    top = sorted(holders.items(), key=lambda item: -item[1])[:8]
    peak = max(limit, SVM_STACK_SIZE) if state['overflow'] else state['peak']
    return peak, {'passes': passes, 'holders': top, 'overflow': state['overflow'],
                  'deduplicated': graph.deduplicated,
                  'live': [(owner.label, size) for _offset, size, owner in state['peak_live']]}
