"""Editable shader inputs for settings that can change without rebuilding masks.

Only link controls to a feature that is actually present in the material.
Camera geometry, depth thresholds, margins, guard resolution and feather width
are baked into attributes or PNGs and deliberately do not get inert controls.
"""


_PREFIX = 'SG Control | '


def _control(nodes, key, label, initial):
    name = _PREFIX + key
    node = nodes.get(name)
    if node is None:
        node = nodes.new('ShaderNodeValue')
        node.name = name
        node.label = label
        node.outputs[0].default_value = float(initial)
        node['sg_layout_stage'] = 'Graph controls'
        node.use_custom_color = True
        node.color = (0.19, 0.29, 0.24)
        node.width = 220
    return node.outputs[0]


def _math(nodes, links, operation, label, a, b, *, stage='Graph control wiring',
          camera_nodes=None, camera_index=None):
    node = nodes.new('ShaderNodeMath')
    node.operation = operation
    node.label = label
    node.hide = True
    node['sg_layout_stage'] = stage
    if camera_nodes is not None and camera_index is not None:
        camera_nodes[node.as_pointer()] = camera_index
    for index, value in enumerate((a, b)):
        if isinstance(value, (int, float)):
            node.inputs[index].default_value = value
        else:
            links.new(value, node.inputs[index])
    return node.outputs[0]


def _controlled_rejection(nodes, links, source, amount, label, *, stage='Graph control wiring',
                          camera_nodes=None, camera_index=None):
    """Replace uses of source by 1 - amount * (1 - source)."""
    targets = [link.to_socket for link in list(source.links)]
    if not targets:
        return None
    placement = dict(stage=stage, camera_nodes=camera_nodes, camera_index=camera_index)
    missing = _math(nodes, links, 'SUBTRACT', label + ' missing', 1.0, source, **placement)
    scaled = _math(nodes, links, 'MULTIPLY', label + ' amount', amount, missing, **placement)
    effective = _math(nodes, links, 'SUBTRACT', label, 1.0, scaled, **placement)
    for target in targets:
        links.new(effective, target)
    return effective


def _indexed(nodes, prefix):
    result = {}
    for node in nodes:
        if not node.label.startswith(prefix):
            continue
        index = node.label[len(prefix):].split('-', 1)[0]
        if index.isdigit():
            result[int(index)] = node
    return result


def install_projection_controls(material, scene, new_nodes, camera_nodes=None):
    """Expose live settings on a newly built projection pass.

    Existing controls are reused by appended local-edit passes. The caller
    passes only the nodes built in this pass so old mask switches are never
    wrapped a second time.
    """
    nodes, links = material.node_tree.nodes, material.node_tree.links
    if camera_nodes is None:
        camera_nodes = {}
    new = [node for node in new_nodes if node.type != 'FRAME']
    if not new:
        return

    angle = [node for node in new if node.label.startswith('AngleThreshold-')
             and node.type == 'MATH']
    scripts = [node for node in new if node.type == 'SCRIPT']
    if angle or any('AngleThreshold' in node.inputs for node in scripts):
        setting = _control(nodes, 'discard_angle', 'Discard Angle (degrees)',
                           scene.discard_factor)
        for node in angle:
            links.new(setting, node.inputs[1])
        for node in scripts:
            if 'AngleThreshold' in node.inputs:
                links.new(setting, node.inputs['AngleThreshold'])

    sharp = [node for node in new if node.type == 'MATH'
             and node.operation == 'POWER' and node.label.startswith('SharpW-')]
    base_power = [node for node in new if node.type == 'MATH'
                  and node.operation == 'POWER' and node.label == 'power_weight']
    script_power = [node for node in scripts if 'Power' in node.inputs]
    if sharp or base_power or script_power:
        setting = _control(nodes, 'exponent', 'Weight Exponent', scene.weight_exponent)
        for node in sharp:
            links.new(setting, node.inputs[1])
        if not sharp:
            for node in base_power:
                links.new(setting, node.inputs[1])
            for node in script_power:
                links.new(setting, node.inputs['Power'])

    guards = [node for node in new if node.label.startswith('ProjectionGuardWeight-')]
    if guards:
        enabled = _control(nodes, 'dppg_enabled', 'DPPG Enabled (0 / 1)', 1.0)
        enabled = _math(nodes, links, 'GREATER_THAN', 'DPPG switch', enabled, 0.5)
        for node in guards:
            _controlled_rejection(nodes, links, node.outputs[0], enabled,
                                  'ProjectionGuardWeightSwitch-' + node.label.split('-', 1)[1],
                                  stage='Depth and part guard', camera_nodes=camera_nodes,
                                  camera_index=camera_nodes.get(node.as_pointer()))

        feathers = [node for node in new if node.label.startswith('ProjectionGuardFeather-')]
        occlusion = [node for node in new if node.name.startswith('SGOcclusionFeather-')]
        influences = [node for node in new if node.name.startswith(
            ('SGFeatherInfluence-', 'SGOcclusionInfluence-'))]
        if feathers or occlusion or influences:
            feather = _control(nodes, 'feather_amount', 'DPPG Feather Amount (0..1)', 1.0)
            feather = _math(nodes, links, 'MINIMUM', 'Feather amount upper bound', feather, 1.0)
            feather = _math(nodes, links, 'MAXIMUM', 'Feather amount lower bound', feather, 0.0)
            amount = _math(nodes, links, 'MULTIPLY', 'Effective DPPG feather', enabled, feather)
            for node in feathers:
                _controlled_rejection(nodes, links, node.outputs[0], amount,
                                      'ProjectionGuardFeatherSwitch-' + node.label.split('-', 1)[1],
                                      stage='Guard feather', camera_nodes=camera_nodes,
                                      camera_index=camera_nodes.get(node.as_pointer()))
            for node in occlusion:
                _controlled_rejection(nodes, links, node.outputs[0], amount,
                                      'Occlusion feather switch ' + node.name,
                                      stage='Occlusion fade', camera_nodes=camera_nodes,
                                      camera_index=camera_nodes.get(node.as_pointer()))
            for node in influences:
                targets = [link.to_socket for link in list(node.outputs[0].links)]
                if targets:
                    gated = _math(nodes, links, 'MULTIPLY',
                                  'Feather influence switch ' + node.name,
                                  node.outputs[0], amount,
                                  stage=('Occlusion fade' if node.name.startswith('SGOcclusion')
                                         else 'Guard feather'), camera_nodes=camera_nodes,
                                  camera_index=camera_nodes.get(node.as_pointer()))
                    for target in targets:
                        links.new(gated, target)

        thresholds = [node for node in new if node.label == 'GuardRescueThreshold']
        actives = [node for node in new if node.label == 'GuardRescueActive']
        if thresholds or actives:
            rescue = _control(nodes, 'coverage_rescue', 'Coverage Rescue (0..1)',
                              scene.sg_guard_coverage_rescue)
            rescue = _math(nodes, links, 'MINIMUM', 'Rescue upper bound', rescue, 1.0,
                           stage='Guard coverage rescue')
            rescue = _math(nodes, links, 'MAXIMUM', 'Rescue lower bound', rescue, 0.0,
                           stage='Guard coverage rescue')
            for node in thresholds:
                links.new(rescue, node.inputs[1])
            for node in actives:
                links.new(rescue, node.inputs[0])

    signed_dots = _indexed(new, 'FacingSignedDot-')
    facing_weights = _indexed(new, 'FacingWeight-')
    if (signed_dots and facing_weights) or any('AllowBackface' in node.inputs for node in scripts):
        backface = _control(nodes, 'allow_backface', 'Allow Backface (0 / 1)',
                            scene.sg_allow_backface_projection)
        backface = _math(nodes, links, 'GREATER_THAN', 'Backface switch', backface, 0.5)
        for node in scripts:
            if 'AllowBackface' in node.inputs:
                links.new(backface, node.inputs['AllowBackface'])
        for index, signed in signed_dots.items():
            current = facing_weights.get(index)
            if current is None:
                continue
            placement = dict(stage='Facing and angle', camera_nodes=camera_nodes,
                             camera_index=camera_nodes.get(current.as_pointer()))
            targets = [link.to_socket for link in list(current.outputs[0].links)]
            if scene.sg_allow_backface_projection:
                back = current.outputs[0]
                front = _math(nodes, links, 'MULTIPLY', f'Front facing {index}',
                              signed.outputs[0], -1.0, **placement)
                front = _math(nodes, links, 'MAXIMUM', f'Front clamp {index}', front, 0.0,
                              **placement)
            else:
                front = current.outputs[0]
                back = _math(nodes, links, 'ABSOLUTE', f'Back facing {index}',
                             signed.outputs[0], 0.0, **placement)
            difference = _math(nodes, links, 'SUBTRACT', f'Backface difference {index}',
                               back, front, **placement)
            selected = _math(nodes, links, 'MULTIPLY', f'Backface selection {index}',
                             backface, difference, **placement)
            selected = _math(nodes, links, 'ADD', f'Facing switch {index}', front, selected,
                             **placement)
            for target in targets:
                links.new(selected, target)

    camera_mixes = [node for node in new if node.label.startswith('CameraMix-')]
    if camera_mixes:
        priority = _control(nodes, 'priority_enabled', 'Prioritize Initial Views (0 / 1)',
                            scene.early_priority)
        priority = _math(nodes, links, 'GREATER_THAN', 'Priority switch', priority, 0.5,
                         stage='Early view priority')
        strength = _control(nodes, 'priority_strength', 'Initial View Priority Strength',
                            scene.early_priority_strength)
        for mix in camera_mixes:
            factor = mix.inputs['Fac']
            if not factor.links:
                continue
            source = factor.links[0].from_socket
            if source.node.type == 'MAP_RANGE':
                mapping = source.node
                unprioritized = mapping.inputs[0].links[0].from_socket
            else:
                unprioritized = source
                mapping = nodes.new('ShaderNodeMapRange')
                mapping.label = 'EarlyPriorityMap-' + mix.label.split('-', 1)[1]
                mapping['sg_layout_stage'] = 'Early view priority'
                mapping.hide = True
                links.new(unprioritized, mapping.inputs[0])
            links.new(strength, mapping.inputs[1])
            placement = dict(stage='Early view priority')
            difference = _math(nodes, links, 'SUBTRACT', 'Priority difference',
                               mapping.outputs[0], unprioritized, **placement)
            added = _math(nodes, links, 'MULTIPLY', 'Priority adjustment', priority,
                          difference, **placement)
            selected = _math(nodes, links, 'ADD', 'Priority result', unprioritized, added,
                             **placement)
            links.new(selected, factor)

    acquisition = [node for node in new if node.label.startswith('AcquisitionQuality-')
                   and node.type == 'MATH' and node.inputs[1].links]
    if acquisition:
        amount = _control(nodes, 'acquisition_quality', 'Acquisition Quality Amount (0..1)', 1.0)
        amount = _math(nodes, links, 'MINIMUM', 'Acquisition upper bound', amount, 1.0)
        amount = _math(nodes, links, 'MAXIMUM', 'Acquisition lower bound', amount, 0.0)
        for node in acquisition:
            placement = dict(stage='Acquisition quality', camera_nodes=camera_nodes,
                             camera_index=camera_nodes.get(node.as_pointer()))
            density = node.inputs[1].links[0].from_socket
            missing = _math(nodes, links, 'SUBTRACT', 'Acquisition missing', 1.0, density,
                            **placement)
            adjustment = _math(nodes, links, 'MULTIPLY', 'Acquisition adjustment', amount,
                               missing, **placement)
            effective = _math(nodes, links, 'SUBTRACT', 'Acquisition factor', 1.0,
                              adjustment, **placement)
            links.new(effective, node.inputs[1])
