use std::sync::Arc;

use super::*;

pub(super) fn sample<'a>(
    project: &'a Project,
    project_root: &Path,
    composition_id: &'a str,
    frame: u64,
) -> Result<Frame<'a>, String> {
    sample_with_profile(
        project,
        project_root,
        composition_id,
        frame,
        quality::RenderProfile::Final,
    )
}

pub(super) fn sample_with_profile<'a>(
    project: &'a Project,
    project_root: &Path,
    composition_id: &'a str,
    frame: u64,
    profile: quality::RenderProfile,
) -> Result<Frame<'a>, String> {
    require(
        project.format == "mogf.project" && project.version == 1,
        "only mogf.project version 1 is supported",
    )?;
    // The MVP rejects document features it cannot execute rather than silently dropping them.
    require(
        project.requirements.is_empty(),
        "operation requirements are unsupported in this MVP",
    )?;
    require(
        project.states.is_empty() && project.functions.is_empty(),
        "state and functions are outside this MVP slice",
    )?;
    validate_assets(&project.assets)?;
    require(
        project.clips.is_empty() && project.media.is_empty(),
        "clips and media are outside this MVP slice",
    )?;
    validation::validate_materials(&project.materials, &project.shaders)?;
    require(
        project.recordings.is_empty() && project.bakes.is_empty(),
        "recordings and bakes are outside this MVP slice",
    )?;
    require(
        project.metadata.is_empty(),
        "metadata is not accepted by the MVP executable",
    )?;

    let composition = project
        .compositions
        .get(composition_id)
        .ok_or_else(|| format!("unknown composition {composition_id:?}"))?;
    require(
        composition.size[0] > 0 && composition.size[1] > 0,
        "composition dimensions must be positive",
    )?;
    let fps = composition.fps.as_positive()?;
    let duration = composition.duration.as_positive()?;
    let n = i128::from(frame)
        .checked_mul(i128::from(fps.d))
        .ok_or("sample-time overflow")?;
    let d = i128::from(fps.n);
    let duration_left = n
        .checked_mul(i128::from(duration.d))
        .ok_or("sample-time comparison overflow")?;
    let duration_right = i128::from(duration.n)
        .checked_mul(d)
        .ok_or("sample-time comparison overflow")?;
    require(
        duration_left < duration_right,
        "frame lies beyond the composition's half-open duration",
    )?;
    require(
        composition.clips.is_empty() && composition.recording.is_none(),
        "clips and recordings are outside this MVP slice",
    )?;
    require(
        composition.audio.len() <= 1,
        "MVP supports one composition-aligned audio asset",
    )?;
    for id in &composition.audio {
        let asset = project
            .assets
            .get(id)
            .ok_or_else(|| format!("missing audio asset {id:?}"))?;
        require(
            asset.kind == "audio",
            "composition audio must reference an audio asset",
        )?;
        let _ = resolve_asset(project_root, asset)?;
    }
    require(
        composition.layers.len() == 1,
        "MVP requires exactly one native_3d layer",
    )?;
    validation::validate_tracks(project, composition_id, &composition.timelines)?;

    let divisor = gcd_i128(n, d);
    let time = Fraction {
        n: (n / divisor).to_string(),
        d: (d / divisor).to_string(),
    };
    let mut camera_lens = None;
    let mut camera_world = None;
    let mut instances = Vec::new();
    let mut frame_lights = Vec::new();
    let mut camera_effect_shaders = Vec::new();
    let mut vertex_shader = None;
    let mut parents = Vec::new();
    let mut visited = std::collections::BTreeSet::new();
    for layer in &composition.layers {
        match layer {
            Layer::Native3d {
                root,
                camera,
                lights: layer_lights,
                transparent_sort,
                effects,
                vertex_shader: layer_vertex_shader,
            } => {
                require(
                    effects.len() <= 8,
                    "MVP supports at most eight ordered camera-final effects",
                )?;
                for effect_id in effects {
                    let effect = project.effects.get(effect_id).ok_or_else(|| {
                        format!("native_3d layer references missing effect {effect_id:?}")
                    })?;
                    let shader = project
                        .shaders
                        .get(&effect.shader)
                        .filter(|shader| shader.stage == "camera_effect")
                        .ok_or_else(|| {
                            format!("effect {effect_id:?} needs a camera_effect shader")
                        })?;
                    camera_effect_shaders.push(shader.source.clone().ok_or_else(|| {
                        format!(
                            "camera-effect shader {:?} has no loaded source",
                            effect.shader
                        )
                    })?);
                }
                if let Some(shader_id) = layer_vertex_shader {
                    let shader = project
                        .shaders
                        .get(shader_id)
                        .filter(|shader| shader.stage == "vertex")
                        .ok_or_else(|| {
                            format!("native_3d layer needs a vertex-stage shader {shader_id:?}")
                        })?;
                    vertex_shader = shader.source.clone();
                }
                let light_ids: std::collections::BTreeSet<String> =
                    layer_lights.iter().cloned().collect();
                require(
                    light_ids.len() == layer_lights.len(),
                    "native_3d layer repeats a light ID",
                )?;
                for light_id in layer_lights {
                    require(
                        project
                            .objects
                            .get(light_id)
                            .is_some_and(|object| matches!(object.body, Body::Light { .. })),
                        format!("layer light {light_id:?} does not reference a light object"),
                    )?;
                }
                require(
                    transparent_sort == "authored_order",
                    "MVP preserves authored instance order and does not sort transparency",
                )?;
                let camera_obj = project
                    .objects
                    .get(camera)
                    .ok_or_else(|| format!("missing camera object {camera:?}"))?;
                let Body::Camera {
                    operation,
                    version,
                    inputs,
                    params,
                } = &camera_obj.body
                else {
                    return Err(format!("object {camera:?} is not a camera"));
                };
                require(
                    *version == 1 && inputs.is_empty(),
                    "unsupported camera operation",
                )?;
                let projection = match operation.as_str() {
                    "mogf.camera.perspective" => {
                        let initial_fov_y = params.fov_y.ok_or("perspective camera needs fov_y")?;
                        require(
                            initial_fov_y.is_finite()
                                && initial_fov_y > 0.0
                                && initial_fov_y < std::f64::consts::PI
                                && params.ortho_height.is_none(),
                            "perspective camera needs fov_y in (0, pi) only",
                        )?;
                        camera::CameraProjection::Perspective {
                            fov_y: initial_fov_y,
                        }
                    }
                    "mogf.camera.orthographic" => {
                        let height = params
                            .ortho_height
                            .ok_or("orthographic camera needs ortho_height")?;
                        require(
                            height.is_finite() && height > 0.0 && params.fov_y.is_none(),
                            "orthographic camera needs a positive ortho_height only",
                        )?;
                        camera::CameraProjection::Orthographic { height }
                    }
                    _ => return Err(format!("unsupported camera operation {operation:?}")),
                };
                require(
                    params.near.is_finite()
                        && params.far.is_finite()
                        && params.near > 0.0
                        && params.far > params.near,
                    "camera clipping range is invalid",
                )?;
                camera_lens = Some((camera.clone(), projection, params.near, params.far));
                walk(
                    project,
                    composition_id,
                    root,
                    camera,
                    project_root,
                    &identity(),
                    frame,
                    fps,
                    &composition.timelines,
                    profile,
                    &mut parents,
                    &mut visited,
                    &mut camera_world,
                    &mut instances,
                    &mut frame_lights,
                    &light_ids,
                )?;
            }
        }
    }
    let (camera_id, mut projection, near, far) =
        camera_lens.ok_or("composition needs one native_3d camera layer")?;
    let camera_world = camera_world.ok_or("camera must be reachable from the layer root")?;
    if let camera::CameraProjection::Perspective { fov_y } = projection {
        projection = camera::CameraProjection::Perspective {
            fov_y: camera::evaluated_fov_y(
                project,
                composition_id,
                &camera_id,
                frame,
                fps,
                &composition.timelines,
                fov_y,
                camera_world[2][3],
            )?,
        };
    }
    let camera = camera::CameraFrame {
        object: camera_id,
        projection,
        near,
        far,
        world_matrix: camera_world,
    };
    Ok(Frame {
        format: "mogf.frame",
        version: 1,
        project_id: &project.project_id,
        composition: composition_id,
        frame,
        time,
        output_profile: &composition.output_profile,
        camera,
        instances,
        lights: frame_lights,
        shader_sources: project
            .shaders
            .iter()
            .filter_map(|(id, shader)| {
                shader
                    .source
                    .as_ref()
                    .map(|source| (id.clone(), std::sync::Arc::clone(source)))
            })
            .collect(),
        camera_effect_shaders,
        vertex_shader,
        scene_color_shader_ids: project
            .shaders
            .iter()
            .filter_map(|(id, shader)| (shader.stage == "surface_scene").then(|| id.clone()))
            .collect(),
    })
}

fn walk(
    project: &Project,
    composition_id: &str,
    object_id: &str,
    camera_id: &str,
    project_root: &Path,
    parent: &[[f64; 4]; 4],
    frame: u64,
    fps: FractionValue,
    timeline_ids: &[String],
    profile: quality::RenderProfile,
    parents: &mut Vec<String>,
    visited: &mut std::collections::BTreeSet<String>,
    camera_world: &mut Option<[[f64; 4]; 4]>,
    out: &mut Vec<InstanceFrame>,
    lights_out: &mut Vec<LightFrame>,
    light_ids: &std::collections::BTreeSet<String>,
) -> Result<(), String> {
    require(
        !parents.iter().any(|id| id == object_id),
        format!("object hierarchy cycle at {object_id:?}"),
    )?;
    require(
        visited.insert(object_id.to_owned()),
        format!("object {object_id:?} occurs more than once in the layer hierarchy"),
    )?;
    let object = project
        .objects
        .get(object_id)
        .ok_or_else(|| format!("missing object {object_id:?}"))?;
    let local = evaluated_transform(
        project,
        composition_id,
        object_id,
        object,
        frame,
        fps,
        timeline_ids,
    )?;
    let world = multiply(parent, &matrix(&local));
    if object_id == camera_id {
        *camera_world = Some(world);
    }
    match &object.body {
        Body::Group { children, scene } => {
            parents.push(object_id.to_owned());
            for child in children {
                walk(
                    project,
                    composition_id,
                    child,
                    camera_id,
                    project_root,
                    &world,
                    frame,
                    fps,
                    timeline_ids,
                    profile,
                    parents,
                    visited,
                    camera_world,
                    out,
                    lights_out,
                    light_ids,
                )?;
            }
            parents.pop();
            if let Some(scene) = scene {
                walk_scene_instance(
                    project,
                    composition_id,
                    object_id,
                    &object.name,
                    camera_id,
                    project_root,
                    &world,
                    frame,
                    fps,
                    timeline_ids,
                    scene,
                    camera_world,
                    out,
                )?;
            }
        }
        Body::Draw { batch } => {
            let graph = project
                .graphs
                .get(&batch.graph)
                .ok_or_else(|| format!("missing graph {:?}", batch.graph))?;
            let output = graph
                .outputs
                .get(&batch.output)
                .ok_or_else(|| format!("missing graph output {:?}", batch.output))?;
            let nodes = graph::evaluate_batch_graph(
                project,
                project_root,
                graph,
                output,
                composition_id,
                object_id,
                frame,
                fps,
                timeline_ids,
                profile,
            )?;
            for spec in nodes {
                let mut placement = spec.placement;
                let mut opacity = 1.0;
                animation::apply_instance_tracks(
                    project,
                    composition_id,
                    object_id,
                    &spec.selection_ids,
                    frame,
                    fps,
                    timeline_ids,
                    &mut placement,
                    &mut opacity,
                )?;
                let placed = multiply(&world, &matrix(&placement));
                out.push(InstanceFrame {
                    source_object: object_id.to_owned(),
                    source_name: object.name.clone(),
                    instance_id: spec.instance_id,
                    prototype: spec.prototype,
                    geometry_kind: spec.geometry_kind,
                    size: spec.size,
                    material: spec.material,
                    image: spec.image,
                    image_sampler: None,
                    imported_material: None,
                    imported_alpha_mode: 0,
                    color: spec.color,
                    shading: spec.shading,
                    lighting: spec.lighting,
                    shader_params: spec.shader_params,
                    shader_program: spec.shader_program,
                    casts_shadow: spec.casts_shadow,
                    receives_shadow: spec.receives_shadow,
                    opacity,
                    selection_ids: spec.selection_ids,
                    character_ids: spec.character_ids,
                    word_id: spec.word_id,
                    glyph_mesh: spec.glyph_mesh,
                    glyph_hash: spec.glyph_hash,
                    scene_vertices: spec.scene_vertices,
                    world_matrix: placed,
                });
            }
        }
        Body::Light {
            operation,
            version,
            inputs,
            params,
        } => {
            if light_ids.contains(object_id) {
                require(
                    *version == 1 && inputs.is_empty(),
                    "unsupported light operation",
                )?;
                require(
                    params
                        .color
                        .iter()
                        .all(|value| value.is_finite() && *value >= 0.0)
                        && params.intensity.is_finite()
                        && params.intensity >= 0.0
                        && params
                            .range
                            .is_none_or(|range| range.is_finite() && range > 0.0)
                        && params.shadow_bias.is_finite()
                        && (0.0..=0.05).contains(&params.shadow_bias)
                        && params.shadow_filter_radius.is_finite()
                        && (0.0..=8.0).contains(&params.shadow_filter_radius)
                        && matches!(params.shadow_resolution, 128 | 256 | 512),
                    format!(
                        "light {object_id:?} has invalid color, intensity, range, or shadow settings"
                    ),
                )?;
                let kind = match operation.as_str() {
                    "mogf.light.ambient" => "ambient",
                    "mogf.light.directional" => "directional",
                    "mogf.light.point" => "point",
                    "mogf.light.spot" => "spot",
                    other => return Err(format!("unsupported light operation {other:?}")),
                };
                let (inner_angle, outer_angle) = if kind == "spot" {
                    let inner = params
                        .inner_angle
                        .ok_or("spot light requires inner_angle and outer_angle in radians")?;
                    let outer = params
                        .outer_angle
                        .ok_or("spot light requires inner_angle and outer_angle in radians")?;
                    require(
                        inner.is_finite()
                            && outer.is_finite()
                            && inner >= 0.0
                            && inner < outer
                            && outer < std::f64::consts::FRAC_PI_2,
                        format!("spot light {object_id:?} has invalid cone angles"),
                    )?;
                    (Some(inner), Some(outer))
                } else {
                    require(
                        params.inner_angle.is_none() && params.outer_angle.is_none(),
                        format!("non-spot light {object_id:?} cannot have cone angles"),
                    )?;
                    (None, None)
                };
                require(
                    !params.casts_shadow || matches!(kind, "directional" | "spot"),
                    "only directional or spot lights can cast shadows in the mobile renderer",
                )?;
                let world_position = renderer::transform_point(&world, [0.0; 3]);
                let mut direction = renderer::transform_point(&world, [0.0, 0.0, -1.0]);
                for axis in 0..3 {
                    direction[axis] -= world_position[axis];
                }
                let direction_length = direction
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                if direction_length > 1e-12 {
                    direction = direction.map(|value| value / direction_length);
                }
                lights_out.push(LightFrame {
                    kind: kind.to_owned(),
                    position: world_position,
                    direction,
                    color: params.color,
                    intensity: params.intensity,
                    range: params.range,
                    inner_angle,
                    outer_angle,
                    shadow_bias: params.shadow_bias,
                    shadow_filter_radius: profile.shadow_filter_radius(params.shadow_filter_radius),
                    shadow_resolution: profile.shadow_resolution(params.shadow_resolution),
                    casts_shadow: params.casts_shadow,
                });
            }
        }
        Body::Camera { .. } => {}
    }
    Ok(())
}

fn walk_scene_instance(
    project: &Project,
    composition_id: &str,
    actor_id: &str,
    actor_name: &str,
    _camera_id: &str,
    _project_root: &Path,
    actor_world: &Mat4,
    frame: u64,
    fps: FractionValue,
    timeline_ids: &[String],
    instance: &SceneInstance,
    _camera_world: &mut Option<Mat4>,
    out: &mut Vec<InstanceFrame>,
) -> Result<(), String> {
    let asset = project.assets.get(&instance.asset).ok_or_else(|| {
        format!(
            "subscene actor {actor_id:?} references missing asset {:?}",
            instance.asset
        )
    })?;
    require(
        asset.kind == "scene",
        "subscene actor must reference a scene asset",
    )?;
    let scene = asset
        .scene
        .as_ref()
        .ok_or_else(|| format!("scene asset {:?} was not prepared", instance.asset))?;
    let selected_scene_index = instance.scene.or(scene.default_scene).ok_or_else(|| {
        "scene asset has no declared default; select a source scene explicitly".to_owned()
    })?;
    let selected_scene = scene.scenes.get(selected_scene_index).ok_or_else(|| {
        let available = scene
            .scenes
            .iter()
            .map(|scene| format!("{}:{:?}", scene.index, scene.name))
            .collect::<Vec<_>>()
            .join(", ");
        format!("scene selection {selected_scene_index} is out of range; available: [{available}]")
    })?;
    for (control, node) in &instance.controls {
        require(
            *node < scene.nodes.len(),
            format!("scene actor control {control:?} refers to missing node {node}"),
        )?;
    }
    let mut local = scene
        .nodes
        .iter()
        .map(|node| Transform {
            translation: node.translation,
            rotation: normalized(node.rotation).unwrap_or([0.0, 0.0, 0.0, 1.0]),
            scale: node.scale,
        })
        .collect::<Vec<_>>();
    let mut morph_weights = scene
        .nodes
        .iter()
        .map(|node| {
            node.morph_weights.clone().unwrap_or_else(|| {
                node.mesh
                    .map(|mesh| scene.meshes[mesh].morph_weights.clone())
                    .unwrap_or_default()
            })
        })
        .collect::<Vec<_>>();

    let sample_time = (frame as f64) * (fps.d as f64) / (fps.n as f64);
    let mut playbacks = instance.playbacks.iter().collect::<Vec<_>>();
    if let Some(playback) = &instance.playback {
        playbacks.push(playback);
    }
    let mut sampled_channels: BTreeMap<
        (usize, crate::scene_asset::AnimatedProperty),
        Vec<(i32, usize, f64, Vec<f64>)>,
    > = BTreeMap::new();
    for (ordinal, playback) in playbacks.into_iter().enumerate() {
        require(
            playback.speed.is_finite()
                && playback.offset.is_finite()
                && playback.weight.is_finite()
                && playback.weight >= 0.0,
            "subscene playback rate, offset, and weight must be finite and weight nonnegative",
        )?;
        let animation = scene
            .animations
            .iter()
            .find(|animation| animation.name == playback.animation)
            .ok_or_else(|| {
                format!(
                    "scene actor selects missing animation {:?}",
                    playback.animation
                )
            })?;
        let mut time = sample_time * playback.speed + playback.offset;
        require(time.is_finite(), "subscene animation time overflow")?;
        if animation.duration > 0.0 {
            match playback.loop_mode.as_str() {
                "repeat" => time = time.rem_euclid(animation.duration),
                "once" => time = time.clamp(0.0, animation.duration),
                "ping_pong" => {
                    let phase = time.rem_euclid(animation.duration * 2.0);
                    time = if phase <= animation.duration {
                        phase
                    } else {
                        animation.duration * 2.0 - phase
                    };
                }
                other => return Err(format!("unsupported subscene loop mode {other:?}")),
            }
            time += animation.start_time;
        } else {
            time = animation.start_time;
        }
        for channel in &animation.channels {
            let value = crate::scene_asset::interpolate_channel(channel, time);
            sampled_channels
                .entry((channel.node, channel.property))
                .or_default()
                .push((playback.priority, ordinal, playback.weight, value));
        }
    }
    for ((node_index, property), mut values) in sampled_channels {
        let node = local
            .get_mut(node_index)
            .ok_or("animation targets a missing subscene node")?;
        values.sort_by_key(|(priority, ordinal, _, _)| (*priority, *ordinal));
        match property {
            crate::scene_asset::AnimatedProperty::Translation
            | crate::scene_asset::AnimatedProperty::Scale => {
                let total = values.iter().map(|(_, _, weight, _)| weight).sum::<f64>();
                if total > 0.0 {
                    let blended: [f64; 3] = std::array::from_fn(|axis| {
                        values
                            .iter()
                            .map(|(_, _, weight, value)| weight * value[axis])
                            .sum::<f64>()
                            / total
                    });
                    if property == crate::scene_asset::AnimatedProperty::Translation {
                        node.translation = [blended[0], blended[1], blended[2]];
                    } else {
                        node.scale = [blended[0], blended[1], blended[2]];
                    }
                }
            }
            crate::scene_asset::AnimatedProperty::Rotation => {
                let mut accumulated = None;
                let mut total = 0.0;
                for (_, _, weight, value) in values {
                    if weight == 0.0 {
                        continue;
                    }
                    let quaternion = normalized(
                        value
                            .as_slice()
                            .try_into()
                            .map_err(|_| "animation rotation has invalid cardinality")?,
                    )?;
                    accumulated = Some(match accumulated {
                        None => quaternion,
                        Some(previous) => crate::scene_asset::slerp_quaternion(
                            previous,
                            quaternion,
                            weight / (total + weight),
                        ),
                    });
                    total += weight;
                }
                if let Some(rotation) = accumulated {
                    node.rotation = rotation;
                }
            }
            crate::scene_asset::AnimatedProperty::Weights => {
                let total = values.iter().map(|(_, _, weight, _)| weight).sum::<f64>();
                if total > 0.0 {
                    let weights = &mut morph_weights[node_index];
                    for (target, weight) in weights.iter_mut().enumerate() {
                        *weight = values
                            .iter()
                            .map(|(_, _, layer_weight, value)| layer_weight * value[target])
                            .sum::<f64>()
                            / total;
                    }
                    require(
                        weights.iter().all(|weight| weight.is_finite()),
                        format!("node {node_index} morph weight blending overflow"),
                    )?;
                }
            }
        }
    }

    for node_index in 0..scene.nodes.len() {
        let selectors = instance
            .controls
            .iter()
            .filter_map(|(name, index)| (*index == node_index).then_some(name.clone()))
            .collect::<Vec<_>>();
        if !selectors.is_empty() {
            let mut opacity = 1.0;
            animation::apply_instance_tracks(
                project,
                composition_id,
                actor_id,
                &selectors,
                frame,
                fps,
                timeline_ids,
                &mut local[node_index],
                &mut opacity,
            )?;
            require(
                opacity == 1.0,
                "subscene node controls do not support opacity tracks yet",
            )?;
        }
    }

    let mut worlds = vec![None; scene.nodes.len()];
    let mut roots = (0..scene.nodes.len())
        .filter(|index| scene.nodes[*index].parent.is_none())
        .collect::<Vec<_>>();
    while let Some(index) = roots.pop() {
        if worlds[index].is_some() {
            continue;
        }
        let local_matrix = crate::matrix(&local[index]);
        worlds[index] = Some(if let Some(parent) = scene.nodes[index].parent {
            let parent_world = worlds[parent].ok_or("imported node parent was not evaluated")?;
            multiply(&parent_world, &local_matrix)
        } else {
            local_matrix
        });
        roots.extend(scene.nodes[index].children.iter().copied());
    }
    if worlds.iter().any(Option::is_none) {
        return Err("cycle in imported scene node hierarchy".to_owned());
    }

    // A glTF document may contain nodes outside its selected/default scene.
    // Keep them available for references and animation evaluation, but never
    // draw those unselected source branches for this mounted actor.
    let mut selected_nodes = vec![false; scene.nodes.len()];
    let mut pending = selected_scene.roots.clone();
    while let Some(index) = pending.pop() {
        if selected_nodes[index] {
            continue;
        }
        selected_nodes[index] = true;
        pending.extend(scene.nodes[index].children.iter().copied());
    }

    for node in &scene.nodes {
        if !selected_nodes[node.index] {
            continue;
        }
        let Some(mesh_index) = node.mesh else {
            continue;
        };
        let mesh = scene
            .meshes
            .get(mesh_index)
            .ok_or("subscene node references missing mesh")?;
        for (primitive_index, primitive) in mesh.primitives.iter().enumerate() {
            if primitive.mode != crate::scene_asset::PrimitiveMode::Triangles {
                continue;
            }
            let imported_material = primitive
                .material
                .and_then(|material| scene.materials.get(material));
            let mut render_imported_material = imported_material.cloned();
            if let Some(material) = &mut render_imported_material {
                for binding in [
                    &mut material.base_color_texture,
                    &mut material.normal_texture,
                    &mut material.metallic_roughness_texture,
                    &mut material.occlusion_texture,
                    &mut material.emissive_texture,
                ] {
                    if binding.as_ref().is_some_and(|binding| {
                        !primitive.texcoords.contains_key(&binding.tex_coord)
                    }) {
                        *binding = None;
                    }
                }
            }
            let texture_bindings = render_imported_material.as_ref().map(|material| {
                [
                    material.base_color_texture.as_ref(),
                    material.normal_texture.as_ref(),
                    material.metallic_roughness_texture.as_ref(),
                    material.occlusion_texture.as_ref(),
                    material.emissive_texture.as_ref(),
                ]
            });
            let base_texture = texture_bindings.and_then(|bindings| bindings[0]);
            let mut vertices = Vec::with_capacity(primitive.indices.len());
            let skin = node.skin.and_then(|skin| scene.skins.get(skin));
            let skin_matrices = if let Some(skin) = skin {
                require(
                    skin.joints.len() == skin.inverse_bind_matrices.len(),
                    "subscene skin joint and inverse-bind counts differ",
                )?;
                let mut matrices = Vec::with_capacity(skin.joints.len());
                for (joint_index, inverse_bind) in
                    skin.joints.iter().zip(&skin.inverse_bind_matrices)
                {
                    let joint_world = worlds
                        .get(*joint_index)
                        .and_then(|matrix| *matrix)
                        .ok_or("subscene skin references missing joint transform")?;
                    matrices.push(multiply(&joint_world, inverse_bind));
                }
                Some(matrices)
            } else {
                None
            };
            for index in &primitive.indices {
                let index = *index as usize;
                let mut position = *primitive
                    .positions
                    .get(index)
                    .ok_or("subscene index is outside vertex buffer")?;
                let mut normal = primitive.normals.get(index).copied().unwrap_or([0.0; 3]);
                // Morph in mesh-local space before applying the skin palette.
                for (target, weight) in primitive
                    .morph_targets
                    .iter()
                    .zip(&morph_weights[node.index])
                {
                    if *weight == 0.0 {
                        continue;
                    }
                    for axis in 0..3 {
                        if let Some(delta) = target.positions.get(index) {
                            position[axis] += weight * delta[axis];
                        }
                        if let Some(delta) = target.normals.get(index) {
                            normal[axis] += weight * delta[axis];
                        }
                    }
                }
                require(
                    position
                        .iter()
                        .chain(normal.iter())
                        .all(|value| value.is_finite()),
                    format!(
                        "subscene node {} vertex {index} morph deformation overflow",
                        node.index
                    ),
                )?;
                if let Some(matrices) = &skin_matrices {
                    let joints = *primitive
                        .joints
                        .get(index)
                        .ok_or("skinned subscene vertex has no JOINTS_0")?;
                    let weights = *primitive
                        .weights
                        .get(index)
                        .ok_or("skinned subscene vertex has no WEIGHTS_0")?;
                    let total = weights.iter().sum::<f64>();
                    require(
                        total.is_finite() && total > 1e-12,
                        "subscene skin weights must have positive finite sum",
                    )?;
                    let morphed_position = position;
                    position = [0.0; 3];
                    let mut blended = [[0.0; 4]; 4];
                    for influence in 0..4 {
                        let weight = weights[influence] / total;
                        if weight == 0.0 {
                            continue;
                        }
                        let matrix = matrices
                            .get(joints[influence] as usize)
                            .ok_or("subscene skin joint index is out of range")?;
                        let transformed = renderer::transform_point(matrix, morphed_position);
                        for axis in 0..3 {
                            position[axis] += transformed[axis] * weight;
                        }
                        for row in 0..4 {
                            for col in 0..4 {
                                blended[row][col] += matrix[row][col] * weight;
                            }
                        }
                    }
                    // Positions have already been deformed into scene space.
                    // Apply the inverse transpose of that same blended linear
                    // map, including joint scale, before actor placement.
                    normal = renderer::inverse(&blended)
                        .map(|inverse| {
                            std::array::from_fn(|axis| {
                                (0..3).map(|row| inverse[row][axis] * normal[row]).sum()
                            })
                        })
                        .unwrap_or([0.0; 3]);
                }
                let default_uv = [position[0] + 0.5, 0.5 - position[1]];
                let mut texture_uvs = [default_uv; 5];
                if let Some(bindings) = texture_bindings {
                    for (slot, binding) in bindings.into_iter().enumerate() {
                        if let Some(binding) = binding {
                            texture_uvs[slot] = primitive
                                .texcoords
                                .get(&binding.tex_coord)
                                .and_then(|set| set.get(index))
                                .copied()
                                .unwrap_or(default_uv);
                        }
                    }
                }
                vertices.push(SceneVertex {
                    position,
                    normal,
                    color: primitive.colors.get(index).copied().unwrap_or([1.0; 4]),
                    texture_uvs,
                });
            }
            let override_id = imported_material
                .and_then(|material| instance.material_overrides.get(&material.name));
            let project_material = override_id.and_then(|id| project.materials.get(id));
            let imported_material_draw = if project_material.is_none() {
                render_imported_material.clone()
            } else {
                None
            };
            let (
                material_id,
                color,
                shading,
                lighting,
                shader_params,
                shader_program,
                image,
                image_sampler,
                selected_imported_material,
                imported_alpha_mode,
            ) = if let Some(material) = project_material {
                let (shading, lighting, shader_params) = material.gpu_shading();
                (
                    override_id.unwrap().clone(),
                    material.params.color,
                    shading,
                    lighting,
                    shader_params,
                    material.params.shader.clone(),
                    material
                        .params
                        .image
                        .as_ref()
                        .and_then(|id| project.assets.get(id))
                        .and_then(|asset| asset.image.clone()),
                    None,
                    None,
                    0,
                )
            } else if let Some(material) = imported_material {
                (
                    format!("{}::{}", instance.asset, material.name),
                    material.base_color,
                    ShadingModel::Principled,
                    [
                        material.metallic,
                        material.roughness,
                        0.5,
                        material.clearcoat,
                    ],
                    [
                        material.emissive[0],
                        material.emissive[1],
                        material.emissive[2],
                        material.alpha_cutoff,
                    ],
                    None,
                    base_texture.map(|binding| Arc::clone(&binding.image)),
                    base_texture.map(|binding| binding.sampler),
                    imported_material_draw,
                    if scene.source == "gltf" {
                        match material.alpha_mode {
                            crate::scene_asset::AlphaMode::Opaque => 1,
                            crate::scene_asset::AlphaMode::Mask => 2,
                            crate::scene_asset::AlphaMode::Blend => 3,
                        }
                    } else {
                        0
                    },
                )
            } else {
                (
                    format!("{}::default", instance.asset),
                    [1.0; 4],
                    ShadingModel::Principled,
                    [0.0, 0.5, 0.5, 0.0],
                    [0.0; 4],
                    None,
                    None,
                    None,
                    None,
                    0,
                )
            };
            let node_world = worlds[node.index].ok_or("missing imported node transform")?;
            let placement = if skin_matrices.is_some() {
                *actor_world
            } else {
                multiply(actor_world, &node_world)
            };
            let selector = instance
                .controls
                .iter()
                .filter_map(|(name, index)| (*index == node.index).then_some(name.clone()))
                .collect::<Vec<_>>();
            let vertex_array: Arc<[SceneVertex]> = Arc::from(vertices);
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            instance.asset.hash(&mut hasher);
            primitive_index.hash(&mut hasher);
            for vertex in vertex_array.iter() {
                vertex
                    .position
                    .iter()
                    .chain(vertex.normal.iter())
                    .chain(vertex.color.iter())
                    .chain(vertex.texture_uvs.iter().flatten())
                    .for_each(|value| value.to_bits().hash(&mut hasher));
            }
            let geometry_hash = hasher.finish();
            out.push(InstanceFrame {
                source_object: actor_id.to_owned(),
                source_name: format!("{actor_name}/{}", node.name),
                instance_id: format!(
                    "{}:node-{}:primitive-{primitive_index}",
                    actor_id, node.index
                ),
                prototype: format!("{}:mesh-{mesh_index}", instance.asset),
                geometry_kind: "scene_mesh".to_owned(),
                size: [1.0; 3],
                material: material_id,
                image,
                image_sampler,
                imported_material: selected_imported_material,
                imported_alpha_mode,
                color,
                shading,
                lighting,
                shader_params,
                shader_program,
                casts_shadow: true,
                receives_shadow: true,
                opacity: 1.0,
                selection_ids: selector,
                character_ids: Vec::new(),
                word_id: None,
                glyph_mesh: None,
                glyph_hash: Some(geometry_hash),
                scene_vertices: Some(vertex_array),
                world_matrix: placement,
            });
        }
    }
    Ok(())
}

fn evaluated_transform(
    project: &Project,
    composition_id: &str,
    object_id: &str,
    object: &Object,
    frame: u64,
    fps: FractionValue,
    timeline_ids: &[String],
) -> Result<Transform, String> {
    validation::finite_transform(&object.local)?;
    let mut value = Transform {
        translation: object.local.translation,
        rotation: normalized(object.local.rotation)?,
        scale: object.local.scale,
    };
    let sample_n = i128::from(frame) * i128::from(fps.d);
    let sample_d = i128::from(fps.n);
    let mut tracks = Vec::new();
    for timeline_id in timeline_ids {
        let timeline = project
            .timelines
            .get(timeline_id)
            .ok_or_else(|| format!("missing timeline {timeline_id:?}"))?;
        for track in &timeline.tracks {
            if track.target.table == "objects" && track.target.record == object_id {
                if track.target.instances.is_some() {
                    continue;
                }
                if track.target.slot == "text.weight" || track.target.slot == "camera.fov_y" {
                    continue;
                }
                let Clock::Composition(clock_id) = &track.clock;
                require(
                    clock_id == composition_id,
                    "track uses another composition clock",
                )?;
                require(
                    track.target.slot == "transform.translation.x"
                        || track.target.slot == "transform.translation.y"
                        || track.target.slot == "transform.translation.z",
                    "MVP timeline target must be a translation component",
                )?;
                require(
                    track.fill == "forward" && track.blend == "replace",
                    "MVP supports only forward-fill replace tracks",
                )?;
                require(
                    track.curve.operation == "mogf.curve.from_to"
                        && track.curve.version == 1
                        && track.curve.inputs.is_empty(),
                    "unsupported curve operation",
                )?;
                let at = track.at.as_positive_or_zero()?;
                let duration = track.duration.as_positive()?;
                let sample_at = sample_n
                    .checked_mul(i128::from(at.d))
                    .ok_or("track-time comparison overflow")?;
                let start = i128::from(at.n)
                    .checked_mul(sample_d)
                    .ok_or("track-time comparison overflow")?;
                let elapsed_n = sample_at - start;
                let before_end = elapsed_n
                    .checked_mul(i128::from(duration.d))
                    .ok_or("track-time comparison overflow")?;
                let duration_scaled = i128::from(duration.n)
                    .checked_mul(sample_d)
                    .and_then(|v| v.checked_mul(i128::from(at.d)))
                    .ok_or("track-time comparison overflow")?;
                let active = if elapsed_n < 0 {
                    None
                } else if before_end >= duration_scaled {
                    Some(1.0)
                } else {
                    Some((before_end as f64 / duration_scaled as f64).clamp(0.0, 1.0))
                };
                if let Some(mut t) = active {
                    t = animation::ease(track, t)?;
                    let (from, to) = animation::scalar_curve_endpoints(&track.curve.params)?;
                    let v = from + (to - from) * t;
                    require(v.is_finite(), "curve produced a non-finite value")?;
                    tracks.push((
                        track.priority,
                        track
                            .ordinal
                            .parse::<u64>()
                            .map_err(|_| "track ordinal must be a u64 string")?,
                        track.target.slot.as_str(),
                        v,
                    ));
                }
            }
        }
    }
    tracks.sort_by_key(|(priority, ordinal, _, _)| (*priority, *ordinal));
    for (_, _, slot, v) in tracks {
        let axis = match slot {
            "transform.translation.x" => 0,
            "transform.translation.y" => 1,
            _ => 2,
        };
        value.translation[axis] = v;
    }
    Ok(Transform {
        translation: value.translation,
        rotation: value.rotation,
        scale: value.scale,
    })
}
