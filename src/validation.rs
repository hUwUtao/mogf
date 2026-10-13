use super::*;

pub(super) fn validate_materials(
    materials: &BTreeMap<String, Material>,
    shaders: &BTreeMap<String, ShaderRecord>,
) -> Result<(), String> {
    for (id, material) in materials {
        let uv = material.params.image_uv_rect;
        require(
            uv.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                && uv[0] < uv[2]
                && uv[1] < uv[3],
            format!("material {id:?} image_uv_rect must be an ordered normalized rectangle"),
        )?;
        require(
            material.params.image.is_none()
                || (material.operation == "mogf.material.unlit"
                    && material.params.shader.is_none()),
            format!("image material {id:?} requires unlit shading without a custom shader"),
        )?;
        require(
            matches!(
                material.operation.as_str(),
                "mogf.material.unlit" | "mogf.material.phong" | "mogf.material.principled"
            ) && material.version == 1
                && material.inputs.is_empty(),
            format!("material {id:?} uses an unsupported operation"),
        )?;
        require(
            material
                .params
                .color
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
            format!("material {id:?} color must be finite linear RGBA in [0, 1]"),
        )?;
        require(
            material.params.color[3] == 1.0,
            format!("material {id:?} color alpha must be 1; use instance opacity for transparency"),
        )?;
        if let Some(shader_id) = &material.params.shader {
            require(
                shaders.get(shader_id).is_some_and(|shader| {
                    matches!(shader.stage.as_str(), "surface" | "surface_scene")
                }),
                format!("material {id:?} references missing or non-surface shader {shader_id:?}"),
            )?;
        }
        require(
            material
                .params
                .shader_params
                .iter()
                .all(|value| value.is_finite() && (*value as f32).is_finite()),
            format!("material {id:?} shader_params must be finite and representable as f32"),
        )?;
        if material.operation == "mogf.material.phong" {
            require(
                [
                    material.params.ambient,
                    material.params.diffuse,
                    material.params.specular,
                    material.params.shininess,
                ]
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
                format!("Phong material {id:?} lighting values must be finite and nonnegative"),
            )?;
            require(
                material.params.shininess > 0.0 && material.params.shininess <= 512.0,
                format!("Phong material {id:?} shininess must be in (0, 512]"),
            )?;
        }
        if material.operation == "mogf.material.principled" {
            require(
                [
                    material.params.metallic,
                    material.params.roughness,
                    material.params.reflectance,
                    material.params.clear_coat,
                ]
                .into_iter()
                .flatten()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(&value)),
                format!("principled material {id:?} values must be finite and in [0, 1]"),
            )?;
            require(
                material.params.roughness.unwrap_or(0.5) > 0.0,
                format!("principled material {id:?} roughness must be greater than zero"),
            )?;
        }
    }
    Ok(())
}

pub(super) fn finite_transform(t: &Transform) -> Result<(), String> {
    require(
        t.translation
            .iter()
            .chain(t.rotation.iter())
            .chain(t.scale.iter())
            .all(|v| v.is_finite()),
        "transform values must be finite",
    )
}

pub(super) fn validate_tracks(
    project: &Project,
    composition_id: &str,
    timeline_ids: &[String],
) -> Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();
    for timeline_id in timeline_ids {
        require(
            seen.insert(timeline_id),
            format!("composition repeats timeline {timeline_id:?}"),
        )?;
        let timeline = project
            .timelines
            .get(timeline_id)
            .ok_or_else(|| format!("missing timeline {timeline_id:?}"))?;
        for track in &timeline.tracks {
            require(
                track.target.table == "objects"
                    && project.objects.contains_key(&track.target.record),
                "MVP track target must reference an existing object",
            )?;
            if track.target.instances.is_some() {
                require(
                    track
                        .target
                        .instances
                        .as_ref()
                        .is_some_and(|ids| !ids.is_empty() && ids.iter().all(|id| !id.is_empty())),
                    "instance target needs nonempty selector IDs",
                )?;
                require(
                    track.target.slot == "transform.translation.x"
                        || track.target.slot == "transform.translation.y"
                        || track.target.slot == "transform.translation.z"
                        || track.target.slot == "transform.rotation"
                        || track.target.slot == "material.opacity",
                    "instance track target must be translation, quaternion rotation, or opacity",
                )?;
            } else {
                require(
                    track.target.slot == "transform.translation.x"
                        || track.target.slot == "transform.translation.y"
                        || track.target.slot == "transform.translation.z"
                        || track.target.slot == "text.weight"
                        || track.target.slot == "camera.fov_y",
                    "object track target must be translation, text.weight, or camera.fov_y",
                )?;
                if track.target.slot == "text.weight" {
                    let (from, to) = animation::scalar_curve_endpoints(&track.curve.params)?;
                    require(
                        (100.0..=900.0).contains(&from) && (100.0..=900.0).contains(&to),
                        "text weight endpoints must be in [100, 900]",
                    )?;
                } else if track.target.slot == "camera.fov_y" {
                    let object = &project.objects[&track.target.record];
                    require(
                        matches!(
                            &object.body,
                            Body::Camera { operation, .. }
                                if operation == "mogf.camera.perspective"
                        ),
                        "camera.fov_y tracks require a perspective camera",
                    )?;
                    let (from, to) = camera::fov_curve_endpoints(&track.curve.params)?;
                    require(
                        from > 0.0
                            && from < std::f64::consts::PI
                            && to > 0.0
                            && to < std::f64::consts::PI,
                        "camera fov_y endpoints must be in (0, pi) radians",
                    )?;
                    let _ = camera::dolly_zoom_params(&track.curve.params)?;
                }
            }
            let Clock::Composition(clock_id) = &track.clock;
            require(
                clock_id == composition_id,
                "track uses another composition clock",
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
            if track.target.slot == "transform.rotation" {
                let _ = animation::quaternion_curve_value(track, 0.0)?;
            } else if track.target.slot == "camera.fov_y" {
                let (from, to) = camera::fov_curve_endpoints(&track.curve.params)?;
                require(
                    from.is_finite() && to.is_finite(),
                    "curve endpoints must be finite",
                )?;
            } else {
                let (from, to) = animation::scalar_curve_endpoints(&track.curve.params)?;
                require(
                    from.is_finite() && to.is_finite(),
                    "curve endpoints must be finite",
                )?;
            }
            require(
                if track.target.slot == "transform.rotation" {
                    matches!(track.curve.params.ease.as_str(), "linear" | "cubic_out")
                } else if track.target.slot == "camera.fov_y" {
                    matches!(
                        track.curve.params.ease.as_str(),
                        "linear"
                            | "cubic_out"
                            | "quadratic_bounce_out"
                            | "quadratic_bounce_in"
                            | "dolly_zoom"
                    )
                } else {
                    matches!(
                        track.curve.params.ease.as_str(),
                        "linear" | "cubic_out" | "quadratic_bounce_out" | "quadratic_bounce_in"
                    )
                },
                "unsupported easing for track property",
            )?;
            let _ = track.at.as_positive_or_zero()?;
            let _ = track.duration.as_positive()?;
            let ordinal = track
                .ordinal
                .parse::<u64>()
                .map_err(|_| "track ordinal must be a canonical u64 string")?;
            require(
                ordinal.to_string() == track.ordinal,
                "track ordinal must be canonical",
            )?;
        }
    }
    Ok(())
}

pub(super) fn require_keys<T>(
    map: &BTreeMap<String, T>,
    keys: &[&str],
    label: &str,
) -> Result<(), String> {
    require(
        map.len() == keys.len() && keys.iter().all(|key| map.contains_key(*key)),
        format!("{label} must contain exactly {}", keys.join(", ")),
    )
}
