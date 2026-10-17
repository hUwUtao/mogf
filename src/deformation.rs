use std::collections::{BTreeMap, BTreeSet};

use crate::model::{AssetRecord, ImageData, JsonMap, SceneVertex};

pub(super) fn apply(
    operation: &str,
    vertices: &mut [SceneVertex],
    inputs: &BTreeMap<String, crate::model::Input>,
    params: &JsonMap,
    assets: &BTreeMap<String, AssetRecord>,
    time: f64,
) -> Result<(), String> {
    match operation {
        "mogf.deform.simple" => simple(vertices, params),
        "mogf.deform.wave" => wave(vertices, params, time),
        "mogf.deform.cast" => cast(vertices, params),
        "mogf.deform.smooth" => smooth(vertices, params),
        "mogf.deform.hook" => hook(vertices, params),
        "mogf.deform.lattice" => lattice(vertices, params),
        "mogf.deform.warp" => warp(vertices, params),
        "mogf.deform.curve" => curve(vertices, params),
        "mogf.deform.displace" => displace(vertices, inputs, params, assets),
        _ => Err(format!(
            "unsupported mesh deformation operation {operation:?}"
        )),
    }?;
    recalculate_normals(vertices);
    if vertices.iter().any(|vertex| {
        vertex
            .position
            .iter()
            .any(|value| !value.is_finite() || !(*value as f32).is_finite())
    }) {
        return Err(format!("{operation} produced a non-finite position"));
    }
    Ok(())
}

fn simple(vertices: &mut [SceneVertex], params: &JsonMap) -> Result<(), String> {
    require_params(
        params,
        &["mode", "axis", "amount"],
        &["origin", "bend_axis"],
        "deform.simple",
    )?;
    let mode = string(params, "mode")?;
    let axis_index = axis(string(params, "axis")?)?;
    let amount = number(params, "amount")?;
    let origin = vector3(params, "origin")?.unwrap_or([0.0; 3]);
    let other = match params.get("bend_axis").and_then(serde_json::Value::as_str) {
        Some(value) => axis(value)?,
        None => (axis_index + 1) % 3,
    };
    if mode == "bend" {
        if axis_index == other {
            return Err("deform.simple bend_axis must differ from axis".to_owned());
        }
    } else if !matches!(mode, "twist" | "taper" | "stretch") {
        return Err(format!("unsupported simple deformation mode {mode:?}"));
    }
    for vertex in vertices {
        let p = &mut vertex.position;
        let distance = p[axis_index] - origin[axis_index];
        match mode {
            "twist" => {
                let angle = amount * distance;
                let a = (axis_index + 1) % 3;
                let b = (axis_index + 2) % 3;
                let x = p[a] - origin[a];
                let y = p[b] - origin[b];
                p[a] = origin[a] + x * angle.cos() - y * angle.sin();
                p[b] = origin[b] + x * angle.sin() + y * angle.cos();
            }
            "bend" => {
                if amount.abs() > f64::EPSILON {
                    let angle = amount * distance;
                    p[axis_index] = origin[axis_index] + angle.sin() / amount;
                    p[other] += (1.0 - angle.cos()) / amount;
                }
            }
            "taper" => {
                let scale = 1.0 + amount * distance;
                for component in 0..3 {
                    if component != axis_index {
                        p[component] =
                            origin[component] + (p[component] - origin[component]) * scale;
                    }
                }
            }
            "stretch" => {
                let scale = 1.0 + amount * distance.abs();
                p[axis_index] = origin[axis_index] + distance * scale;
                let inverse = if scale.abs() > 1.0e-12 {
                    scale.sqrt().recip()
                } else {
                    1.0
                };
                for component in 0..3 {
                    if component != axis_index {
                        p[component] =
                            origin[component] + (p[component] - origin[component]) * inverse;
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn wave(vertices: &mut [SceneVertex], params: &JsonMap, time: f64) -> Result<(), String> {
    require_params(
        params,
        &["amplitude", "wavelength", "speed", "axis"],
        &["phase", "decay", "origin"],
        "deform.wave",
    )?;
    let amplitude = number(params, "amplitude")?;
    let wavelength = number(params, "wavelength")?;
    if wavelength <= 0.0 {
        return Err("deform.wave wavelength must be positive".to_owned());
    }
    let speed = number(params, "speed")?;
    let phase = params
        .get("phase")
        .map(|_| number(params, "phase"))
        .transpose()?
        .unwrap_or(0.0);
    let decay = params
        .get("decay")
        .map(|_| number(params, "decay"))
        .transpose()?
        .unwrap_or(0.0);
    if decay < 0.0 {
        return Err("deform.wave decay must be nonnegative".to_owned());
    }
    let axis = axis(string(params, "axis")?)?;
    let origin = vector3(params, "origin")?.unwrap_or([0.0; 3]);
    let radial = (0..3)
        .filter(|component| *component != axis)
        .collect::<Vec<_>>();
    for vertex in vertices {
        let p = &mut vertex.position;
        let distance = ((p[radial[0]] - origin[radial[0]]).powi(2)
            + (p[radial[1]] - origin[radial[1]]).powi(2))
        .sqrt();
        let fade = if decay > 0.0 {
            (-distance / decay).exp()
        } else {
            1.0
        };
        let angle = std::f64::consts::TAU * distance / wavelength - speed * time + phase;
        p[axis] += amplitude * angle.sin() * fade;
    }
    Ok(())
}

fn cast(vertices: &mut [SceneVertex], params: &JsonMap) -> Result<(), String> {
    require_params(
        params,
        &["shape", "factor"],
        &["origin", "axis"],
        "deform.cast",
    )?;
    let shape = string(params, "shape")?;
    if !matches!(shape, "sphere" | "cylinder" | "cuboid") {
        return Err(format!("unsupported cast target {shape:?}"));
    }
    let factor = number(params, "factor")?;
    if !(0.0..=1.0).contains(&factor) {
        return Err("deform.cast factor must be in [0, 1]".to_owned());
    }
    let center = vector3(params, "origin")?.unwrap_or([0.0; 3]);
    let axis = axis(
        params
            .get("axis")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("z"),
    )?;
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for vertex in vertices.iter() {
        for component in 0..3 {
            low[component] = low[component].min(vertex.position[component]);
            high[component] = high[component].max(vertex.position[component]);
        }
    }
    let radius = (0..3)
        .filter(|component| *component != axis)
        .map(|component| (high[component] - low[component]) * 0.5)
        .sum::<f64>()
        * 0.5;
    for vertex in vertices {
        let p = &mut vertex.position;
        let delta = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
        let target = match shape {
            "sphere" => {
                let length = dot(delta, delta).sqrt();
                if length <= 1.0e-12 {
                    continue;
                }
                [
                    center[0] + delta[0] / length * radius,
                    center[1] + delta[1] / length * radius,
                    center[2] + delta[2] / length * radius,
                ]
            }
            "cylinder" => {
                let a = (axis + 1) % 3;
                let b = (axis + 2) % 3;
                let length = (delta[a] * delta[a] + delta[b] * delta[b]).sqrt();
                if length <= 1.0e-12 {
                    continue;
                }
                let mut q = *p;
                q[a] = center[a] + delta[a] / length * radius;
                q[b] = center[b] + delta[b] / length * radius;
                q
            }
            _ => {
                let mut q = *p;
                for component in 0..3 {
                    let half = (high[component] - low[component]) * 0.5;
                    q[component] = center[component] + delta[component].signum() * half;
                }
                q
            }
        };
        for component in 0..3 {
            p[component] += (target[component] - p[component]) * factor;
        }
    }
    Ok(())
}

fn smooth(vertices: &mut [SceneVertex], params: &JsonMap) -> Result<(), String> {
    crate::validation::require_keys(params, &["factor", "iterations"], "deform.smooth params")?;
    let factor = number(params, "factor")?;
    if !(0.0..=1.0).contains(&factor) {
        return Err("deform.smooth factor must be in [0, 1]".to_owned());
    }
    let iterations = params
        .get("iterations")
        .and_then(serde_json::Value::as_u64)
        .ok_or("deform.smooth iterations must be an unsigned integer")?;
    if iterations == 0 || iterations > 64 {
        return Err("deform.smooth iterations must be in 1..=64".to_owned());
    }
    let (mut positions, vertex_ids, mut neighbors) = indexed_positions(vertices);
    for list in &mut neighbors {
        list.sort_unstable();
        list.dedup();
    }
    for _ in 0..iterations {
        let previous = positions.clone();
        for (index, adjacent) in neighbors.iter().enumerate() {
            if adjacent.is_empty() {
                continue;
            }
            let mut average = [0.0; 3];
            for neighbor in adjacent {
                for axis in 0..3 {
                    average[axis] += previous[*neighbor][axis] / adjacent.len() as f64;
                }
            }
            for axis in 0..3 {
                positions[index][axis] += (average[axis] - previous[index][axis]) * factor;
            }
        }
    }
    for (vertex, index) in vertices.iter_mut().zip(vertex_ids) {
        vertex.position = positions[index];
    }
    Ok(())
}

fn displace(
    vertices: &mut [SceneVertex],
    inputs: &BTreeMap<String, crate::model::Input>,
    params: &JsonMap,
    assets: &BTreeMap<String, AssetRecord>,
) -> Result<(), String> {
    crate::validation::require_keys(
        params,
        &["strength", "midlevel", "axis"],
        "deform.displace params",
    )?;
    let Some(crate::model::Input::Record { table, record }) = inputs.get("texture") else {
        return Err("deform.displace texture must reference an asset record".to_owned());
    };
    if table != "assets" {
        return Err("deform.displace texture reference must use assets".to_owned());
    }
    let asset = assets
        .get(record)
        .ok_or_else(|| format!("missing displacement texture {record:?}"))?;
    if asset.kind != "image" {
        return Err(format!(
            "displacement texture {record:?} is not an image asset"
        ));
    }
    let image: &ImageData = asset
        .image
        .as_deref()
        .ok_or("displacement texture has no decoded image")?;
    let strength = number(params, "strength")?;
    let midlevel = number(params, "midlevel")?;
    let axis_value = string(params, "axis")?;
    let direction = match axis_value {
        "normal" => None,
        value => Some(axis(value)?),
    };
    for vertex in vertices {
        let uv = vertex.texture_uvs[0];
        let x = ((uv[0].clamp(0.0, 1.0) * (image.width.saturating_sub(1)) as f64).round() as u32)
            .min(image.width.saturating_sub(1));
        let y = ((uv[1].clamp(0.0, 1.0) * (image.height.saturating_sub(1)) as f64).round() as u32)
            .min(image.height.saturating_sub(1));
        let offset = ((y * image.width + x) * 4) as usize;
        let sample = image
            .rgba
            .get(offset..offset + 4)
            .ok_or("displacement texture has invalid pixels")?;
        let value =
            (0.2126 * sample[0] as f64 + 0.7152 * sample[1] as f64 + 0.0722 * sample[2] as f64)
                / 255.0;
        let amount = (value - midlevel) * strength;
        if let Some(axis) = direction {
            vertex.position[axis] += amount;
        } else {
            for axis in 0..3 {
                vertex.position[axis] += vertex.normal[axis] * amount;
            }
        }
    }
    Ok(())
}

fn hook(vertices: &mut [SceneVertex], params: &JsonMap) -> Result<(), String> {
    require_params(
        params,
        &["center", "offset", "radius", "strength"],
        &["falloff"],
        "deform.hook",
    )?;
    let center = vector3(params, "center")?.ok_or("deform.hook needs center")?;
    let offset = vector3(params, "offset")?.ok_or("deform.hook needs offset")?;
    let radius = number(params, "radius")?;
    let strength = number(params, "strength")?;
    if radius <= 0.0 {
        return Err("deform.hook radius must be positive".to_owned());
    }
    let falloff = params
        .get("falloff")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("smooth");
    if !matches!(falloff, "constant" | "linear" | "smooth") {
        return Err(format!("unsupported hook falloff {falloff:?}"));
    }
    for vertex in vertices {
        let distance = ((vertex.position[0] - center[0]).powi(2)
            + (vertex.position[1] - center[1]).powi(2)
            + (vertex.position[2] - center[2]).powi(2))
        .sqrt();
        let t = (distance / radius).clamp(0.0, 1.0);
        let weight = if distance >= radius {
            0.0
        } else {
            match falloff {
                "constant" => 1.0,
                "linear" => 1.0 - t,
                _ => 1.0 - t * t * (3.0 - 2.0 * t),
            }
        };
        for axis in 0..3 {
            vertex.position[axis] += offset[axis] * strength * weight;
        }
    }
    Ok(())
}

fn lattice(vertices: &mut [SceneVertex], params: &JsonMap) -> Result<(), String> {
    crate::validation::require_keys(
        params,
        &[
            "bounds_min",
            "bounds_max",
            "dimensions",
            "points",
            "strength",
        ],
        "deform.lattice params",
    )?;
    let bounds_min = vector3(params, "bounds_min")?.ok_or("deform.lattice needs bounds_min")?;
    let bounds_max = vector3(params, "bounds_max")?.ok_or("deform.lattice needs bounds_max")?;
    if (0..3).any(|axis| bounds_min[axis] >= bounds_max[axis]) {
        return Err("deform.lattice bounds must have positive extent".to_owned());
    }
    let dimensions: [usize; 3] = serde_json::from_value(
        params
            .get("dimensions")
            .cloned()
            .ok_or("deform.lattice needs dimensions")?,
    )
    .map_err(|error| format!("invalid lattice dimensions: {error}"))?;
    if dimensions
        .iter()
        .any(|dimension| !(2..=8).contains(dimension))
    {
        return Err("deform.lattice dimensions must each be in 2..=8".to_owned());
    }
    let expected = dimensions[0] * dimensions[1] * dimensions[2];
    let points: Vec<[f64; 3]> = serde_json::from_value(
        params
            .get("points")
            .cloned()
            .ok_or("deform.lattice needs points")?,
    )
    .map_err(|error| format!("invalid lattice points: {error}"))?;
    if points.len() != expected || points.iter().flatten().any(|value| !value.is_finite()) {
        return Err(format!(
            "deform.lattice needs {expected} finite control points"
        ));
    }
    let strength = number(params, "strength")?;
    if !(0.0..=1.0).contains(&strength) {
        return Err("deform.lattice strength must be in [0, 1]".to_owned());
    }
    let index = |x: usize, y: usize, z: usize| x + dimensions[0] * (y + dimensions[1] * z);
    for vertex in vertices {
        let original = vertex.position;
        let u = std::array::from_fn::<_, 3, _>(|axis| {
            ((original[axis] - bounds_min[axis]) / (bounds_max[axis] - bounds_min[axis]))
                .clamp(0.0, 1.0)
                * (dimensions[axis] - 1) as f64
        });
        let base = u.map(|value| value.floor() as usize);
        let next =
            std::array::from_fn::<_, 3, _>(|axis| (base[axis] + 1).min(dimensions[axis] - 1));
        let frac = std::array::from_fn::<_, 3, _>(|axis| u[axis] - base[axis] as f64);
        let mut target = [0.0; 3];
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let ix = if x == 0 { base[0] } else { next[0] };
                    let iy = if y == 0 { base[1] } else { next[1] };
                    let iz = if z == 0 { base[2] } else { next[2] };
                    let weight = (if x == 0 { 1.0 - frac[0] } else { frac[0] })
                        * (if y == 0 { 1.0 - frac[1] } else { frac[1] })
                        * (if z == 0 { 1.0 - frac[2] } else { frac[2] });
                    let point = points[index(ix, iy, iz)];
                    for axis in 0..3 {
                        target[axis] += point[axis] * weight;
                    }
                }
            }
        }
        for axis in 0..3 {
            vertex.position[axis] = original[axis] + (target[axis] - original[axis]) * strength;
        }
    }
    Ok(())
}

fn warp(vertices: &mut [SceneVertex], params: &JsonMap) -> Result<(), String> {
    require_params(
        params,
        &["from", "to", "radius", "strength"],
        &["falloff"],
        "deform.warp",
    )?;
    let from = vector3(params, "from")?.ok_or("deform.warp needs from")?;
    let to = vector3(params, "to")?.ok_or("deform.warp needs to")?;
    let radius = number(params, "radius")?;
    let strength = number(params, "strength")?;
    if radius <= 0.0 {
        return Err("deform.warp radius must be positive".to_owned());
    }
    let falloff = params
        .get("falloff")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("smooth");
    if !matches!(falloff, "constant" | "linear" | "smooth") {
        return Err(format!("unsupported warp falloff {falloff:?}"));
    }
    let offset = sub(to, from);
    for vertex in vertices {
        let delta = sub(vertex.position, from);
        let distance = dot(delta, delta).sqrt();
        let t = (distance / radius).clamp(0.0, 1.0);
        let weight = if distance >= radius {
            0.0
        } else {
            match falloff {
                "constant" => 1.0,
                "linear" => 1.0 - t,
                _ => 1.0 - t * t * (3.0 - 2.0 * t),
            }
        };
        for axis in 0..3 {
            vertex.position[axis] += offset[axis] * strength * weight;
        }
    }
    Ok(())
}

fn curve(vertices: &mut [SceneVertex], params: &JsonMap) -> Result<(), String> {
    crate::validation::require_keys(
        params,
        &["points", "axis", "strength"],
        "deform.curve params",
    )?;
    let points: Vec<[f64; 3]> = serde_json::from_value(
        params
            .get("points")
            .cloned()
            .ok_or("deform.curve needs points")?,
    )
    .map_err(|error| format!("invalid curve points: {error}"))?;
    if points.len() < 2
        || points.len() > 4096
        || points.iter().flatten().any(|value| !value.is_finite())
    {
        return Err("deform.curve needs 2..=4096 finite points".to_owned());
    }
    let axis = axis(string(params, "axis")?)?;
    let strength = number(params, "strength")?;
    if !(0.0..=1.0).contains(&strength) {
        return Err("deform.curve strength must be in [0, 1]".to_owned());
    }
    let across: Vec<usize> = (0..3).filter(|component| *component != axis).collect();
    let low = vertices
        .iter()
        .map(|vertex| vertex.position[axis])
        .fold(f64::INFINITY, f64::min);
    let high = vertices
        .iter()
        .map(|vertex| vertex.position[axis])
        .fold(f64::NEG_INFINITY, f64::max);
    let center = vertices.iter().fold([0.0; 3], |mut sum, vertex| {
        for component in 0..3 {
            sum[component] += vertex.position[component] / vertices.len() as f64;
        }
        sum
    });
    let lengths: Vec<f64> = points
        .windows(2)
        .map(|pair| length(sub(pair[1], pair[0])))
        .collect();
    let total_length: f64 = lengths.iter().sum();
    if total_length <= 1.0e-12 {
        return Err("deform.curve points must not all coincide".to_owned());
    }
    for vertex in vertices {
        let t = if high > low {
            ((vertex.position[axis] - low) / (high - low)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let target_distance = t * total_length;
        let mut accumulated = 0.0;
        let mut segment = lengths.len() - 1;
        for (index, segment_length) in lengths.iter().enumerate() {
            if target_distance <= accumulated + segment_length {
                segment = index;
                break;
            }
            accumulated += segment_length;
        }
        let local_t =
            ((target_distance - accumulated) / lengths[segment].max(1.0e-12)).clamp(0.0, 1.0);
        let tangent = normalize(sub(points[segment + 1], points[segment]));
        let path_point: [f64; 3] = std::array::from_fn(|component| {
            points[segment][component] * (1.0 - local_t) + points[segment + 1][component] * local_t
        });
        let up = if tangent[2].abs() > 0.95 {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        };
        let normal = normalize(cross(tangent, up));
        let binormal = normalize(cross(tangent, normal));
        let offset_a = vertex.position[across[0]] - center[across[0]];
        let offset_b = vertex.position[across[1]] - center[across[1]];
        let target: [f64; 3] = std::array::from_fn(|component| {
            path_point[component] + normal[component] * offset_a + binormal[component] * offset_b
        });
        for component in 0..3 {
            vertex.position[component] +=
                (target[component] - vertex.position[component]) * strength;
        }
    }
    Ok(())
}

fn recalculate_normals(vertices: &mut [SceneVertex]) {
    for triangle in vertices.chunks_exact_mut(3) {
        let a = triangle[0].position;
        let b = triangle[1].position;
        let c = triangle[2].position;
        let normal = normalize(cross(sub(b, a), sub(c, a)));
        for vertex in triangle {
            vertex.normal = normal;
        }
    }
}

fn indexed_positions(vertices: &[SceneVertex]) -> (Vec<[f64; 3]>, Vec<usize>, Vec<Vec<usize>>) {
    let mut ids = BTreeMap::<[u64; 3], usize>::new();
    let mut positions = Vec::new();
    let mut vertex_ids = Vec::with_capacity(vertices.len());
    for vertex in vertices {
        let key = vertex
            .position
            .map(|value| if value == 0.0 { 0 } else { value.to_bits() });
        let index = *ids.entry(key).or_insert_with(|| {
            positions.push(vertex.position);
            positions.len() - 1
        });
        vertex_ids.push(index);
    }
    let mut neighbors = vec![BTreeSet::new(); positions.len()];
    for triangle in vertex_ids.chunks_exact(3) {
        for (a, b) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            if a != b {
                neighbors[a].insert(b);
                neighbors[b].insert(a);
            }
        }
    }
    (
        positions,
        vertex_ids,
        neighbors
            .into_iter()
            .map(|set| set.into_iter().collect())
            .collect(),
    )
}

fn string<'a>(params: &'a JsonMap, key: &str) -> Result<&'a str, String> {
    params
        .get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("deformation needs string {key:?}"))
}

fn require_params(
    params: &JsonMap,
    required: &[&str],
    optional: &[&str],
    label: &str,
) -> Result<(), String> {
    if required.iter().any(|key| !params.contains_key(*key))
        || params
            .keys()
            .any(|key| !required.contains(&key.as_str()) && !optional.contains(&key.as_str()))
    {
        return Err(format!(
            "{label} params require [{}] and allow [{}]",
            required.join(", "),
            optional.join(", ")
        ));
    }
    Ok(())
}

fn number(params: &JsonMap, key: &str) -> Result<f64, String> {
    let value = params
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("deformation needs number {key:?}"))?;
    if !value.is_finite() {
        return Err(format!("deformation parameter {key:?} must be finite"));
    }
    Ok(value)
}
fn vector3(params: &JsonMap, key: &str) -> Result<Option<[f64; 3]>, String> {
    params
        .get(key)
        .map(|value| {
            let vector: [f64; 3] = serde_json::from_value(value.clone())
                .map_err(|error| format!("invalid {key}: {error}"))?;
            if !vector.iter().all(|value| value.is_finite()) {
                return Err(format!("{key} values must be finite"));
            }
            Ok(vector)
        })
        .transpose()
}
fn axis(value: &str) -> Result<usize, String> {
    match value {
        "x" => Ok(0),
        "y" => Ok(1),
        "z" => Ok(2),
        _ => Err(format!("unsupported deformation axis {value:?}")),
    }
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn length(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let length = dot(v, v).sqrt();
    if length <= 1.0e-12 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / length, v[1] / length, v[2] / length]
    }
}
