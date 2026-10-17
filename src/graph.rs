use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::*;

const MAX_GRAPH_GEOMETRY_VERTICES: usize = 1_000_000;

pub(super) fn evaluate_batch_graph(
    project: &Project,
    project_root: &Path,
    graph: &Graph,
    output: &NodePort,
    composition_id: &str,
    object_id: &str,
    frame: u64,
    fps: FractionValue,
    timeline_ids: &[String],
    profile: quality::RenderProfile,
) -> Result<Vec<InstanceSpec>, String> {
    require(graph.outputs.len() == 1, "MVP graph must expose one output")?;
    require(
        output.port == "batch",
        "MVP graph output must be a batch port",
    )?;
    let node = graph
        .nodes
        .get(&output.node)
        .ok_or_else(|| format!("missing graph node {:?}", output.node))?;
    let Node::Native {
        operation,
        version,
        inputs,
        params,
    } = node;
    require(*version == 1, "unsupported batch operation version")?;
    match operation.as_str() {
        "mogf.batch.explicit" => evaluate_explicit_batch(
            project,
            graph,
            output,
            inputs,
            params,
            frame as f64 * fps.d as f64 / fps.n as f64,
        ),
        "mogf.text.shaped" => evaluate_shaped_text(
            project,
            project_root,
            graph,
            output,
            inputs,
            params,
            composition_id,
            object_id,
            frame,
            fps,
            timeline_ids,
            profile,
        ),
        _ => Err(format!(
            "unsupported batch-producing operation {operation:?}"
        )),
    }
}

fn evaluate_explicit_batch(
    project: &Project,
    graph: &Graph,
    _output: &NodePort,
    inputs: &BTreeMap<String, Input>,
    params: &JsonMap,
    time: f64,
) -> Result<Vec<InstanceSpec>, String> {
    validation::require_keys(inputs, &["geometry"], "batch.explicit inputs")?;
    validation::require_keys(params, &["rows"], "batch.explicit params")?;
    let Some(Input::Node {
        node: prototype_id,
        port,
    }) = inputs.get("geometry")
    else {
        return Err("batch.explicit geometry must reference a graph node".to_owned());
    };
    require(
        port == "geometry",
        "batch geometry input has the wrong port",
    )?;
    let mut visited = std::collections::BTreeSet::new();
    let geometry = evaluate_geometry(project, graph, prototype_id, port, &mut visited, time)?;
    require(
        graph.nodes.len() == visited.len() + 1,
        "explicit-batch graph contains nodes outside its geometry chain",
    )?;
    let rows: Vec<BatchRow> =
        serde_json::from_value(params.get("rows").cloned().ok_or("batch needs rows")?)
            .map_err(|e| format!("invalid batch rows: {e}"))?;
    require(rows.len() <= 100_000, "batch exceeds the MVP row limit")?;
    let mut ids = std::collections::BTreeSet::new();
    rows.into_iter()
        .map(|row| {
            require(
                ids.insert(row.id.clone()),
                format!("duplicate instance ID {:?}", row.id),
            )?;
            let material = project
                .materials
                .get(&row.material)
                .ok_or_else(|| format!("missing material {:?}", row.material))?;
            let (shading, lighting, shader_params) = material.gpu_shading();
            let image = material
                .params
                .image
                .as_ref()
                .map(|id| {
                    require(geometry.is_plane, "image materials require planar geometry")?;
                    let asset = project
                        .assets
                        .get(id)
                        .ok_or_else(|| format!("missing image asset {id:?}"))?;
                    require(
                        asset.kind == "image",
                        "material image must reference an image asset",
                    )?;
                    asset
                        .image
                        .clone()
                        .ok_or_else(|| format!("image asset {id:?} has no decoded pixels"))
                })
                .transpose()?;
            validation::finite_transform(&row.placement)?;
            Ok(InstanceSpec {
                instance_id: row.id.clone(),
                prototype: prototype_id.clone(),
                geometry_kind: if geometry.modified {
                    "scene_mesh".to_owned()
                } else {
                    geometry.kind.clone()
                },
                size: if geometry.modified {
                    [1.0; 3]
                } else {
                    geometry.size
                },
                material: row.material,
                image,
                color: material.params.color,
                shading,
                lighting,
                shader_params,
                shader_program: material.params.shader.clone(),
                casts_shadow: material.params.casts_shadow,
                receives_shadow: material.params.receives_shadow,
                placement: Transform {
                    translation: row.placement.translation,
                    rotation: normalized(row.placement.rotation)?,
                    scale: row.placement.scale,
                },
                selection_ids: vec![row.id],
                character_ids: Vec::new(),
                word_id: None,
                glyph_mesh: None,
                glyph_hash: geometry.modified.then_some(geometry.hash),
                scene_vertices: geometry.modified.then(|| Arc::clone(&geometry.vertices)),
            })
        })
        .collect()
}

struct EvaluatedGeometry {
    vertices: Arc<[SceneVertex]>,
    is_plane: bool,
    hash: u64,
    kind: String,
    size: [f64; 3],
    modified: bool,
}

fn evaluate_geometry(
    project: &Project,
    graph: &Graph,
    node_id: &str,
    port: &str,
    visited: &mut std::collections::BTreeSet<String>,
    time: f64,
) -> Result<EvaluatedGeometry, String> {
    require(
        port == "geometry",
        "geometry chain uses a non-geometry port",
    )?;
    require(
        visited.insert(node_id.to_owned()),
        format!("geometry graph contains a cycle or repeated node at {node_id:?}"),
    )?;
    let node = graph
        .nodes
        .get(node_id)
        .ok_or_else(|| format!("missing geometry node {node_id:?}"))?;
    let Node::Native {
        operation,
        version,
        inputs,
        params,
    } = node;
    require(
        *version == 1,
        format!("unsupported operation version for {operation:?}"),
    )?;
    if operation == "mogf.geometry.plane" || operation == "mogf.geometry.box" {
        require(
            inputs.is_empty(),
            format!("{operation} does not accept inputs"),
        )?;
        let mut geometry = if operation == "mogf.geometry.plane" {
            plane_geometry(params)?
        } else {
            box_geometry(params)?
        };
        recalculate_normals(&mut geometry.vertices);
        let is_plane = operation == "mogf.geometry.plane";
        let size = if is_plane {
            let dimensions: [f64; 2] =
                serde_json::from_value(params.get("size").cloned().ok_or("plane needs size")?)
                    .map_err(|e| format!("invalid plane size: {e}"))?;
            [dimensions[0], dimensions[1], 0.0]
        } else {
            serde_json::from_value(params.get("size").cloned().ok_or("box needs size")?)
                .map_err(|e| format!("invalid box size: {e}"))?
        };
        return Ok(finish_geometry(
            geometry.vertices,
            is_plane,
            if is_plane { "plane" } else { "box" }.to_owned(),
            size,
            false,
        ));
    }
    if operation == "mogf.deform.displace" {
        require(
            inputs.len() == 2 && inputs.contains_key("texture"),
            "deform.displace inputs need geometry and texture",
        )?;
    } else {
        validation::require_keys(inputs, &["geometry"], "deformation node inputs")?;
    }
    let Some(Input::Node {
        node: input_node,
        port: input_port,
    }) = inputs.get("geometry")
    else {
        return Err(format!(
            "{operation} geometry input must reference a graph node"
        ));
    };
    let mut geometry = evaluate_geometry(project, graph, input_node, input_port, visited, time)?;
    let mut vertices = geometry.vertices.to_vec();
    if operation == "mogf.geometry.subdivide" {
        validation::require_keys(params, &["levels"], "geometry.subdivide params")?;
        let levels = params
            .get("levels")
            .and_then(serde_json::Value::as_u64)
            .ok_or("geometry.subdivide levels must be an unsigned integer")?;
        if levels == 0 || levels > 5 {
            return Err("geometry.subdivide levels must be in 1..=5".to_owned());
        }
        let projected_vertices = (0..levels).try_fold(vertices.len(), |count, _| {
            count
                .checked_mul(4)
                .filter(|next| *next <= MAX_GRAPH_GEOMETRY_VERTICES)
        });
        if projected_vertices.is_none() {
            return Err(format!(
                "subdivided geometry exceeds {MAX_GRAPH_GEOMETRY_VERTICES} vertices"
            ));
        }
        for _ in 0..levels {
            vertices = subdivide(&vertices)?;
        }
        recalculate_normals(&mut vertices);
    } else {
        deformation::apply(
            operation,
            &mut vertices,
            inputs,
            params,
            &project.assets,
            time,
        )?;
    }
    geometry = finish_geometry(
        vertices,
        geometry.is_plane,
        geometry.kind,
        geometry.size,
        true,
    );
    Ok(geometry)
}

struct GeometryBuilder {
    vertices: Vec<SceneVertex>,
}

impl GeometryBuilder {
    fn push_triangle(
        &mut self,
        a: ([f64; 3], [f64; 2]),
        b: ([f64; 3], [f64; 2]),
        c: ([f64; 3], [f64; 2]),
    ) {
        let normal = normalized_cross(a.0, b.0, c.0);
        for (position, uv) in [a, b, c] {
            self.vertices.push(SceneVertex {
                position,
                normal,
                color: [1.0; 4],
                texture_uvs: [uv; 5],
            });
        }
    }
    fn quad(&mut self, p00: [f64; 3], p10: [f64; 3], p11: [f64; 3], p01: [f64; 3], segments: u32) {
        for y in 0..segments {
            for x in 0..segments {
                let u0 = x as f64 / segments as f64;
                let u1 = (x + 1) as f64 / segments as f64;
                let v0 = y as f64 / segments as f64;
                let v1 = (y + 1) as f64 / segments as f64;
                let point = |u: f64, v: f64| {
                    let mut out = [0.0; 3];
                    for axis in 0..3 {
                        out[axis] = p00[axis] * (1.0 - u) * (1.0 - v)
                            + p10[axis] * u * (1.0 - v)
                            + p11[axis] * u * v
                            + p01[axis] * (1.0 - u) * v;
                    }
                    out
                };
                let a = (point(u0, v0), [u0, v0]);
                let b = (point(u1, v0), [u1, v0]);
                let c = (point(u1, v1), [u1, v1]);
                let d = (point(u0, v1), [u0, v1]);
                self.push_triangle(a, b, c);
                self.push_triangle(a, c, d);
            }
        }
    }
}

fn plane_geometry(params: &JsonMap) -> Result<GeometryBuilder, String> {
    require_geometry_params(params, "geometry.plane")?;
    let size: [f64; 2] =
        serde_json::from_value(params.get("size").cloned().ok_or("plane needs size")?)
            .map_err(|e| format!("invalid plane size: {e}"))?;
    if !size.iter().all(|value| value.is_finite() && *value > 0.0) {
        return Err("plane dimensions must be finite and positive".to_owned());
    }
    let segments = params
        .get("segments")
        .map(|value| {
            serde_json::from_value::<u32>(value.clone())
                .map_err(|e| format!("invalid plane segments: {e}"))
        })
        .transpose()?
        .unwrap_or(1);
    if segments == 0 || segments > 256 {
        return Err("plane segments must be in 1..=256".to_owned());
    }
    let x = size[0] * 0.5;
    let y = size[1] * 0.5;
    let mut builder = GeometryBuilder {
        vertices: Vec::new(),
    };
    builder.quad(
        [-x, -y, 0.0],
        [x, -y, 0.0],
        [x, y, 0.0],
        [-x, y, 0.0],
        segments,
    );
    Ok(builder)
}

fn box_geometry(params: &JsonMap) -> Result<GeometryBuilder, String> {
    require_geometry_params(params, "geometry.box")?;
    let size: [f64; 3] =
        serde_json::from_value(params.get("size").cloned().ok_or("box needs size")?)
            .map_err(|e| format!("invalid box size: {e}"))?;
    if !size.iter().all(|value| value.is_finite() && *value > 0.0) {
        return Err("box dimensions must be finite and positive".to_owned());
    }
    let segments = params
        .get("segments")
        .map(|value| {
            serde_json::from_value::<u32>(value.clone())
                .map_err(|e| format!("invalid box segments: {e}"))
        })
        .transpose()?
        .unwrap_or(1);
    if segments == 0 || segments > 128 {
        return Err("box segments must be in 1..=128".to_owned());
    }
    let [x, y, z] = size.map(|value| value * 0.5);
    let mut builder = GeometryBuilder {
        vertices: Vec::new(),
    };
    builder.quad([-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z], segments);
    builder.quad([x, -y, -z], [-x, -y, -z], [-x, y, -z], [x, y, -z], segments);
    builder.quad([x, -y, z], [x, -y, -z], [x, y, -z], [x, y, z], segments);
    builder.quad([-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, -z], segments);
    builder.quad([-x, y, z], [x, y, z], [x, y, -z], [-x, y, -z], segments);
    builder.quad([-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z], segments);
    Ok(builder)
}

fn require_geometry_params(params: &JsonMap, label: &str) -> Result<(), String> {
    require(
        params.contains_key("size") && params.keys().all(|key| key == "size" || key == "segments"),
        format!("{label} params need size and may include segments"),
    )
}

fn subdivide(vertices: &[SceneVertex]) -> Result<Vec<SceneVertex>, String> {
    if vertices.len() % 3 != 0 {
        return Err("geometry triangle stream is malformed".to_owned());
    }
    let mut out = Vec::with_capacity(vertices.len().saturating_mul(4));
    let midpoint = |a: &SceneVertex, b: &SceneVertex| {
        let position = std::array::from_fn(|i| (a.position[i] + b.position[i]) * 0.5);
        let uv = std::array::from_fn(|i| (a.texture_uvs[0][i] + b.texture_uvs[0][i]) * 0.5);
        SceneVertex {
            position,
            normal: [0.0, 0.0, 1.0],
            color: [1.0; 4],
            texture_uvs: [uv; 5],
        }
    };
    for tri in vertices.chunks_exact(3) {
        let a = &tri[0];
        let b = &tri[1];
        let c = &tri[2];
        let ab = midpoint(a, b);
        let bc = midpoint(b, c);
        let ca = midpoint(c, a);
        for triangle in [
            [a.clone(), ab.clone(), ca.clone()],
            [ab.clone(), b.clone(), bc.clone()],
            [ca.clone(), bc.clone(), c.clone()],
            [ab, bc, ca],
        ] {
            out.extend(triangle);
        }
    }
    Ok(out)
}

fn recalculate_normals(vertices: &mut [SceneVertex]) {
    for triangle in vertices.chunks_exact_mut(3) {
        let normal = normalized_cross(
            triangle[0].position,
            triangle[1].position,
            triangle[2].position,
        );
        for vertex in triangle {
            vertex.normal = normal;
        }
    }
}

fn normalized_cross(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if length > 1.0e-12 {
        [n[0] / length, n[1] / length, n[2] / length]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn finish_geometry(
    vertices: Vec<SceneVertex>,
    is_plane: bool,
    kind: String,
    size: [f64; 3],
    modified: bool,
) -> EvaluatedGeometry {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for vertex in &vertices {
        for value in vertex.position {
            value.to_bits().hash(&mut hasher);
        }
        for value in vertex.texture_uvs[0] {
            value.to_bits().hash(&mut hasher);
        }
    }
    EvaluatedGeometry {
        vertices: Arc::from(vertices),
        is_plane,
        hash: hasher.finish(),
        kind,
        size,
        modified,
    }
}

fn evaluate_shaped_text(
    project: &Project,
    project_root: &Path,
    graph: &Graph,
    output: &NodePort,
    inputs: &BTreeMap<String, Input>,
    params: &JsonMap,
    composition_id: &str,
    object_id: &str,
    frame: u64,
    fps: FractionValue,
    timeline_ids: &[String],
    profile: quality::RenderProfile,
) -> Result<Vec<InstanceSpec>, String> {
    require(
        graph.nodes.len() == 1,
        "shaped-text graph must contain one node",
    )?;
    validation::require_keys(inputs, &["font"], "text.shaped inputs")?;
    let required = [
        "text",
        "size",
        "token_ids",
        "material",
        "tolerance",
        "quantization",
        "fill_rule",
        "extrusion_depth",
    ];
    require(
        required.iter().all(|key| params.contains_key(*key)),
        "text.shaped params are missing required fields",
    )?;
    require(
        params.keys().all(|key| {
            required.contains(&key.as_str()) || key == "max_width" || key == "font_weight"
        }),
        "text.shaped params contain an unsupported field",
    )?;
    let Some(Input::Record { table, record }) = inputs.get("font") else {
        return Err("text.shaped font must reference an asset record".to_owned());
    };
    require(
        table == "assets",
        "text.shaped font reference must use the assets table",
    )?;
    let asset = project
        .assets
        .get(record)
        .ok_or_else(|| format!("missing font asset {record:?}"))?;
    require(
        asset.kind == "font",
        "text.shaped input does not reference a font asset",
    )?;
    let font_path = resolve_asset(project_root, asset)?;
    let font_bytes = crate::imports::read_cached_asset(&font_path)
        .map_err(|error| format!("cannot read font {}: {error}", font_path.display()))?;
    let typed: TextParams = serde_json::to_value(params)
        .and_then(serde_json::from_value)
        .map_err(|error| format!("invalid text params: {error}"))?;
    require(
        typed.size.is_finite() && typed.size > 0.0,
        "text size must be finite and positive",
    )?;
    require(
        typed.tolerance.is_finite()
            && typed.tolerance > 0.0
            && typed.quantization.is_finite()
            && typed.quantization > 0.0,
        "text sampling values must be finite and positive",
    )?;
    require(
        typed.fill_rule == "nonzero" || typed.fill_rule == "evenodd",
        "unsupported text fill rule",
    )?;
    require(
        typed.extrusion_depth.is_finite() && (0.0..=2.0).contains(&typed.extrusion_depth),
        "text extrusion depth must be finite and in [0, 2]",
    )?;
    require(
        typed
            .max_width
            .is_none_or(|width| width.is_finite() && width > 0.0),
        "text max_width must be finite and positive",
    )?;
    let font_weight = animation::evaluated_text_weight(
        project,
        composition_id,
        object_id,
        frame,
        fps,
        timeline_ids,
        typed.font_weight,
    )?;
    let material = project
        .materials
        .get(&typed.material)
        .ok_or_else(|| format!("missing text material {:?}", typed.material))?;
    let (shading, lighting, shader_params) = material.gpu_shading();
    let font_stamp = crate::imports::SourceStamp::read(&font_path)
        .ok_or_else(|| format!("cannot stat font {}", font_path.display()))?;
    let glyphs = typography::shape_text_cached(
        &typed.text,
        &font_path,
        &font_stamp,
        &font_bytes,
        typed.size,
        &typed.token_ids,
        typed.max_width,
        font_weight as f32,
        typed.extrusion_depth,
        &typed.fill_rule,
        typed.tolerance * profile.text_tolerance_scale(),
    )?;
    glyphs
        .iter()
        .map(|glyph| {
            let pivot = [glyph.size[0] * 0.5, glyph.size[1] * 0.5];
            let mut selection_ids = glyph.character_ids.clone();
            if let Some(word_id) = &glyph.word_id {
                selection_ids.push(word_id.clone());
            }
            selection_ids.push(glyph.instance_id.clone());
            Ok(InstanceSpec {
                instance_id: glyph.instance_id.clone(),
                prototype: output.node.clone(),
                geometry_kind: "glyph_mesh".to_owned(),
                size: [glyph.size[0], glyph.size[1], typed.extrusion_depth],
                material: typed.material.clone(),
                image: None,
                color: material.params.color,
                shading,
                lighting,
                shader_params,
                shader_program: material.params.shader.clone(),
                casts_shadow: material.params.casts_shadow,
                receives_shadow: material.params.receives_shadow,
                placement: Transform {
                    translation: [glyph.origin[0] + pivot[0], glyph.origin[1] + pivot[1], 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0; 3],
                },
                selection_ids,
                character_ids: glyph.character_ids.clone(),
                word_id: glyph.word_id.clone(),
                glyph_mesh: Some(Arc::clone(&glyph.triangles)),
                glyph_hash: Some(glyph.geometry_hash),
                scene_vertices: None,
            })
        })
        .collect()
}
