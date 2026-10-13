use std::sync::Arc;

use super::*;

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
        "mogf.batch.explicit" => evaluate_explicit_batch(project, graph, output, inputs, params),
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
) -> Result<Vec<InstanceSpec>, String> {
    require(
        graph.nodes.len() == 2,
        "explicit-batch graph needs exactly a batch node and one prototype",
    )?;
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
    let prototype = graph
        .nodes
        .get(prototype_id)
        .ok_or_else(|| format!("missing prototype node {prototype_id:?}"))?;
    let Node::Native {
        operation,
        version,
        inputs,
        params: geometry_params,
    } = prototype;
    require(
        *version == 1 && inputs.is_empty(),
        "prototype version or inputs are unsupported",
    )?;
    let (geometry_kind, size) = match operation.as_str() {
        "mogf.geometry.plane" => {
            validation::require_keys(geometry_params, &["size"], "geometry.plane params")?;
            let dimensions: [f64; 2] = serde_json::from_value(
                geometry_params
                    .get("size")
                    .cloned()
                    .ok_or("plane needs size")?,
            )
            .map_err(|e| format!("invalid plane size: {e}"))?;
            require(
                dimensions.iter().all(|v| v.is_finite() && *v > 0.0),
                "plane dimensions must be finite and positive",
            )?;
            ("plane".to_owned(), [dimensions[0], dimensions[1], 0.0])
        }
        "mogf.geometry.box" => {
            validation::require_keys(geometry_params, &["size"], "geometry.box params")?;
            let dimensions: [f64; 3] = serde_json::from_value(
                geometry_params
                    .get("size")
                    .cloned()
                    .ok_or("box needs size")?,
            )
            .map_err(|e| format!("invalid box size: {e}"))?;
            require(
                dimensions.iter().all(|v| v.is_finite() && *v > 0.0),
                "box dimensions must be finite and positive",
            )?;
            ("box".to_owned(), dimensions)
        }
        _ => return Err(format!("unsupported geometry operation {operation:?}")),
    };
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
                    require(
                        geometry_kind == "plane",
                        "image materials require planar geometry",
                    )?;
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
                geometry_kind: geometry_kind.clone(),
                size,
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
                glyph_hash: None,
                scene_vertices: None,
            })
        })
        .collect()
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
