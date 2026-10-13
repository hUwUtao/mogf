//! Prepared static and animated scene assets from glTF/GLB and Wavefront OBJ.
//!
//! The imported representation keeps source node/primitive indices stable and
//! owns all decoded resources. It is renderer-independent and contains no GPU
//! handles.

use std::{
    collections::HashMap,
    fs, io,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use crate::model::ImageData;

pub(super) const MAX_SCENE_BYTES: u64 = 256 * 1024 * 1024;
pub(super) const MAX_VERTICES: usize = 4_000_000;
pub(super) const MAX_INDICES: usize = 12_000_000;
pub(super) const MAX_NODES: usize = 100_000;
pub(super) const MAX_DECODED_TEXTURE_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug)]
pub(super) struct SceneAsset {
    pub(super) source: SceneSource,
    /// SHA-256 of the primary source and every external dependency, computed
    /// after decode and used as the prepared-template revision for loadover.
    pub(super) revision: [u8; 32],
    pub(super) scenes: Vec<SceneDefinition>,
    pub(super) default_scene: Option<usize>,
    pub(super) nodes: Vec<SceneNode>,
    pub(super) meshes: Vec<SceneMesh>,
    pub(super) materials: Vec<SceneMaterial>,
    pub(super) animations: Vec<SceneAnimation>,
    pub(super) skins: Vec<SceneSkin>,
    pub(super) warnings: Vec<String>,
    /// Canonical external files read by the decoder, excluding the primary source.
    pub(super) dependencies: Vec<PathBuf>,
}

#[derive(Debug)]
pub(super) struct SceneDefinition {
    pub(super) index: usize,
    pub(super) name: String,
    pub(super) roots: Vec<usize>,
}

/// Open source-format identifier. Unlike a closed enum, adding a decoder does
/// not require extending the normalized asset model or dispatcher.
pub(super) type SceneSource = &'static str;

#[derive(Debug)]
pub(super) struct SceneNode {
    /// Source node index, never the non-unique source name.
    pub(super) index: usize,
    pub(super) name: String,
    pub(super) parent: Option<usize>,
    pub(super) children: Vec<usize>,
    pub(super) mesh: Option<usize>,
    pub(super) skin: Option<usize>,
    /// Node weights override mesh defaults; absent weights use mesh defaults.
    pub(super) morph_weights: Option<Vec<f64>>,
    pub(super) translation: [f64; 3],
    pub(super) rotation: [f64; 4],
    pub(super) scale: [f64; 3],
}

#[derive(Debug)]
pub(super) struct SceneMesh {
    pub(super) name: String,
    pub(super) primitives: Vec<ScenePrimitive>,
    pub(super) morph_weights: Vec<f64>,
}

#[derive(Debug)]
pub(super) struct ScenePrimitive {
    pub(super) positions: Vec<[f64; 3]>,
    pub(super) normals: Vec<[f64; 3]>,
    /// Tangent XYZ and handedness from glTF TANGENT, when authored.
    pub(super) tangents: Vec<[f64; 4]>,
    /// Linear vertex color from glTF COLOR_0, when authored.
    pub(super) colors: Vec<[f64; 4]>,
    /// Vertex UVs keyed by TEXCOORD_n set index.
    pub(super) texcoords: HashMap<u32, Vec<[f64; 2]>>,
    pub(super) indices: Vec<u32>,
    pub(super) joints: Vec<[u16; 4]>,
    pub(super) weights: Vec<[f64; 4]>,
    pub(super) morph_targets: Vec<SceneMorphTarget>,
    pub(super) material: Option<usize>,
    pub(super) mode: PrimitiveMode,
}

#[derive(Debug)]
pub(super) struct SceneMorphTarget {
    /// Missing attributes contribute zero deltas.
    pub(super) positions: Vec<[f64; 3]>,
    pub(super) normals: Vec<[f64; 3]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrimitiveMode {
    Triangles,
    Lines,
    Points,
}

#[derive(Debug, Clone)]
pub(super) struct SceneMaterial {
    pub(super) name: String,
    pub(super) base_color: [f64; 4],
    pub(super) metallic: f64,
    pub(super) roughness: f64,
    pub(super) clearcoat: f64,
    pub(super) emissive: [f64; 3],
    pub(super) emissive_strength: f64,
    pub(super) normal_scale: f64,
    pub(super) occlusion_strength: f64,
    pub(super) alpha_mode: AlphaMode,
    pub(super) alpha_cutoff: f64,
    pub(super) double_sided: bool,
    pub(super) base_color_texture: Option<SceneTextureBinding>,
    pub(super) normal_texture: Option<SceneTextureBinding>,
    pub(super) metallic_roughness_texture: Option<SceneTextureBinding>,
    pub(super) occlusion_texture: Option<SceneTextureBinding>,
    pub(super) emissive_texture: Option<SceneTextureBinding>,
}

/// A glTF texture use binds image data together with per-use sampling state.
/// The same image can therefore be used with different UV sets or samplers.
#[derive(Debug, Clone)]
pub(super) struct SceneTextureBinding {
    pub(super) image: Arc<ImageData>,
    pub(super) tex_coord: u32,
    pub(super) sampler: SceneSampler,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct SceneSampler {
    pub(super) wrap_s: u32,
    pub(super) wrap_t: u32,
    /// glTF enum values; absent filters retain the source's unspecified state.
    pub(super) mag_filter: Option<u32>,
    pub(super) min_filter: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

#[derive(Debug)]
pub(super) struct SceneAnimation {
    pub(super) name: String,
    pub(super) start_time: f64,
    pub(super) duration: f64,
    pub(super) channels: Vec<SceneAnimationChannel>,
}

#[derive(Debug)]
pub(super) struct SceneAnimationChannel {
    pub(super) node: usize,
    pub(super) property: AnimatedProperty,
    pub(super) interpolation: Interpolation,
    pub(super) times: Vec<f64>,
    /// Flattened samples; CUBICSPLINE stores [in tangent, value, out tangent]
    /// per key. TRS samples have 3/4 components; weight samples have one
    /// component per morph target, without a fixed target-count ceiling.
    pub(super) values: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum AnimatedProperty {
    Translation,
    Rotation,
    Scale,
    Weights,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Interpolation {
    Step,
    Linear,
    CubicSpline,
}

#[derive(Debug)]
pub(super) struct SceneSkin {
    pub(super) name: String,
    pub(super) joints: Vec<usize>,
    pub(super) inverse_bind_matrices: Vec<[[f64; 4]; 4]>,
    pub(super) skeleton_root: Option<usize>,
}

/// Format-specific readers normalize source data into the renderer-independent
/// scene template. Runtime evaluation never needs to know which reader ran.
pub(super) trait SceneFormatLoader: Send + Sync {
    /// Stable normalized source-kind tag used for diagnostics and provenance.
    fn format(&self) -> SceneSource;
    /// Case-insensitive extension tokens, without a leading dot.
    fn extensions(&self) -> &'static [&'static str];
    /// Decode into the renderer-independent template; post-validation is
    /// applied centrally after this method returns.
    fn load(&self, context: &LoadContext<'_>) -> Result<SceneAsset, SceneLoadError>;
}

impl<T: SceneFormatLoader + ?Sized> SceneFormatLoader for Box<T> {
    fn format(&self) -> SceneSource {
        (**self).format()
    }

    fn extensions(&self) -> &'static [&'static str] {
        (**self).extensions()
    }

    fn load(&self, context: &LoadContext<'_>) -> Result<SceneAsset, SceneLoadError> {
        (**self).load(context)
    }
}

pub(super) struct LoadContext<'a> {
    pub(super) source: &'a Path,
    pub(super) allowed_root: &'a Path,
}

#[derive(Debug)]
pub(super) struct SceneLoadError {
    pub(super) format: SceneSource,
    pub(super) source: PathBuf,
    pub(super) message: String,
}

impl std::fmt::Display for SceneLoadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{:?} loader failed for {}: {}",
            self.format,
            self.source.display(),
            self.message
        )
    }
}

impl std::error::Error for SceneLoadError {}

fn texture_binding(
    texture: gltf::Texture<'_>,
    tex_coord: u32,
    images: &[Arc<ImageData>],
) -> SceneTextureBinding {
    let sampler = texture.sampler();
    SceneTextureBinding {
        image: Arc::clone(&images[texture.source().index()]),
        tex_coord,
        sampler: SceneSampler {
            wrap_s: sampler.wrap_s().as_gl_enum(),
            wrap_t: sampler.wrap_t().as_gl_enum(),
            mag_filter: sampler.mag_filter().map(|filter| filter.as_gl_enum()),
            min_filter: sampler.min_filter().map(|filter| filter.as_gl_enum()),
        },
    }
}

struct GltfLoader;
struct ObjLoader;
struct UsdLoader;

impl SceneFormatLoader for GltfLoader {
    fn format(&self) -> SceneSource {
        "gltf"
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["glb", "gltf"]
    }
    fn load(&self, context: &LoadContext<'_>) -> Result<SceneAsset, SceneLoadError> {
        load_gltf(context.source, context.allowed_root).map_err(|message| SceneLoadError {
            format: self.format(),
            source: context.source.to_path_buf(),
            message,
        })
    }
}

impl SceneFormatLoader for ObjLoader {
    fn format(&self) -> SceneSource {
        "obj"
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["obj"]
    }
    fn load(&self, context: &LoadContext<'_>) -> Result<SceneAsset, SceneLoadError> {
        load_obj(context.source, context.allowed_root).map_err(|message| SceneLoadError {
            format: self.format(),
            source: context.source.to_path_buf(),
            message,
        })
    }
}

impl SceneFormatLoader for UsdLoader {
    fn format(&self) -> SceneSource {
        "usd"
    }
    fn extensions(&self) -> &'static [&'static str] {
        &["usd", "usda", "usdc", "usdz"]
    }
    fn load(&self, context: &LoadContext<'_>) -> Result<SceneAsset, SceneLoadError> {
        load_usd(context.source, context.allowed_root).map_err(|message| SceneLoadError {
            format: self.format(),
            source: context.source.to_path_buf(),
            message,
        })
    }
}

pub(super) fn load(path: &Path, allowed_root: &Path) -> Result<SceneAsset, String> {
    let builtin_loaders: [Box<dyn SceneFormatLoader>; 3] = [
        Box::new(GltfLoader),
        Box::new(ObjLoader),
        Box::new(UsdLoader),
    ];
    load_with_loaders(path, allowed_root, &builtin_loaders)
}

/// Resolve, dispatch, and validate an imported scene using any loader
/// collection. `Box<dyn SceneFormatLoader>` supports heterogeneous adapters;
/// concrete loader arrays work too. Source-format policy stays outside the
/// evaluator, and every adapter shares one normalization/validation pipeline.
pub(super) fn load_with_loaders<L: SceneFormatLoader>(
    path: &Path,
    allowed_root: &Path,
    loaders: &[L],
) -> Result<SceneAsset, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("cannot stat scene asset {}: {error}", path.display()))?;
    if metadata.len() > MAX_SCENE_BYTES {
        return Err(format!("scene asset exceeds {MAX_SCENE_BYTES} bytes"));
    }
    let canonical = fs::canonicalize(path)
        .map_err(|error| format!("cannot resolve scene asset {}: {error}", path.display()))?;
    let root = fs::canonicalize(allowed_root).map_err(|error| {
        format!(
            "cannot resolve asset root {}: {error}",
            allowed_root.display()
        )
    })?;
    if !canonical.starts_with(&root) {
        return Err("scene asset resolves outside its declared asset root".to_owned());
    }
    let extension = canonical
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let context = LoadContext {
        source: &canonical,
        allowed_root: &root,
    };
    let mut matches = loaders.iter().filter(|loader| {
        loader
            .extensions()
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(&extension))
    });
    let loader = matches
        .next()
        .ok_or_else(|| format!("no scene format loader accepts .{extension}"))?;
    if matches.next().is_some() {
        return Err(format!("ambiguous scene format loader for .{extension}"));
    }
    let mut asset = loader.load(&context).map_err(|error| error.to_string())?;
    if asset.source != loader.format() {
        return Err(format!(
            "{:?} loader returned an asset tagged {:?}",
            loader.format(),
            asset.source
        ));
    }
    asset.revision = source_revision(&canonical, &asset.dependencies, &root, asset.source)?;
    validate_asset(&asset)?;
    asset.dependencies.sort();
    asset.dependencies.dedup();
    Ok(asset)
}

fn source_revision(
    source: &Path,
    dependencies: &[PathBuf],
    allowed_root: &Path,
    format: SceneSource,
) -> Result<[u8; 32], String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    const MAX_MANIFEST_BYTES: u64 = 2 * 1024 * 1024 * 1024;
    let mut paths = dependencies
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    paths.insert(source.to_path_buf());
    let mut manifest = std::collections::BTreeMap::new();
    for path in paths {
        let canonical = fs::canonicalize(&path).map_err(|error| {
            format!(
                "cannot resolve scene manifest dependency {}: {error}",
                path.display()
            )
        })?;
        let relative = canonical.strip_prefix(allowed_root).map_err(|_| {
            format!(
                "scene manifest dependency {} resolves outside its declared asset root",
                canonical.display()
            )
        })?;
        let relative = relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        if manifest.insert(relative.clone(), canonical).is_some() {
            return Err(format!("duplicate scene manifest path {relative:?}"));
        }
    }
    let mut total_bytes = 0_u64;
    let mut digest = Sha256::new();
    digest.update(b"MOGF-SCENE-MANIFEST\0v1\0");
    digest.update(format.as_bytes());
    digest.update([0]);
    for (relative, canonical) in manifest {
        let metadata = fs::metadata(&canonical).map_err(|error| {
            format!(
                "cannot stat scene manifest dependency {}: {error}",
                canonical.display()
            )
        })?;
        total_bytes = total_bytes
            .checked_add(metadata.len())
            .ok_or("scene dependency manifest size overflow")?;
        if total_bytes > MAX_MANIFEST_BYTES {
            return Err(format!(
                "scene source and dependencies exceed the {MAX_MANIFEST_BYTES}-byte manifest limit"
            ));
        }
        let path_bytes = relative.as_bytes();
        digest.update((path_bytes.len() as u64).to_le_bytes());
        digest.update(path_bytes);
        digest.update(metadata.len().to_le_bytes());
        let mut file = fs::File::open(&canonical).map_err(|error| {
            format!(
                "cannot read scene manifest dependency {}: {error}",
                canonical.display()
            )
        })?;
        let mut buffer = [0_u8; 64 * 1024];
        let mut read_bytes = 0_u64;
        loop {
            let count = file.read(&mut buffer).map_err(|error| {
                format!(
                    "cannot read scene manifest dependency {}: {error}",
                    canonical.display()
                )
            })?;
            if count == 0 {
                break;
            }
            read_bytes = read_bytes.saturating_add(count as u64);
            if read_bytes > metadata.len() {
                return Err(format!(
                    "scene manifest dependency {} grew while it was being fingerprinted",
                    canonical.display()
                ));
            }
            digest.update(&buffer[..count]);
        }
        if read_bytes != metadata.len() {
            return Err(format!(
                "scene manifest dependency {} changed while it was being fingerprinted",
                canonical.display()
            ));
        }
        let handle_metadata = file.metadata().map_err(|error| {
            format!(
                "cannot verify scene manifest dependency {}: {error}",
                canonical.display()
            )
        })?;
        let path_metadata = fs::metadata(&canonical).map_err(|error| {
            format!(
                "cannot recheck scene manifest dependency {}: {error}",
                canonical.display()
            )
        })?;
        if handle_metadata.len() != metadata.len()
            || handle_metadata.modified().ok() != metadata.modified().ok()
            || path_metadata.len() != metadata.len()
            || path_metadata.modified().ok() != metadata.modified().ok()
        {
            return Err(format!(
                "scene manifest dependency {} changed while it was being fingerprinted",
                canonical.display()
            ));
        }
    }
    Ok(digest.finalize().into())
}

/// Enforce the adapter contract before data can reach animation or rendering.
fn validate_asset(asset: &SceneAsset) -> Result<(), String> {
    let node_count = asset.nodes.len();
    if node_count > MAX_NODES {
        return Err(format!("scene exceeds {MAX_NODES} nodes"));
    }
    for (index, node) in asset.nodes.iter().enumerate() {
        if node.index != index {
            return Err(format!(
                "{:?} loader returned non-contiguous node index {} at slot {index}",
                asset.source, node.index
            ));
        }
        if !node
            .translation
            .iter()
            .chain(node.rotation.iter())
            .chain(node.scale.iter())
            .all(|value| value.is_finite())
        {
            return Err(format!("scene node {index} has a non-finite transform"));
        }
        if node.parent.is_some_and(|parent| parent >= node_count)
            || node.children.iter().any(|child| *child >= node_count)
        {
            return Err(format!(
                "scene node {index} has an out-of-range hierarchy edge"
            ));
        }
        if node
            .parent
            .is_some_and(|parent| !asset.nodes[parent].children.contains(&index))
            || node
                .children
                .iter()
                .any(|child| asset.nodes[*child].parent != Some(index))
        {
            return Err(format!(
                "scene node {index} has inconsistent hierarchy links"
            ));
        }
        if node.mesh.is_some_and(|mesh| mesh >= asset.meshes.len())
            || node.skin.is_some_and(|skin| skin >= asset.skins.len())
        {
            return Err(format!(
                "scene node {index} has an out-of-range mesh or skin reference"
            ));
        }
        if let Some(weights) = &node.morph_weights {
            let count = node
                .mesh
                .map(|mesh| asset.meshes[mesh].morph_weights.len())
                .unwrap_or(0);
            if node.mesh.is_none()
                || weights.len() != count
                || weights.iter().any(|weight| !weight.is_finite())
            {
                return Err(format!(
                    "scene node {index} has invalid morph weights; expected {count} finite weights on a mesh node"
                ));
            }
        }
    }
    if asset
        .default_scene
        .is_some_and(|index| index >= asset.scenes.len())
    {
        return Err("default scene index is out of range".to_owned());
    }
    for (index, scene) in asset.scenes.iter().enumerate() {
        if scene.index != index || scene.roots.iter().any(|root| *root >= node_count) {
            return Err(format!(
                "scene definition {index} has invalid identity or roots"
            ));
        }
        for root in &scene.roots {
            if asset.nodes[*root].parent.is_some() {
                return Err(format!("scene {index} root {root} has a parent"));
            }
        }
    }
    let mut colors = vec![0u8; node_count];
    for start in 0..node_count {
        if colors[start] == 2 {
            continue;
        }
        let mut stack = vec![(start, false)];
        while let Some((index, exiting)) = stack.pop() {
            if exiting {
                colors[index] = 2;
                continue;
            }
            match colors[index] {
                1 => return Err(format!("cycle in scene hierarchy at node {index}")),
                2 => continue,
                _ => {}
            }
            colors[index] = 1;
            stack.push((index, true));
            for child in asset.nodes[index].children.iter().rev() {
                if asset.nodes[*child].parent != Some(index) {
                    return Err(format!(
                        "scene hierarchy edge {index}->{child} is inconsistent"
                    ));
                }
                stack.push((*child, false));
            }
        }
    }
    if asset.meshes.len() > MAX_NODES {
        return Err("scene has too many mesh records".to_owned());
    }
    let mut total_vertices = 0usize;
    let mut total_indices = 0usize;
    let mut total_morph_vertices = 0usize;
    for (mesh_index, mesh) in asset.meshes.iter().enumerate() {
        let target_count = mesh
            .primitives
            .first()
            .map_or(0, |primitive| primitive.morph_targets.len());
        if mesh.morph_weights.len() != target_count
            || mesh.morph_weights.iter().any(|weight| !weight.is_finite())
        {
            return Err(format!(
                "mesh {mesh_index} has invalid default morph weights; expected {target_count} finite weights"
            ));
        }
        for (primitive_index, primitive) in mesh.primitives.iter().enumerate() {
            let label = format!("mesh {mesh_index} primitive {primitive_index}");
            if primitive.morph_targets.len() != target_count {
                return Err(format!(
                    "{label} has {} morph targets; mesh requires {target_count}",
                    primitive.morph_targets.len()
                ));
            }
            for (target_index, target) in primitive.morph_targets.iter().enumerate() {
                for (attribute, deltas, base_count) in [
                    ("POSITION", &target.positions, primitive.positions.len()),
                    ("NORMAL", &target.normals, primitive.normals.len()),
                ] {
                    if !deltas.is_empty()
                        && (deltas.len() != primitive.positions.len() || base_count == 0)
                    {
                        return Err(format!(
                            "{label} morph target {target_index} {attribute} count does not match its base attribute"
                        ));
                    }
                    if deltas.iter().flatten().any(|value| !value.is_finite()) {
                        return Err(format!(
                            "{label} morph target {target_index} {attribute} has non-finite deltas"
                        ));
                    }
                    total_morph_vertices = total_morph_vertices
                        .checked_add(deltas.len())
                        .ok_or("scene morph delta count overflow")?;
                    if total_morph_vertices > MAX_VERTICES {
                        return Err(format!(
                            "{label} morph target {target_index} exceeds aggregate morph attribute limit ({MAX_VERTICES} vectors)"
                        ));
                    }
                }
            }
            if primitive.positions.len() > MAX_VERTICES || primitive.indices.len() > MAX_INDICES {
                return Err(format!("{label} exceeds geometry limits"));
            }
            total_vertices = total_vertices
                .checked_add(primitive.positions.len())
                .ok_or("scene vertex count overflow")?;
            total_indices = total_indices
                .checked_add(primitive.indices.len())
                .ok_or("scene index count overflow")?;
            if total_vertices > MAX_VERTICES || total_indices > MAX_INDICES {
                return Err("scene geometry exceeds aggregate vertex/index limits".to_owned());
            }
            if primitive.normals.len() != 0 && primitive.normals.len() != primitive.positions.len()
            {
                return Err(format!("{label} normal count does not match positions"));
            }
            if primitive.tangents.len() != 0
                && primitive.tangents.len() != primitive.positions.len()
            {
                return Err(format!("{label} tangent count does not match positions"));
            }
            if primitive.colors.len() != 0 && primitive.colors.len() != primitive.positions.len() {
                return Err(format!("{label} color count does not match positions"));
            }
            if primitive
                .texcoords
                .values()
                .any(|coordinates| coordinates.len() != primitive.positions.len())
            {
                return Err(format!("{label} UV count does not match positions"));
            }
            if primitive.joints.len() != primitive.weights.len()
                || (!primitive.joints.is_empty()
                    && primitive.joints.len() != primitive.positions.len())
            {
                return Err(format!(
                    "{label} joint/weight counts do not match positions"
                ));
            }
            if primitive
                .positions
                .iter()
                .flatten()
                .chain(primitive.normals.iter().flatten())
                .chain(primitive.tangents.iter().flatten())
                .chain(primitive.colors.iter().flatten())
                .chain(primitive.texcoords.values().flatten().flatten())
                .chain(primitive.weights.iter().flatten())
                .any(|value| !value.is_finite())
            {
                return Err(format!("{label} contains non-finite vertex data"));
            }
            if primitive
                .indices
                .iter()
                .any(|index| *index as usize >= primitive.positions.len())
            {
                return Err(format!("{label} index is outside its vertex buffer"));
            }
            if primitive.mode == PrimitiveMode::Triangles && primitive.indices.len() % 3 != 0 {
                return Err(format!(
                    "{label} triangle index count is not divisible by three"
                ));
            }
            if primitive
                .material
                .is_some_and(|material| material >= asset.materials.len())
            {
                return Err(format!("{label} has an out-of-range material"));
            }
        }
    }
    for skin in &asset.skins {
        if skin.joints.len() != skin.inverse_bind_matrices.len()
            || skin.joints.iter().any(|joint| *joint >= node_count)
            || skin.skeleton_root.is_some_and(|root| root >= node_count)
        {
            return Err(format!(
                "skin {:?} has invalid joint or skeleton-root indices",
                skin.name
            ));
        }
        if skin
            .inverse_bind_matrices
            .iter()
            .flatten()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(format!(
                "skin {:?} has a non-finite inverse-bind matrix",
                skin.name
            ));
        }
    }
    for node in &asset.nodes {
        if let (Some(mesh_index), Some(skin_index)) = (node.mesh, node.skin) {
            let joint_count = asset.skins[skin_index].joints.len();
            for primitive in &asset.meshes[mesh_index].primitives {
                if primitive
                    .joints
                    .iter()
                    .flatten()
                    .any(|joint| usize::from(*joint) >= joint_count)
                {
                    return Err(format!(
                        "node {} has a vertex joint index outside its skin",
                        node.index
                    ));
                }
                if primitive.weights.iter().any(|weights| {
                    weights.iter().any(|weight| *weight < 0.0)
                        || weights.iter().sum::<f64>() <= 1e-12
                }) {
                    return Err(format!("node {} has invalid skin weights", node.index));
                }
            }
        }
    }
    for (material_index, material) in asset.materials.iter().enumerate() {
        if !material
            .base_color
            .iter()
            .chain(material.emissive.iter())
            .chain([
                &material.metallic,
                &material.roughness,
                &material.alpha_cutoff,
                &material.emissive_strength,
                &material.normal_scale,
                &material.occlusion_strength,
            ])
            .all(|value| value.is_finite())
        {
            return Err(format!(
                "material {material_index} has non-finite PBR attributes"
            ));
        }
        for image in [
            &material.base_color_texture,
            &material.normal_texture,
            &material.metallic_roughness_texture,
            &material.occlusion_texture,
            &material.emissive_texture,
        ]
        .into_iter()
        .flatten()
        {
            let expected = u64::from(image.image.width)
                .checked_mul(u64::from(image.image.height))
                .and_then(|pixels| pixels.checked_mul(4))
                .ok_or("scene texture dimensions overflow")?;
            if expected != image.image.rgba.len() as u64 {
                return Err(format!(
                    "material {material_index} has an invalid decoded texture"
                ));
            }
        }
    }
    for animation in &asset.animations {
        let mut destinations = std::collections::BTreeSet::new();
        for (channel_index, channel) in animation.channels.iter().enumerate() {
            let expected_values = if channel.interpolation == Interpolation::CubicSpline {
                channel.times.len().saturating_mul(3)
            } else {
                channel.times.len()
            };
            if channel.node >= node_count
                || !destinations.insert((channel.node, channel.property))
                || channel.times.is_empty()
                || channel.values.len() != expected_values
                || channel.times.iter().any(|time| !time.is_finite())
                || channel.times.windows(2).any(|keys| keys[0] >= keys[1])
                || channel
                    .values
                    .iter()
                    .flatten()
                    .any(|value| !value.is_finite())
            {
                return Err(format!(
                    "animation {:?} channel {channel_index} node {} has invalid timing, samples, or a duplicate destination",
                    animation.name, channel.node
                ));
            }
            let components = match channel.property {
                AnimatedProperty::Translation | AnimatedProperty::Scale => 3,
                AnimatedProperty::Rotation => 4,
                AnimatedProperty::Weights => {
                    let mesh = asset.nodes[channel.node].mesh.ok_or_else(|| format!("animation {:?} channel {channel_index} weights target node {} has no mesh", animation.name, channel.node))?;
                    let count = asset.meshes[mesh].morph_weights.len();
                    if count == 0 {
                        return Err(format!(
                            "animation {:?} channel {channel_index} weights target node {} has no morph targets",
                            animation.name, channel.node
                        ));
                    }
                    count
                }
            };
            if channel.values.iter().any(|value| value.len() != components) {
                return Err(format!(
                    "animation {:?} channel {channel_index} node {} has inconsistent sample cardinality; expected {components} components",
                    animation.name, channel.node
                ));
            }
        }
        if !animation.start_time.is_finite()
            || !animation.duration.is_finite()
            || animation.duration < 0.0
        {
            return Err(format!("animation {:?} has invalid timing", animation.name));
        }
    }
    Ok(())
}

#[cfg(test)]
mod loader_trait_tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct FixtureLoader;
    struct AlternateFixtureLoader;

    fn empty_asset(source: SceneSource) -> SceneAsset {
        SceneAsset {
            source,
            revision: [0; 32],
            scenes: Vec::new(),
            default_scene: None,
            nodes: Vec::new(),
            meshes: Vec::new(),
            materials: Vec::new(),
            animations: Vec::new(),
            skins: Vec::new(),
            warnings: Vec::new(),
            dependencies: Vec::new(),
        }
    }

    impl SceneFormatLoader for FixtureLoader {
        fn format(&self) -> SceneSource {
            "fixture"
        }
        fn extensions(&self) -> &'static [&'static str] {
            &["fixture"]
        }
        fn load(&self, _context: &LoadContext<'_>) -> Result<SceneAsset, SceneLoadError> {
            Ok(empty_asset(self.format()))
        }
    }

    impl SceneFormatLoader for AlternateFixtureLoader {
        fn format(&self) -> SceneSource {
            "alternate-fixture"
        }
        fn extensions(&self) -> &'static [&'static str] {
            &["alternate"]
        }
        fn load(&self, _context: &LoadContext<'_>) -> Result<SceneAsset, SceneLoadError> {
            Ok(empty_asset(self.format()))
        }
    }

    #[test]
    fn generic_dispatch_accepts_heterogeneous_boxed_adapters() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("mogf-loader-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("scene.fixture");
        let alternate_source = root.join("scene.alternate");
        fs::write(&source, b"fixture source").unwrap();
        fs::write(&alternate_source, b"alternate fixture source").unwrap();

        let loaders: [Box<dyn SceneFormatLoader>; 2] =
            [Box::new(FixtureLoader), Box::new(AlternateFixtureLoader)];
        let loaded = load_with_loaders(&source, &root, &loaders).unwrap();
        assert_eq!(loaded.source, "fixture");
        assert_ne!(loaded.revision, [0; 32]);
        let alternate = load_with_loaders(&alternate_source, &root, &loaders).unwrap();
        assert_eq!(alternate.source, "alternate-fixture");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn blender_glb_harness_retains_core_texture_bindings() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/glb-obj/harnesses");
        let asset = load(&root.join("scene_harness.glb"), &root).unwrap();
        let material = asset
            .materials
            .iter()
            .find(|material| material.name == "PBR_PaintedMetal")
            .expect("Blender harness material");
        let base = material.base_color_texture.as_ref().unwrap();
        let normal = material.normal_texture.as_ref().unwrap();
        let metallic_roughness = material.metallic_roughness_texture.as_ref().unwrap();
        let occlusion = material.occlusion_texture.as_ref().unwrap();
        let emissive = material.emissive_texture.as_ref().unwrap();
        assert_eq!(base.tex_coord, 1);
        assert_eq!(normal.tex_coord, 1);
        assert_ne!(base.image.hash, normal.image.hash);
        assert_eq!(metallic_roughness.tex_coord, 1);
        assert_eq!(occlusion.tex_coord, 0);
        assert_eq!(emissive.tex_coord, 1);
        assert_eq!(normal.sampler, base.sampler);
        assert!((material.normal_scale - 0.65).abs() < 1e-6);
        assert!((material.occlusion_strength - 0.4).abs() < 1e-6);
        assert_eq!(material.emissive_strength, 1.0);
        assert!(
            asset
                .meshes
                .iter()
                .flat_map(|mesh| &mesh.primitives)
                .any(|primitive| {
                    !primitive.tangents.is_empty()
                        && primitive.tangents.len() == primitive.positions.len()
                        && !primitive.colors.is_empty()
                        && primitive.colors.len() == primitive.positions.len()
                })
        );
        for (actual, expected) in material.emissive.into_iter().zip([0.2, 0.1, 0.05]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn blender_usd_text_binary_and_package_harnesses_load() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/glb-obj/harnesses");
        for extension in ["usda", "usdc", "usdz"] {
            let asset = load(&root.join(format!("scene_harness.{extension}")), &root).unwrap();
            assert_eq!(asset.source, "usd");
            assert!(!asset.nodes.is_empty(), "{extension} should contain prims");
            assert!(
                !asset.meshes.is_empty(),
                "{extension} should contain meshes"
            );
            let material = asset
                .materials
                .iter()
                .find(|material| material.name.contains("USD_PreviewSurface_Scalars"))
                .unwrap_or_else(|| {
                    panic!("{extension} should import bound PreviewSurface material")
                });
            assert!(
                (material.base_color[0] - 0.15).abs() < 1e-5,
                "{extension} diffuse red"
            );
            assert!(
                (material.base_color[1] - 0.55).abs() < 1e-5,
                "{extension} diffuse green"
            );
            assert!(
                (material.metallic - 0.35).abs() < 1e-5,
                "{extension} metallic"
            );
            assert!(
                (material.roughness - 0.27).abs() < 1e-5,
                "{extension} roughness"
            );
            assert!(
                asset
                    .meshes
                    .iter()
                    .flat_map(|mesh| &mesh.primitives)
                    .any(|primitive| primitive.material.is_some()),
                "{extension} binds imported material to geometry"
            );
        }
    }

    #[test]
    fn blender_obj_harness_loads_mtl_diffuse_texture() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/glb-obj/harnesses");
        let asset = load(&root.join("scene_harness.obj"), &root).unwrap();
        assert!(asset.materials.iter().any(|material| {
            material
                .base_color_texture
                .as_ref()
                .is_some_and(|binding| binding.image.width == 4 && binding.image.height == 4)
        }));
        assert!(
            asset
                .dependencies
                .iter()
                .any(|path| path.ends_with("HarnessChecker.png"))
        );
    }
}

fn checked_resource(root: &Path, base: &Path, uri: &str) -> Result<PathBuf, String> {
    if uri.starts_with("data:") {
        return Err("data URI should be handled by the format decoder".to_owned());
    }
    if uri.contains(':') {
        return Err(format!("external URI schemes are unsupported: {uri:?}"));
    }
    let decoded =
        urlencoding::decode(uri).map_err(|error| format!("invalid URI {uri:?}: {error}"))?;
    let relative = Path::new(decoded.as_ref());
    if relative.is_absolute()
        || !relative
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
    {
        return Err(format!(
            "resource path must be relative and traversal-free: {uri:?}"
        ));
    }
    let path = base.join(relative);
    let canonical = fs::canonicalize(&path)
        .map_err(|error| format!("cannot resolve resource {}: {error}", path.display()))?;
    if !canonical.starts_with(root) {
        return Err(format!(
            "resource {} resolves outside the asset root",
            path.display()
        ));
    }
    let length = fs::metadata(&canonical)
        .map_err(|error| format!("cannot stat resource {}: {error}", canonical.display()))?
        .len();
    if length > MAX_SCENE_BYTES {
        return Err(format!(
            "resource {} exceeds the scene asset byte limit",
            canonical.display()
        ));
    }
    Ok(canonical)
}

fn image_hash(width: u32, height: u32, bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    width.hash(&mut hasher);
    height.hash(&mut hasher);
    bytes.hash(&mut hasher);
    hasher.finish()
}

fn expand_gltf_image(image: gltf::image::Data) -> Result<image::RgbaImage, String> {
    use gltf::image::Format as F;
    let (channels, bytes_per_channel) = match image.format {
        F::R8 => (1, 1),
        F::R8G8 => (2, 1),
        F::R8G8B8 => (3, 1),
        F::R8G8B8A8 => (4, 1),
        F::R16 => (1, 2),
        F::R16G16 => (2, 2),
        F::R16G16B16 => (3, 2),
        F::R16G16B16A16 => (4, 2),
        F::R32G32B32FLOAT => (3, 4),
        F::R32G32B32A32FLOAT => (4, 4),
    };
    let pixel_count = (image.width as usize)
        .checked_mul(image.height as usize)
        .ok_or("glTF image dimensions overflow")?;
    let expected = pixel_count
        .checked_mul(channels)
        .and_then(|n| n.checked_mul(bytes_per_channel))
        .ok_or("glTF image byte count overflow")?;
    if image.pixels.len() != expected {
        return Err("decoded glTF texture has an invalid pixel payload".to_owned());
    }
    let sample = |offset: usize| -> u8 {
        match bytes_per_channel {
            1 => image.pixels[offset],
            2 => u16::from_le_bytes([image.pixels[offset], image.pixels[offset + 1]]).div_ceil(257)
                as u8,
            _ => {
                let value =
                    f32::from_le_bytes(image.pixels[offset..offset + 4].try_into().unwrap());
                (value.clamp(0.0, 1.0) * 255.0).round() as u8
            }
        }
    };
    let mut rgba = Vec::with_capacity(pixel_count * 4);
    for pixel in 0..pixel_count {
        let base = pixel * channels * bytes_per_channel;
        let channel = |index: usize| sample(base + index * bytes_per_channel);
        let rgb = match channels {
            1 => [channel(0), channel(0), channel(0)],
            2 => [channel(0), channel(0), channel(0)],
            _ => [channel(0), channel(1), channel(2)],
        };
        let alpha = match channels {
            2 => channel(1),
            4 => channel(3),
            _ => 255,
        };
        rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], alpha]);
    }
    image::RgbaImage::from_raw(image.width, image.height, rgba)
        .ok_or_else(|| "decoded glTF texture dimensions are invalid".to_owned())
}

fn load_gltf(path: &Path, root: &Path) -> Result<SceneAsset, String> {
    let parsed = gltf::Gltf::open(path).map_err(|error| format!("invalid glTF/GLB: {error}"))?;
    let required = parsed.document.extensions_required();
    if let Some(extension) = required.into_iter().next() {
        return Err(format!(
            "required glTF extension {extension:?} is unsupported"
        ));
    }
    let mut dependencies = Vec::new();
    for buffer in parsed.document.buffers() {
        if let gltf::buffer::Source::Uri(uri) = buffer.source()
            && !uri.starts_with("data:")
        {
            dependencies.push(checked_resource(root, path.parent().unwrap_or(root), uri)?);
        }
    }
    for image in parsed.document.images() {
        if let gltf::image::Source::Uri { uri, .. } = image.source()
            && !uri.starts_with("data:")
        {
            dependencies.push(checked_resource(root, path.parent().unwrap_or(root), uri)?);
        }
    }
    let document = &parsed.document;
    let buffers = gltf::import_buffers(document, path.parent(), parsed.blob.clone())
        .map_err(|error| format!("cannot decode glTF buffers: {error}"))?;
    for view in document.views() {
        let buffer = buffers
            .get(view.buffer().index())
            .ok_or_else(|| format!("buffer view {} references missing buffer", view.index()))?;
        let end = view
            .offset()
            .checked_add(view.length())
            .ok_or_else(|| format!("buffer view {} range overflows", view.index()))?;
        if end > buffer.0.len() {
            return Err(format!("buffer view {} exceeds its buffer", view.index()));
        }
    }
    let images = gltf::import_images(document, path.parent(), &buffers)
        .map_err(|error| format!("cannot decode glTF images: {error}"))?;
    if document.nodes().len() > MAX_NODES {
        return Err(format!("scene exceeds {MAX_NODES} nodes"));
    }
    let mut decoded_bytes = 0usize;
    let images = images
        .into_iter()
        .map(|image| {
            let rgba = expand_gltf_image(image)?;
            decoded_bytes = decoded_bytes
                .checked_add(rgba.as_raw().len())
                .ok_or("decoded texture byte count overflow")?;
            if decoded_bytes > MAX_DECODED_TEXTURE_BYTES {
                return Err("decoded glTF textures exceed 256 MiB".to_owned());
            }
            let width = rgba.width();
            let height = rgba.height();
            let rgba = rgba.into_raw();
            let hash = image_hash(width, height, &rgba);
            Ok(Arc::new(ImageData {
                width,
                height,
                rgba,
                hash,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;

    let materials = document
        .materials()
        .map(|material| {
            let pbr = material.pbr_metallic_roughness();
            Ok(SceneMaterial {
                name: material
                    .name()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("material:{}", material.index().unwrap_or(0))),
                base_color: pbr.base_color_factor().map(f64::from),
                metallic: f64::from(pbr.metallic_factor()),
                roughness: f64::from(pbr.roughness_factor()),
                clearcoat: 0.0,
                emissive: material.emissive_factor().map(f64::from),
                emissive_strength: f64::from(material.emissive_strength().unwrap_or(1.0)),
                normal_scale: material
                    .normal_texture()
                    .map(|info| f64::from(info.scale()))
                    .unwrap_or(1.0),
                occlusion_strength: material
                    .occlusion_texture()
                    .map(|info| f64::from(info.strength()))
                    .unwrap_or(1.0),
                alpha_mode: match material.alpha_mode() {
                    gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                    gltf::material::AlphaMode::Mask => AlphaMode::Mask,
                    gltf::material::AlphaMode::Blend => AlphaMode::Blend,
                },
                alpha_cutoff: f64::from(material.alpha_cutoff().unwrap_or(0.5)),
                double_sided: material.double_sided(),
                base_color_texture: pbr
                    .base_color_texture()
                    .map(|info| texture_binding(info.texture(), info.tex_coord(), &images)),
                normal_texture: material
                    .normal_texture()
                    .map(|info| texture_binding(info.texture(), info.tex_coord(), &images)),
                metallic_roughness_texture: pbr
                    .metallic_roughness_texture()
                    .map(|info| texture_binding(info.texture(), info.tex_coord(), &images)),
                occlusion_texture: material
                    .occlusion_texture()
                    .map(|info| texture_binding(info.texture(), info.tex_coord(), &images)),
                emissive_texture: material
                    .emissive_texture()
                    .map(|info| texture_binding(info.texture(), info.tex_coord(), &images)),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut warnings = Vec::new();
    for extension in document
        .extensions_used()
        .filter(|extension| *extension != "KHR_materials_emissive_strength")
    {
        warnings.push(format!(
            "glTF extension {extension:?} is optional and is not interpreted by this loader"
        ));
    }
    if document.cameras().next().is_some() {
        warnings.push("glTF cameras are loaded as source data only; native camera records are not mounted yet".to_owned());
    }
    if document
        .extensions_used()
        .any(|extension| extension == "KHR_lights_punctual")
    {
        warnings.push("glTF punctual lights are loaded as source data only; native light records are not mounted yet".to_owned());
    }
    for material in &materials {
        if material.double_sided {
            warnings.push(format!(
                "material {:?} is double-sided; the renderer still uses its configured face-culling mode",
                material.name
            ));
        }
    }
    let meshes = document
        .meshes()
        .map(|mesh| {
            let primitives = mesh
                .primitives()
                .map(|primitive| {
                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()].0));
                    let positions = reader
                        .read_positions()
                        .ok_or("glTF primitive has no POSITION accessor")?
                        .map(|value| value.map(f64::from))
                        .collect::<Vec<_>>();
                    if positions.len() > MAX_VERTICES {
                        return Err("glTF primitive exceeds the vertex limit".to_owned());
                    }
                    let normals = reader
                        .read_normals()
                        .map(|iter| iter.map(|v| v.map(f64::from)).collect())
                        .unwrap_or_default();
                    let tangents = reader
                        .read_tangents()
                        .map(|iter| iter.map(|v| v.map(f64::from)).collect())
                        .unwrap_or_default();
                    let colors = reader
                        .read_colors(0)
                        .map(|iter| {
                            iter.into_rgba_f32()
                                .map(|value| value.map(f64::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    let texcoord_sets = primitive
                        .attributes()
                        .filter_map(|(semantic, _)| match semantic {
                            gltf::mesh::Semantic::TexCoords(set) => Some(set),
                            _ => None,
                        })
                        .collect::<std::collections::BTreeSet<_>>();
                    let mut texcoords = HashMap::new();
                    for set in texcoord_sets {
                        let coordinates = reader
                            .read_tex_coords(set)
                            .ok_or("glTF TEXCOORD accessor disappeared during decode")?
                            .into_f32()
                            .map(|value| value.map(f64::from))
                            .collect();
                        texcoords.insert(set, coordinates);
                    }
                    let joints = reader
                        .read_joints(0)
                        .map(|iter| iter.into_u16().collect())
                        .unwrap_or_default();
                    let weights = reader
                        .read_weights(0)
                        .map(|iter| iter.into_f32().map(|v| v.map(f64::from)).collect())
                        .unwrap_or_default();
                    let morph_targets = reader.read_morph_targets().enumerate().map(|(target_index, (positions, normals, tangents))| {
                        if tangents.is_some() {
                            warnings.push(format!("mesh {} primitive {} morph target {target_index} TANGENT deltas are not rendered", mesh.index(), primitive.index()));
                        }
                        SceneMorphTarget {
                            positions: positions.map(|values| values.map(|value| value.map(f64::from)).collect()).unwrap_or_default(),
                            normals: normals.map(|values| values.map(|value| value.map(f64::from)).collect()).unwrap_or_default(),
                        }
                    }).collect();
                    let raw_indices = reader
                        .read_indices()
                        .map(|iter| iter.into_u32().collect())
                        .unwrap_or_else(|| (0..positions.len() as u32).collect());
                    let (mode, indices) = triangulate_gltf(primitive.mode(), raw_indices)?;
                    if indices.len() > MAX_INDICES {
                        return Err("glTF primitive exceeds the index limit".to_owned());
                    }
                    Ok(ScenePrimitive {
                        positions,
                        normals,
                        tangents,
                        colors,
                        texcoords,
                        indices,
                        joints,
                        weights,
                        morph_targets,
                        material: primitive.material().index(),
                        mode,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(SceneMesh {
                name: mesh
                    .name()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("mesh:{}", mesh.index())),
                morph_weights: mesh.weights().map(|weights| weights.iter().copied().map(f64::from).collect()).unwrap_or_else(|| vec![0.0; primitives.first().map_or(0, |primitive| primitive.morph_targets.len())]),
                primitives,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    for mesh in &meshes {
        for primitive in &mesh.primitives {
            let Some(material) = primitive.material.and_then(|index| materials.get(index)) else {
                continue;
            };
            for (slot, binding) in [
                ("base-color", material.base_color_texture.as_ref()),
                ("normal", material.normal_texture.as_ref()),
                (
                    "metallic-roughness",
                    material.metallic_roughness_texture.as_ref(),
                ),
                ("occlusion", material.occlusion_texture.as_ref()),
                ("emissive", material.emissive_texture.as_ref()),
            ] {
                if let Some(binding) = binding
                    && !primitive.texcoords.contains_key(&binding.tex_coord)
                {
                    warnings.push(format!(
                        "material {:?} {slot} binding selects missing TEXCOORD_{} on mesh {:?}; default UVs will be used",
                        material.name, binding.tex_coord, mesh.name
                    ));
                }
            }
        }
    }

    let mut nodes = (0..document.nodes().len())
        .map(|index| {
            let node = document.nodes().nth(index).expect("node index from range");
            let (translation, rotation, scale) = node.transform().decomposed();
            SceneNode {
                index,
                name: node
                    .name()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("node:{index}")),
                parent: None,
                children: node.children().map(|child| child.index()).collect(),
                mesh: node.mesh().map(|mesh| mesh.index()),
                skin: node.skin().map(|skin| skin.index()),
                morph_weights: node
                    .weights()
                    .map(|weights| weights.iter().copied().map(f64::from).collect()),
                translation: translation.map(f64::from),
                rotation: rotation.map(f64::from),
                scale: scale.map(f64::from),
            }
        })
        .collect::<Vec<_>>();
    for node in document.nodes() {
        for child in node.children() {
            nodes[child.index()].parent = Some(node.index());
        }
    }
    let scenes = document
        .scenes()
        .map(|scene| SceneDefinition {
            index: scene.index(),
            name: scene
                .name()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("scene:{}", scene.index())),
            roots: scene.nodes().map(|node| node.index()).collect(),
        })
        .collect();
    let default_scene = document.default_scene().map(|scene| scene.index());
    let skins = document
        .skins()
        .map(|skin| {
            let inverse_bind_matrices = skin
                .reader(|buffer| Some(&buffers[buffer.index()].0))
                .read_inverse_bind_matrices()
                .map(|iter| {
                    iter.map(|matrix| matrix.map(|column| column.map(f64::from)))
                        .collect()
                })
                .unwrap_or_else(|| vec![identity(); skin.joints().count()]);
            SceneSkin {
                name: skin
                    .name()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("skin:{}", skin.index())),
                joints: skin.joints().map(|joint| joint.index()).collect(),
                inverse_bind_matrices,
                skeleton_root: skin.skeleton().map(|node| node.index()),
            }
        })
        .collect();
    let animations = document
        .animations()
        .map(|animation| {
            let mut channels = Vec::new();
            let mut start_time = f64::INFINITY;
            let mut end_time: f64 = f64::NEG_INFINITY;
            for (channel_index, channel) in animation.channels().enumerate() {
                let sampler = channel.sampler();
                let reader = channel.reader(|buffer| Some(&buffers[buffer.index()].0));
                let times = reader
                    .read_inputs()
                    .ok_or("glTF animation has no input accessor")?
                    .map(f64::from)
                    .collect::<Vec<_>>();
                if times.is_empty()
                    || times.iter().any(|time| !time.is_finite())
                    || times.windows(2).any(|pair| pair[0] > pair[1])
                {
                    return Err("glTF animation key times must be finite and ordered".to_owned());
                }
                start_time = start_time.min(times[0]);
                end_time = end_time.max(*times.last().unwrap());
                let interpolation = match sampler.interpolation() {
                    gltf::animation::Interpolation::Step => Interpolation::Step,
                    gltf::animation::Interpolation::Linear => Interpolation::Linear,
                    gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
                };
                let (property, values) = match reader
                    .read_outputs()
                    .ok_or("glTF animation has no output accessor")?
                {
                    gltf::animation::util::ReadOutputs::Translations(iter) => (
                        AnimatedProperty::Translation,
                        iter.map(|v| v.map(f64::from).to_vec())
                            .collect(),
                    ),
                    gltf::animation::util::ReadOutputs::Rotations(iter) => (
                        AnimatedProperty::Rotation,
                        iter.into_f32().map(|v| v.map(f64::from).to_vec()).collect(),
                    ),
                    gltf::animation::util::ReadOutputs::Scales(iter) => (
                        AnimatedProperty::Scale,
                        iter.map(|v| v.map(f64::from).to_vec())
                            .collect(),
                    ),
                    gltf::animation::util::ReadOutputs::MorphTargetWeights(iter) => {
                        let node = channel.target().node().index();
                        let target_count = nodes[node].mesh.map(|mesh| meshes[mesh].morph_weights.len()).unwrap_or(0);
                        if target_count == 0 {
                            return Err(format!("glTF animation {} channel {channel_index} weights target node {node} has no morph targets", animation.index()));
                        }
                        let weights = iter.into_f32().map(f64::from).collect::<Vec<_>>();
                        if weights.len() % target_count != 0 {
                            return Err(format!("glTF animation {} weights channel node {node} output count {} is not divisible by {target_count} morph targets", animation.index(), weights.len()));
                        }
                        (AnimatedProperty::Weights, weights.chunks_exact(target_count).map(|sample| sample.to_vec()).collect())
                    },
                };
                channels.push(SceneAnimationChannel {
                    node: channel.target().node().index(),
                    property,
                    interpolation,
                    times,
                    values,
                });
            }
            let (start_time, duration) = if start_time.is_finite() && end_time.is_finite() {
                (start_time, (end_time - start_time).max(0.0))
            } else {
                (0.0, 0.0)
            };
            Ok(SceneAnimation {
                name: animation
                    .name()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("animation:{}", animation.index())),
                start_time,
                duration,
                channels,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(SceneAsset {
        source: "gltf",
        revision: [0; 32],
        scenes,
        default_scene,
        nodes,
        meshes,
        materials,
        animations,
        skins,
        warnings,
        dependencies,
    })
}

fn triangulate_gltf(
    mode: gltf::mesh::Mode,
    indices: Vec<u32>,
) -> Result<(PrimitiveMode, Vec<u32>), String> {
    match mode {
        gltf::mesh::Mode::Triangles => {
            if indices.len() % 3 != 0 {
                return Err("triangle primitive index count is not divisible by three".to_owned());
            }
            Ok((PrimitiveMode::Triangles, indices))
        }
        gltf::mesh::Mode::TriangleStrip => {
            let mut triangles = Vec::new();
            for i in 2..indices.len() {
                let tri = if i % 2 == 0 {
                    [indices[i - 2], indices[i - 1], indices[i]]
                } else {
                    [indices[i - 1], indices[i - 2], indices[i]]
                };
                if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                    triangles.extend(tri);
                }
            }
            Ok((PrimitiveMode::Triangles, triangles))
        }
        gltf::mesh::Mode::TriangleFan => {
            let mut triangles = Vec::new();
            for i in 2..indices.len() {
                triangles.extend([indices[0], indices[i - 1], indices[i]]);
            }
            Ok((PrimitiveMode::Triangles, triangles))
        }
        gltf::mesh::Mode::Lines | gltf::mesh::Mode::LineLoop | gltf::mesh::Mode::LineStrip => {
            Ok((PrimitiveMode::Lines, indices))
        }
        gltf::mesh::Mode::Points => Ok((PrimitiveMode::Points, indices)),
    }
}

fn load_obj(path: &Path, root: &Path) -> Result<SceneAsset, String> {
    let source = fs::read_to_string(path).map_err(|error| format!("cannot read OBJ: {error}"))?;
    let dependencies = std::cell::RefCell::new(Vec::new());
    let material_error = std::cell::RefCell::new(None);
    for (line_number, line) in source.lines().enumerate() {
        let mut parts = line.split_whitespace();
        let directive = parts.next().unwrap_or("");
        if matches!(directive, "csh" | "call") {
            return Err(format!(
                "OBJ line {} uses forbidden executable/include directive {directive:?}",
                line_number + 1
            ));
        }
        if matches!(
            directive,
            "curv" | "curv2" | "surf" | "parm" | "trim" | "hole" | "scrv" | "sp" | "end"
        ) {
            return Err(format!(
                "OBJ line {} uses unsupported free-form geometry directive {directive:?}",
                line_number + 1
            ));
        }
    }
    let options = tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ..Default::default()
    };
    let mut reader = std::io::BufReader::new(std::io::Cursor::new(source.as_bytes()));
    let (models, materials) = tobj::load_obj_buf(&mut reader, &options, |material_path| {
        let resolved = match checked_resource(
            root,
            path.parent().unwrap_or(root),
            &material_path.to_string_lossy(),
        ) {
            Ok(path) => path,
            Err(error) => {
                *material_error.borrow_mut() = Some(error);
                return Err(tobj::LoadError::OpenFileFailed);
            }
        };
        dependencies.borrow_mut().push(resolved.clone());
        let bytes = match fs::read(&resolved) {
            Ok(bytes) => bytes,
            Err(error) => {
                *material_error.borrow_mut() =
                    Some(format!("cannot read OBJ material library: {error}"));
                return Err(tobj::LoadError::OpenFileFailed);
            }
        };
        tobj::load_mtl_buf(&mut std::io::BufReader::new(std::io::Cursor::new(bytes)))
    })
    .map_err(|error| format!("invalid OBJ: {error}"))?;
    if let Some(error) = material_error.into_inner() {
        return Err(error);
    }
    let materials = materials.map_err(|error| format!("invalid OBJ material library: {error}"))?;
    let mut dependencies = dependencies.into_inner();
    let mtl_dirs = dependencies
        .iter()
        .filter_map(|path| path.parent().map(Path::to_path_buf))
        .collect::<Vec<_>>();
    let mut scene_materials = Vec::with_capacity(materials.len());
    for material in materials {
        let image = material
            .diffuse_texture
            .as_deref()
            .map(|uri| {
                let texture_path = mtl_dirs
                    .iter()
                    .find_map(|directory| checked_resource(root, directory, uri).ok())
                    .or_else(|| checked_resource(root, path.parent().unwrap_or(root), uri).ok())
                    .ok_or_else(|| {
                        format!(
                            "cannot resolve OBJ diffuse texture {uri:?} relative to its MTL library"
                        )
                    })?;
                dependencies.push(texture_path.clone());
                let pixels = image::open(&texture_path)
                    .map_err(|error| {
                        format!(
                            "cannot decode MTL diffuse texture {}: {error}",
                            texture_path.display()
                        )
                    })?
                    .into_rgba8();
                if pixels.as_raw().len() > MAX_DECODED_TEXTURE_BYTES {
                    return Err("OBJ texture exceeds decoded size limit".to_owned());
                }
                let width = pixels.width();
                let height = pixels.height();
                let rgba = pixels.into_raw();
                Ok(Arc::new(ImageData {
                    width,
                    height,
                    hash: image_hash(width, height, &rgba),
                    rgba,
                }))
            })
            .transpose()?;
        let diffuse = material.diffuse.unwrap_or([1.0; 3]);
        scene_materials.push(SceneMaterial {
            name: material.name,
            base_color: [
                f64::from(diffuse[0]),
                f64::from(diffuse[1]),
                f64::from(diffuse[2]),
                f64::from(material.dissolve.unwrap_or(1.0)),
            ],
            metallic: 0.0,
            roughness: (2.0 / (f64::from(material.shininess.unwrap_or(32.0)) + 2.0))
                .sqrt()
                .clamp(0.04, 1.0),
            clearcoat: 0.0,
            emissive: material.emissive.unwrap_or([0.0; 3]).map(f64::from),
            emissive_strength: 1.0,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            alpha_mode: if material.dissolve.is_some_and(|alpha| alpha < 1.0) {
                AlphaMode::Blend
            } else {
                AlphaMode::Opaque
            },
            alpha_cutoff: 0.5,
            double_sided: false,
            base_color_texture: image.map(|image| SceneTextureBinding {
                image,
                tex_coord: 0,
                sampler: SceneSampler {
                    wrap_s: 33_071,
                    wrap_t: 33_071,
                    mag_filter: Some(9729),
                    min_filter: Some(9729),
                },
            }),
            normal_texture: None,
            metallic_roughness_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
        });
    }
    let mut meshes = Vec::with_capacity(models.len());
    for model in models {
        let mesh = model.mesh;
        if mesh.positions.len() / 3 > MAX_VERTICES || mesh.indices.len() > MAX_INDICES {
            return Err(format!(
                "OBJ object {:?} exceeds geometry limits",
                model.name
            ));
        }
        let positions = mesh
            .positions
            .chunks_exact(3)
            .map(|p| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
            .collect();
        let normals = mesh
            .normals
            .chunks_exact(3)
            .map(|n| [f64::from(n[0]), f64::from(n[1]), f64::from(n[2])])
            .collect();
        let texcoords: Vec<[f64; 2]> = mesh
            .texcoords
            .chunks_exact(2)
            .map(|uv| [f64::from(uv[0]), f64::from(uv[1])])
            .collect();
        let primitive = ScenePrimitive {
            positions,
            normals,
            tangents: Vec::new(),
            colors: Vec::new(),
            texcoords: if texcoords.is_empty() {
                HashMap::new()
            } else {
                HashMap::from([(0, texcoords)])
            },
            indices: mesh.indices,
            joints: Vec::new(),
            weights: Vec::new(),
            morph_targets: Vec::new(),
            material: mesh.material_id,
            mode: PrimitiveMode::Triangles,
        };
        meshes.push(SceneMesh {
            name: model.name.clone(),
            primitives: vec![primitive],
            morph_weights: Vec::new(),
        });
    }
    let nodes = meshes
        .iter()
        .enumerate()
        .map(|(index, mesh)| SceneNode {
            index,
            name: mesh.name.clone(),
            parent: None,
            children: Vec::new(),
            mesh: Some(index),
            skin: None,
            morph_weights: None,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        })
        .collect::<Vec<_>>();
    let roots = (0..nodes.len()).collect();
    let scenes = vec![SceneDefinition {
        index: 0,
        name: "default".to_owned(),
        roots,
    }];
    let warnings = vec!["OBJ/MTL uses an approximate dielectric metallic-roughness mapping; source values are preserved only in import diagnostics".to_owned()];
    Ok(SceneAsset {
        source: "obj",
        revision: [0; 32],
        scenes,
        default_scene: Some(0),
        nodes,
        meshes,
        materials: scene_materials,
        animations: Vec::new(),
        skins: Vec::new(),
        warnings,
        dependencies,
    })
}

struct RestrictedUsdResolver {
    root: PathBuf,
    inner: openusd::ar::DefaultResolver,
}

impl RestrictedUsdResolver {
    fn canonical_permitted(&self, path: &openusd::ar::ResolvedPath) -> io::Result<String> {
        let path = (&**path)
            .to_str()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "USD path is not UTF-8"))?;
        let (physical_path, package_entry) =
            match openusd::ar::split_package_relative_path_outer(path) {
                Some((package, entry)) => (package, Some(entry)),
                None => (path.to_owned(), None),
            };
        let canonical = Path::new(&physical_path).canonicalize()?;
        if !canonical.starts_with(&self.root) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "USD asset escapes the declared root",
            ));
        }
        let canonical = canonical
            .to_str()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "USD path is not UTF-8"))?;
        if let Some(entry) = package_entry {
            if !openusd::usdz::package_contains(&self.inner, canonical, &entry) {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "USD package entry does not exist",
                ));
            }
            Ok(openusd::ar::join_package_relative_path(canonical, &entry))
        } else {
            Ok(canonical.to_owned())
        }
    }
}

impl openusd::ar::Resolver for RestrictedUsdResolver {
    fn create_identifier(
        &self,
        asset_path: &str,
        anchor: Option<&openusd::ar::ResolvedPath>,
    ) -> String {
        openusd::ar::Resolver::create_identifier(&self.inner, asset_path, anchor)
    }

    fn resolve(&self, asset_path: &str) -> Option<openusd::ar::ResolvedPath> {
        let resolved = openusd::ar::Resolver::resolve(&self.inner, asset_path)?;
        let canonical = self.canonical_permitted(&resolved).ok()?;
        Some(openusd::ar::ResolvedPath::new(canonical))
    }

    fn resolve_for_new_asset(&self, _asset_path: &str) -> Option<openusd::ar::ResolvedPath> {
        None
    }

    fn open_asset(
        &self,
        resolved_path: &openusd::ar::ResolvedPath,
    ) -> io::Result<Box<dyn openusd::ar::Asset>> {
        // Re-canonicalize immediately before opening and pass that exact path
        // to the backing resolver, so a symlink cannot redirect the open.
        let canonical = self.canonical_permitted(resolved_path)?;
        openusd::ar::Resolver::open_asset(&self.inner, &openusd::ar::ResolvedPath::new(canonical))
    }

    fn identity(&self) -> String {
        format!("mogf-usd-root:{}", self.root.display())
    }
}

fn load_usd(path: &Path, root: &Path) -> Result<SceneAsset, String> {
    use openusd::usd::{PrimPredicate, Stage};
    use openusd_schemas::geom::XformCache;
    use openusd_schemas::geom::{Imageable, ImageableExt, Purpose, PurposeVisibility};
    use openusd_schemas::shade::{Channel, MaterialBindingAPI, read_preview_surface};

    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot canonicalize USD asset root: {error}"))?;
    let stage = Stage::builder()
        .resolver(RestrictedUsdResolver {
            root: root.clone(),
            inner: openusd::ar::DefaultResolver::with_search_paths([&root]),
        })
        .schema_registry(openusd_schemas::schema_registry())
        .open(path.to_str().ok_or("USD asset path is not UTF-8")?)
        .map_err(|error| format!("cannot compose USD stage: {error}"))?;
    let composition_errors = stage.composition_errors();
    if !composition_errors.is_empty() {
        return Err(format!(
            "USD stage has composition errors: {}",
            composition_errors
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }
    let mut paths = Vec::new();
    stage
        .traverse(PrimPredicate::DEFAULT, |prim_path| {
            paths.push(prim_path.clone())
        })
        .map_err(|error| format!("cannot traverse composed USD stage: {error}"))?;
    // References and payloads are opened lazily by OpenUSD. Traverse first,
    // then capture the reached layer set so the revision covers composed data.
    let primary_path = path
        .canonicalize()
        .map_err(|error| format!("cannot canonicalize USD source: {error}"))?;
    let mut dependencies = std::collections::BTreeSet::new();
    for identifier in stage.layer_identifiers() {
        let (layer_path, _) = openusd::ar::split_package_relative_path_outer(&identifier)
            .unwrap_or_else(|| (identifier.clone(), String::new()));
        let layer_path = PathBuf::from(layer_path);
        let canonical = layer_path.canonicalize().map_err(|error| {
            format!(
                "cannot canonicalize composed USD layer {}: {error}",
                layer_path.display()
            )
        })?;
        if !canonical.starts_with(&root) {
            return Err(format!(
                "composed USD layer {} escapes the declared asset root",
                canonical.display()
            ));
        }
        if canonical != primary_path {
            dependencies.insert(canonical);
        }
    }
    let dependencies = dependencies.into_iter().collect();
    if paths.len() > MAX_NODES {
        return Err(format!("USD stage exceeds {MAX_NODES} prims"));
    }
    paths.retain(|path| path.to_string() != "/");
    let node_indices: HashMap<_, _> = paths
        .iter()
        .enumerate()
        .map(|(index, path)| (path.to_string(), index))
        .collect();
    let mut nodes = Vec::with_capacity(paths.len());
    let mut meshes = Vec::new();
    let mut warnings = vec!["USD import profile supports static UsdPreviewSurface scalar materials; textures, skeletons, and time samples are not imported".to_owned()];
    let mut materials = Vec::new();
    let mut material_indices = HashMap::new();
    let mut xforms = XformCache::default();
    for (node_index, prim_path) in paths.iter().enumerate() {
        let prim = stage
            .prim(prim_path.clone())
            .map_err(|error| format!("invalid USD prim path: {error}"))?;
        let kind = prim
            .type_name()
            .map_err(|error| error.to_string())?
            .map(|token| token.as_str().to_owned())
            .unwrap_or_default();
        if kind == "Camera" || kind.ends_with("Light") || kind.contains("Camera") {
            return Err(format!(
                "USD prim {} uses unsupported camera/light schema {kind}",
                prim_path
            ));
        }
        for attribute in prim.attributes().map_err(|error| error.to_string())? {
            if attribute
                .num_time_samples()
                .map_err(|error| error.to_string())?
                > 0
            {
                return Err(format!(
                    "USD prim {} is animated; the static USD profile cannot sample time-varying attributes",
                    prim_path
                ));
            }
        }
        if matches!(
            kind.as_str(),
            "Skeleton" | "SkelRoot" | "SkelAnimation" | "BlendShape"
        ) {
            return Err(format!(
                "USD prim {} uses unsupported skeletal or morph schema {kind}",
                prim_path
            ));
        }
        if prim.is_instance().map_err(|error| error.to_string())? {
            return Err(format!(
                "USD prim {} is an instance; prototype instancing is not supported by the static hierarchy profile",
                prim_path
            ));
        }
        if let Some(imageable) = Imageable::from_prim(prim.clone()).map_err(|error| {
            format!(
                "USD prim {} has invalid imageable schema: {error}",
                prim_path
            )
        })? {
            let purpose = imageable.compute_purpose().map_err(|error| {
                format!(
                    "USD prim {} has invalid effective purpose: {error}",
                    prim_path
                )
            })?;
            if matches!(purpose, Purpose::Proxy | Purpose::Guide) {
                return Err(format!(
                    "USD prim {} has effective purpose {purpose:?}; proxy and guide geometry are not imported",
                    prim_path
                ));
            }
            let visibility = imageable
                .compute_effective_visibility(purpose, None)
                .map_err(|error| {
                    format!(
                        "USD prim {} has invalid effective visibility: {error}",
                        prim_path
                    )
                })?;
            if visibility == PurposeVisibility::Invisible {
                return Err(format!(
                    "USD prim {} is invisible for effective purpose {purpose:?}; visibility is not represented by the static hierarchy profile",
                    prim_path,
                ));
            }
        }
        let parent_path = prim
            .parent()
            .filter(|parent| parent.path().to_string() != "/")
            .map(|parent| parent.path().to_string());
        let parent = parent_path
            .as_ref()
            .map(|path| {
                node_indices
                    .get(path)
                    .copied()
                    .ok_or_else(|| format!("USD prim {} has a missing parent {}", prim_path, path))
            })
            .transpose()?;
        let (local_transform, resets_stack) = xforms
            .local_transformation(&prim)
            .map_err(|error| error.to_string())?;
        if resets_stack && parent.is_some() {
            return Err(format!(
                "USD prim {} resets the transform stack; this hierarchy profile cannot preserve reset semantics",
                prim_path
            ));
        }
        let (translation, rotation, scale) = decompose_usd_transform(local_transform)
            .map_err(|reason| format!("USD prim {} transform {reason}", prim_path))?;
        let mut mesh = None;
        if kind == "Mesh" {
            let material_index = if let Some(binding) =
                MaterialBindingAPI::get(&stage, prim_path.clone())
                    .map_err(|error| error.to_string())?
            {
                if let Some(material) = binding
                    .compute_bound_material("")
                    .map_err(|error| error.to_string())?
                {
                    let material_path = material;
                    if let Some(index) = material_indices.get(material_path.as_str()).copied() {
                        Some(index)
                    } else {
                        let preview = read_preview_surface(&stage, &material_path)
                            .map_err(|error| {
                                format!(
                                    "USD material {} cannot read UsdPreviewSurface: {error}",
                                    material_path
                                )
                            })?
                            .ok_or_else(|| {
                                format!(
                                    "USD material {} is not a supported UsdPreviewSurface material",
                                    material_path
                                )
                            })?;
                        let color = |channel: Channel<openusd::gf::Vec3f>,
                                     default: [f64; 3],
                                     slot: &str|
                         -> Result<[f64; 3], String> {
                            match channel {
                                Channel::Unset => Ok(default),
                                Channel::Value(value) => {
                                    Ok([f64::from(value.x), f64::from(value.y), f64::from(value.z)])
                                }
                                Channel::Texture(path) => Err(format!(
                                    "USD material {} uses unsupported {slot} texture {path}",
                                    material_path
                                )),
                            }
                        };
                        let scalar = |channel: Channel<f32>,
                                      default: f64,
                                      slot: &str|
                         -> Result<f64, String> {
                            match channel {
                                Channel::Unset => Ok(default),
                                Channel::Value(value) => Ok(f64::from(value)),
                                Channel::Texture(path) => Err(format!(
                                    "USD material {} uses unsupported {slot} texture {path}",
                                    material_path
                                )),
                            }
                        };
                        let base = color(preview.diffuse_color, [0.18; 3], "diffuseColor")?;
                        let emissive = color(preview.emissive_color, [0.0; 3], "emissiveColor")?;
                        let alpha = scalar(preview.opacity, 1.0, "opacity")?;
                        let cutoff = scalar(preview.opacity_threshold, 0.0, "opacityThreshold")?;
                        let imported = SceneMaterial {
                            name: material_path.as_str().to_owned(),
                            base_color: [base[0], base[1], base[2], alpha],
                            metallic: scalar(preview.metallic, 0.0, "metallic")?,
                            roughness: scalar(preview.roughness, 0.5, "roughness")?,
                            clearcoat: scalar(preview.clearcoat, 0.0, "clearcoat")?,
                            emissive,
                            emissive_strength: 1.0,
                            normal_scale: 1.0,
                            occlusion_strength: 1.0,
                            alpha_mode: if cutoff > 0.0 {
                                AlphaMode::Mask
                            } else if alpha < 1.0 {
                                AlphaMode::Blend
                            } else {
                                AlphaMode::Opaque
                            },
                            alpha_cutoff: cutoff,
                            double_sided: false,
                            base_color_texture: None,
                            normal_texture: None,
                            metallic_roughness_texture: None,
                            occlusion_texture: None,
                            emissive_texture: None,
                        };
                        let index = materials.len();
                        materials.push(imported);
                        material_indices.insert(material_path.as_str().to_owned(), index);
                        Some(index)
                    }
                } else {
                    None
                }
            } else {
                None
            };
            let unsupported_primvars = prim
                .attributes()
                .map_err(|error| error.to_string())?
                .into_iter()
                .filter_map(|attribute| {
                    let name = attribute.name();
                    let relevant = name.starts_with("primvars:")
                        || matches!(name, "normals" | "displayColor" | "displayOpacity");
                    relevant.then(|| {
                        attribute
                            .has_authored_value()
                            .map(|authored| authored.then(|| name.to_owned()))
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| error.to_string())?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            if !unsupported_primvars.is_empty() {
                warnings.push(format!(
                    "USD mesh {} authors unimported primvars/display data: {}",
                    prim_path,
                    unsupported_primvars.join(", ")
                ));
            }
            let subdivision = prim
                .attribute("subdivisionScheme")
                .get::<openusd::tf::Token>()
                .map_err(|error| {
                    format!(
                        "USD mesh {} has invalid subdivision scheme: {error}",
                        prim_path
                    )
                })?;
            if subdivision
                .as_deref()
                .is_some_and(|scheme| scheme != "none")
            {
                return Err(format!(
                    "USD mesh {} uses subdivision scheme {:?}; only unsubdivided meshes are supported",
                    prim_path, subdivision
                ));
            }
            let orientation = prim
                .attribute("orientation")
                .get::<openusd::tf::Token>()
                .map_err(|error| {
                    format!("USD mesh {} has invalid orientation: {error}", prim_path)
                })?;
            if orientation.as_deref().is_some_and(|orientation| {
                orientation != "rightHanded" && orientation != "leftHanded"
            }) {
                return Err(format!(
                    "USD mesh {} has unsupported orientation {:?}",
                    prim_path, orientation
                ));
            }
            let points = prim
                .attribute("points")
                .get::<Vec<openusd::gf::Vec3f>>()
                .map_err(|error| format!("USD mesh {} has invalid points: {error}", prim_path))?
                .ok_or_else(|| format!("USD mesh {} has no points", prim_path))?;
            let counts = prim
                .attribute("faceVertexCounts")
                .get::<Vec<i32>>()
                .map_err(|error| {
                    format!("USD mesh {} has invalid face counts: {error}", prim_path)
                })?
                .ok_or_else(|| format!("USD mesh {} has no face counts", prim_path))?;
            let face_indices = prim
                .attribute("faceVertexIndices")
                .get::<Vec<i32>>()
                .map_err(|error| {
                    format!("USD mesh {} has invalid face indices: {error}", prim_path)
                })?
                .ok_or_else(|| format!("USD mesh {} has no face indices", prim_path))?;
            let hole_indices =
                prim.attribute("holeIndices")
                    .get::<Vec<i32>>()
                    .map_err(|error| {
                        format!("USD mesh {} has invalid hole indices: {error}", prim_path)
                    })?;
            if hole_indices
                .as_ref()
                .is_some_and(|indices| !indices.is_empty())
            {
                return Err(format!(
                    "USD mesh {} contains hole faces; holes are not supported by the static triangle profile",
                    prim_path
                ));
            }
            if points.len() > MAX_VERTICES || face_indices.len() > MAX_INDICES {
                return Err(format!("USD mesh {} exceeds geometry limits", prim_path));
            }
            let mut indices = Vec::new();
            let mut offset = 0usize;
            for count in counts {
                if count != 3 {
                    return Err(format!(
                        "USD mesh {} contains a non-triangle face; triangulation is required before import",
                        prim_path
                    ));
                }
                let end = offset
                    .checked_add(count as usize)
                    .ok_or("USD face index overflow")?;
                let face = face_indices.get(offset..end).ok_or_else(|| {
                    format!("USD mesh {} face counts exceed its index array", prim_path)
                })?;
                for index in face {
                    let index =
                        usize::try_from(*index).map_err(|_| "USD face index is negative")?;
                    if index >= points.len() {
                        return Err(format!(
                            "USD mesh {} has an out-of-range point index",
                            prim_path
                        ));
                    }
                    indices.push(u32::try_from(index).map_err(|_| "USD point index exceeds u32")?);
                }
                offset = end;
            }
            if offset != face_indices.len() {
                return Err(format!("USD mesh {} has trailing face indices", prim_path));
            }
            if orientation.as_deref() == Some("leftHanded") {
                for triangle in indices.chunks_exact_mut(3) {
                    triangle.swap(1, 2);
                }
            }
            let name = prim_path.to_string();
            let mesh_index = meshes.len();
            meshes.push(SceneMesh {
                name,
                morph_weights: Vec::new(),
                primitives: vec![ScenePrimitive {
                    positions: points
                        .into_iter()
                        .map(|point| [f64::from(point.x), f64::from(point.y), f64::from(point.z)])
                        .collect(),
                    normals: Vec::new(),
                    tangents: Vec::new(),
                    colors: Vec::new(),
                    texcoords: HashMap::new(),
                    indices,
                    joints: Vec::new(),
                    weights: Vec::new(),
                    morph_targets: Vec::new(),
                    material: material_index,
                    mode: PrimitiveMode::Triangles,
                }],
            });
            mesh = Some(mesh_index);
        }
        nodes.push(SceneNode {
            index: node_index,
            name: prim_path.to_string(),
            parent,
            children: Vec::new(),
            mesh,
            skin: None,
            morph_weights: None,
            translation,
            rotation,
            scale,
        });
    }
    for child in 0..nodes.len() {
        if let Some(parent) = nodes[child].parent {
            nodes[parent].children.push(child);
        }
    }
    if meshes.is_empty() {
        return Err("USD stage contains no supported triangle meshes".to_owned());
    }
    warnings.push(
        "USD units and up-axis are preserved as authored; no normalization has been applied"
            .to_owned(),
    );
    let roots = nodes
        .iter()
        .filter_map(|node| node.parent.is_none().then_some(node.index))
        .collect();
    let scenes = vec![SceneDefinition {
        index: 0,
        name: "default".to_owned(),
        roots,
    }];
    Ok(SceneAsset {
        source: "usd",
        revision: [0; 32],
        scenes,
        default_scene: Some(0),
        nodes,
        meshes,
        materials,
        animations: Vec::new(),
        skins: Vec::new(),
        warnings,
        dependencies,
    })
}

fn identity() -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[cfg(test)]
mod revision_tests {
    use super::source_revision;
    use std::{fs, path::PathBuf, time::SystemTime};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let suffix = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "mogf-scene-revision-{}-{suffix}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn revision_covers_source_and_dependencies_independent_of_order() {
        let temp = TempDir::new();
        let source = temp.0.join("scene.gltf");
        let buffer = temp.0.join("scene.bin");
        let image = temp.0.join("albedo.png");
        fs::write(&source, b"source-v1").unwrap();
        fs::write(&buffer, b"geometry-v1").unwrap();
        fs::write(&image, b"image-v1").unwrap();

        let first =
            source_revision(&source, &[buffer.clone(), image.clone()], &temp.0, "gltf").unwrap();
        let reordered =
            source_revision(&source, &[image.clone(), buffer.clone()], &temp.0, "gltf").unwrap();
        assert_eq!(first, reordered);

        let relocated = TempDir::new();
        fs::write(relocated.0.join("scene.gltf"), b"source-v1").unwrap();
        fs::write(relocated.0.join("scene.bin"), b"geometry-v1").unwrap();
        fs::write(relocated.0.join("albedo.png"), b"image-v1").unwrap();
        assert_eq!(
            first,
            source_revision(
                &relocated.0.join("scene.gltf"),
                &[
                    relocated.0.join("scene.bin"),
                    relocated.0.join("albedo.png")
                ],
                &relocated.0,
                "gltf",
            )
            .unwrap()
        );

        fs::write(&image, b"image-v2").unwrap();
        let dependency_changed =
            source_revision(&source, &[buffer.clone(), image.clone()], &temp.0, "gltf").unwrap();
        assert_ne!(first, dependency_changed);

        fs::write(&source, b"source-v2").unwrap();
        let source_changed = source_revision(&source, &[buffer, image], &temp.0, "gltf").unwrap();
        assert_ne!(dependency_changed, source_changed);
    }
}

/// Convert OpenUSD's row-vector affine matrix into the runtime's TRS fields.
/// The scene model cannot encode shear or reset-stack semantics, so those
/// inputs are rejected rather than approximated or baked into mesh vertices.
fn decompose_usd_transform(
    matrix: openusd::gf::Matrix4d,
) -> Result<([f64; 3], [f64; 4], [f64; 3]), &'static str> {
    let m = matrix.0;
    if !m.iter().all(|value| value.is_finite()) {
        return Err("contains a non-finite component");
    }
    const EPSILON: f64 = 1.0e-8;
    if m[3].abs() > EPSILON
        || m[7].abs() > EPSILON
        || m[11].abs() > EPSILON
        || (m[15] - 1.0).abs() > EPSILON
    {
        return Err("is not affine");
    }
    let mut scale = [
        (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt(),
        (m[4] * m[4] + m[5] * m[5] + m[6] * m[6]).sqrt(),
        (m[8] * m[8] + m[9] * m[9] + m[10] * m[10]).sqrt(),
    ];
    if scale.iter().any(|component| *component <= EPSILON) {
        return Err("has a singular scale");
    }
    let mut rotation = [
        [m[0] / scale[0], m[4] / scale[1], m[8] / scale[2]],
        [m[1] / scale[0], m[5] / scale[1], m[9] / scale[2]],
        [m[2] / scale[0], m[6] / scale[1], m[10] / scale[2]],
    ];
    let determinant = rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]);
    if determinant < 0.0 {
        scale[0] = -scale[0];
        for row in &mut rotation {
            row[0] = -row[0];
        }
    }
    for left in 0..3 {
        for right in 0..3 {
            let dot = (0..3)
                .map(|row| rotation[row][left] * rotation[row][right])
                .sum::<f64>();
            let expected = if left == right { 1.0 } else { 0.0 };
            if (dot - expected).abs() > 1.0e-5 {
                return Err("contains shear and cannot be represented as TRS");
            }
        }
    }
    if (determinant.abs() - 1.0).abs() > 1.0e-5 {
        return Err("contains a non-orthonormal rotation");
    }

    let trace = rotation[0][0] + rotation[1][1] + rotation[2][2];
    let quaternion = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (rotation[2][1] - rotation[1][2]) / s,
            (rotation[0][2] - rotation[2][0]) / s,
            (rotation[1][0] - rotation[0][1]) / s,
            0.25 * s,
        ]
    } else if rotation[0][0] > rotation[1][1] && rotation[0][0] > rotation[2][2] {
        let s = (1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]).sqrt() * 2.0;
        [
            0.25 * s,
            (rotation[0][1] + rotation[1][0]) / s,
            (rotation[0][2] + rotation[2][0]) / s,
            (rotation[2][1] - rotation[1][2]) / s,
        ]
    } else if rotation[1][1] > rotation[2][2] {
        let s = (1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]).sqrt() * 2.0;
        [
            (rotation[0][1] + rotation[1][0]) / s,
            0.25 * s,
            (rotation[1][2] + rotation[2][1]) / s,
            (rotation[0][2] - rotation[2][0]) / s,
        ]
    } else {
        let s = (1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]).sqrt() * 2.0;
        [
            (rotation[0][2] + rotation[2][0]) / s,
            (rotation[1][2] + rotation[2][1]) / s,
            0.25 * s,
            (rotation[1][0] - rotation[0][1]) / s,
        ]
    };
    let norm = quaternion
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm <= EPSILON || !norm.is_finite() {
        return Err("has an invalid rotation");
    }
    Ok((
        [m[12], m[13], m[14]],
        quaternion.map(|value| value / norm),
        scale,
    ))
}

pub(super) fn interpolate_channel(channel: &SceneAnimationChannel, time: f64) -> Vec<f64> {
    let value_at = |key: usize| -> &[f64] {
        if channel.interpolation == Interpolation::CubicSpline {
            &channel.values[key * 3 + 1]
        } else {
            &channel.values[key]
        }
    };
    let upper = channel.times.partition_point(|key| *key <= time);
    if upper == 0 {
        return value_at(0).to_vec();
    }
    if upper >= channel.times.len() {
        return value_at(channel.times.len() - 1).to_vec();
    }
    let left = upper - 1;
    if channel.interpolation == Interpolation::Step {
        return value_at(left).to_vec();
    }
    let t0 = channel.times[left];
    let t1 = channel.times[upper];
    let span = t1 - t0;
    let t = ((time - t0) / span).clamp(0.0, 1.0);
    if channel.interpolation == Interpolation::CubicSpline {
        let a = value_at(left);
        let out_tangent = &channel.values[left * 3 + 2];
        let in_tangent = &channel.values[upper * 3];
        let b = value_at(upper);
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        let mut value = (0..a.len())
            .map(|index| {
                h00 * a[index]
                    + h10 * span * out_tangent[index]
                    + h01 * b[index]
                    + h11 * span * in_tangent[index]
            })
            .collect::<Vec<_>>();
        if channel.property == AnimatedProperty::Rotation {
            let mut rotation: [f64; 4] = value
                .as_slice()
                .try_into()
                .expect("validated rotation cardinality");
            normalize_quaternion(&mut rotation);
            value.copy_from_slice(&rotation);
        }
        return value;
    }
    if channel.property == AnimatedProperty::Rotation {
        return slerp_quaternion(
            value_at(left)
                .try_into()
                .expect("validated rotation cardinality"),
            value_at(upper)
                .try_into()
                .expect("validated rotation cardinality"),
            t,
        )
        .to_vec();
    }
    (0..value_at(left).len())
        .map(|index| value_at(left)[index] + (value_at(upper)[index] - value_at(left)[index]) * t)
        .collect()
}

pub(super) fn slerp_quaternion(mut a: [f64; 4], mut b: [f64; 4], t: f64) -> [f64; 4] {
    normalize_quaternion(&mut a);
    normalize_quaternion(&mut b);
    let mut dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    if dot < 0.0 {
        b.iter_mut().for_each(|value| *value = -*value);
        dot = -dot;
    }
    dot = dot.clamp(-1.0, 1.0);
    let mut result = if dot > 0.9995 {
        std::array::from_fn(|index| a[index] + (b[index] - a[index]) * t)
    } else {
        let angle = dot.acos();
        let sine = angle.sin();
        let wa = ((1.0 - t) * angle).sin() / sine;
        let wb = (t * angle).sin() / sine;
        std::array::from_fn(|index| wa * a[index] + wb * b[index])
    };
    normalize_quaternion(&mut result);
    result
}

fn normalize_quaternion(value: &mut [f64; 4]) {
    let norm = value
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    if norm > 1e-15 && norm.is_finite() {
        value.iter_mut().for_each(|component| *component /= norm);
    }
}
