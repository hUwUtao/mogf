use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::SystemTime,
};

use serde_json::{Map, Value};

use crate::model::Project;

const TABLES: &[&str] = &[
    "assets",
    "states",
    "compositions",
    "objects",
    "graphs",
    "functions",
    "materials",
    "shaders",
    "timelines",
    "clips",
    "media",
    "effects",
    "recordings",
    "bakes",
    "metadata",
];

pub(crate) struct LoadedProject {
    pub(crate) project: Project,
    /// The normalized merged document used for record-level live rehydration.
    pub(crate) json: Value,
    /// Source JSON and existing asset stamps watched without repeatedly loading their contents.
    pub(crate) source_files: BTreeMap<PathBuf, Option<SourceStamp>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SourceStamp {
    len: u64,
    modified: Option<SystemTime>,
}

impl SourceStamp {
    pub(crate) fn read(path: &Path) -> Option<Self> {
        let metadata = fs::metadata(path).ok()?;
        Some(Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }
}

const MAX_CACHED_ASSET_BYTES: usize = 128 * 1024 * 1024;
const MAX_PROJECT_SURFACE_SHADERS: usize = 16;
const MAX_PROJECT_SHADER_BYTES: usize = 1024 * 1024;

struct CachedAssetBytes {
    stamp: SourceStamp,
    bytes: Arc<[u8]>,
    last_used: u64,
}

#[derive(Default)]
struct AssetByteCache {
    files: HashMap<PathBuf, CachedAssetBytes>,
    bytes: usize,
    serial: u64,
}

pub(crate) fn read_cached_asset(path: &Path) -> std::io::Result<Arc<[u8]>> {
    let Some(stamp) = SourceStamp::read(path) else {
        return fs::read(path).map(Arc::from);
    };
    static CACHE: OnceLock<Mutex<AssetByteCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(AssetByteCache::default()));
    if let Ok(mut cache) = cache.lock() {
        cache.serial = cache.serial.wrapping_add(1);
        let serial = cache.serial;
        if let Some(entry) = cache.files.get_mut(path) {
            if entry.stamp == stamp {
                entry.last_used = serial;
                return Ok(Arc::clone(&entry.bytes));
            }
        }
        if let Some(previous) = cache.files.remove(path) {
            cache.bytes = cache.bytes.saturating_sub(previous.bytes.len());
        }
    }

    let bytes: Arc<[u8]> = Arc::from(fs::read(path)?);
    if bytes.len() <= MAX_CACHED_ASSET_BYTES
        && let Ok(mut cache) = cache.lock()
    {
        cache.serial = cache.serial.wrapping_add(1);
        let serial = cache.serial;
        while cache.bytes.saturating_add(bytes.len()) > MAX_CACHED_ASSET_BYTES {
            let Some(evict) = cache
                .files
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(path, _)| path.clone())
            else {
                break;
            };
            if let Some(removed) = cache.files.remove(&evict) {
                cache.bytes = cache.bytes.saturating_sub(removed.bytes.len());
            }
        }
        cache.bytes += bytes.len();
        cache.files.insert(
            path.to_path_buf(),
            CachedAssetBytes {
                stamp,
                bytes: Arc::clone(&bytes),
                last_used: serial,
            },
        );
    }
    Ok(bytes)
}

pub(crate) fn watched_sources_changed(sources: &BTreeMap<PathBuf, Option<SourceStamp>>) -> bool {
    sources.is_empty()
        || sources
            .iter()
            .any(|(path, previous)| SourceStamp::read(path) != *previous)
}

#[derive(Debug)]
pub(crate) struct LoadError {
    pub(crate) code: &'static str,
    pub(crate) message: String,
    pub(crate) line: Option<usize>,
    pub(crate) column: Option<usize>,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

fn load_error(code: &'static str, message: impl Into<String>) -> LoadError {
    LoadError {
        code,
        message: message.into(),
        line: None,
        column: None,
    }
}

struct Document {
    path: PathBuf,
    value: Value,
    assets: Vec<String>,
}

struct Resolver {
    stack: Vec<PathBuf>,
    visited: BTreeSet<PathBuf>,
    documents: Vec<Document>,
}

pub(crate) fn load_project(path: &Path) -> Result<LoadedProject, LoadError> {
    let root = fs::canonicalize(path).map_err(|error| {
        load_error(
            "E_PROJECT_IO",
            format!("cannot resolve project {}: {error}", path.display()),
        )
    })?;
    let mut resolver = Resolver {
        stack: Vec::new(),
        visited: BTreeSet::new(),
        documents: Vec::new(),
    };
    resolver.collect(&root)?;
    let root_document = resolver
        .documents
        .last()
        .ok_or_else(|| load_error("E_PROJECT_EMPTY", "project resolver produced no documents"))?;
    let root_project_id = root_document
        .value
        .get("project_id")
        .cloned()
        .ok_or_else(|| load_error("E_PROJECT_SHAPE", "project is missing project_id"))?;
    let root_format = root_document.value.get("format");
    let root_version = root_document.value.get("version");
    let mut merged = root_document.value.clone();
    for table in TABLES {
        merged[*table] = Value::Object(Map::new());
    }
    merged["requirements"] = Value::Array(Vec::new());
    merged["imports"] = Value::Array(Vec::new());

    let mut asset_roots = BTreeMap::new();
    let mut source_files = BTreeMap::new();
    for document in &resolver.documents {
        if document.value.get("format") != root_format
            || document.value.get("version") != root_version
        {
            return Err(load_error(
                "E_IMPORT_FORMAT",
                format!(
                    "import {} has a different project format or version",
                    document.path.display()
                ),
            ));
        }
        source_files.insert(document.path.clone(), SourceStamp::read(&document.path));
        let source_root = document
            .path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        for id in &document.assets {
            asset_roots.insert(id.clone(), source_root.clone());
        }
        merge_document(&mut merged, document)?;
    }
    merged["project_id"] = root_project_id;

    let mut project: Project = serde_json::from_value(merged.clone()).map_err(|error| {
        load_error(
            "E_IMPORT_MERGE",
            format!("merged project has invalid structure: {error}"),
        )
    })?;
    project.imports.clear();
    for (id, asset) in &mut project.assets {
        asset.source_root = asset_roots.get(id).cloned();
        let asset_path = asset
            .source_root
            .as_deref()
            .unwrap_or_else(|| root.parent().unwrap_or_else(|| Path::new(".")))
            .join(&asset.uri);
        if let (Ok(canonical), Ok(_bytes)) = (fs::canonicalize(&asset_path), fs::read(&asset_path))
        {
            source_files.insert(canonical.clone(), SourceStamp::read(&canonical));
        }
    }
    for (id, shader) in &mut project.shaders {
        if !matches!(
            shader.stage.as_str(),
            "surface" | "surface_scene" | "camera_effect" | "vertex"
        ) {
            return Err(load_error(
                "E_SHADER_STAGE",
                format!(
                    "shader {id:?} has unsupported stage {:?}; expected \"surface\", \"surface_scene\", \"camera_effect\", or \"vertex\"",
                    shader.stage
                ),
            ));
        }
        if shader.abi_version != 1 {
            return Err(load_error(
                "E_SHADER_ABI",
                format!(
                    "shader {id:?} declares unsupported ABI version {}; this renderer supports version 1",
                    shader.abi_version
                ),
            ));
        }
        let asset = project.assets.get(&shader.asset).ok_or_else(|| {
            load_error(
                "E_SHADER_ASSET",
                format!("shader {id:?} references missing asset {:?}", shader.asset),
            )
        })?;
        if asset.kind != "shader" {
            return Err(load_error(
                "E_SHADER_ASSET",
                format!(
                    "shader {id:?} asset {:?} must have kind \"shader\"",
                    shader.asset
                ),
            ));
        }
        let relative = Path::new(&asset.uri);
        if relative.is_absolute()
            || !relative
                .components()
                .all(|part| matches!(part, Component::Normal(_)))
        {
            return Err(load_error(
                "E_SHADER_URI",
                format!(
                    "shader asset {:?} must use a relative path without traversal",
                    shader.asset
                ),
            ));
        }
        let asset_root = asset
            .source_root
            .as_deref()
            .unwrap_or_else(|| root.parent().unwrap_or_else(|| Path::new(".")));
        let asset_path = asset_root.join(relative);
        let bytes = read_cached_asset(&asset_path).map_err(|error| {
            load_error(
                "E_SHADER_IO",
                format!(
                    "cannot read WGSL for shader {id:?} from {}: {error}",
                    asset_path.display()
                ),
            )
        })?;
        if bytes.len() > MAX_PROJECT_SHADER_BYTES {
            return Err(load_error(
                "E_SHADER_LIMIT",
                format!("WGSL for shader {id:?} exceeds the 1 MiB source limit"),
            ));
        }
        let source = String::from_utf8(bytes.to_vec()).map_err(|error| {
            load_error(
                "E_SHADER_UTF8",
                format!("WGSL for shader {id:?} is not UTF-8: {error}"),
            )
        })?;
        let validation = match shader.stage.as_str() {
            "surface" => crate::editor_gpu::validate_surface_shader(&source),
            "surface_scene" => crate::editor_gpu::validate_surface_scene_shader(&source),
            "camera_effect" => crate::editor_gpu::validate_camera_effect_shader(&source),
            "vertex" => crate::editor_gpu::validate_vertex_shader(&source),
            _ => unreachable!("shader stage validated above"),
        };
        validation.map_err(|error| {
            load_error(
                "E_SHADER_WGSL",
                format!("invalid WGSL for shader {id:?}: {error}"),
            )
        })?;
        shader.source = Some(std::sync::Arc::from(source));
    }
    // Decode authored image assets at preparation time, rather than every scene frame.
    let mut decoded_bytes = 0usize;
    for (id, asset) in &mut project.assets {
        if asset.kind != "image" {
            continue;
        }
        let path = crate::resolve_asset(root.parent().unwrap_or(Path::new(".")), asset)
            .map_err(|error| load_error("E_IMAGE_URI", error))?;
        let bytes =
            read_cached_asset(&path).map_err(|e| load_error("E_IMAGE_IO", e.to_string()))?;
        let mut reader =
            image::ImageReader::with_format(std::io::Cursor::new(&bytes), image::ImageFormat::Png);
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(4096);
        limits.max_image_height = Some(4096);
        limits.max_alloc = Some(64 * 1024 * 1024);
        reader.limits(limits);
        let pixels = reader
            .decode()
            .map_err(|e| load_error("E_IMAGE_DECODE", format!("image {id:?}: {e}")))?
            .into_rgba8();
        decoded_bytes += pixels.len();
        if decoded_bytes > 128 * 1024 * 1024 {
            return Err(load_error(
                "E_IMAGE_LIMIT",
                "project decoded images exceed 128 MiB",
            ));
        }
        use std::hash::{Hash, Hasher};
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        pixels.dimensions().hash(&mut hash);
        pixels.as_raw().hash(&mut hash);
        asset.image = Some(Arc::new(crate::model::ImageData {
            width: pixels.width(),
            height: pixels.height(),
            hash: hash.finish(),
            rgba: pixels.into_raw(),
        }));
    }
    for (id, asset) in &mut project.assets {
        if asset.kind != "scene" {
            continue;
        }
        let path = crate::resolve_asset(root.parent().unwrap_or(Path::new(".")), asset)
            .map_err(|error| load_error("E_SCENE_URI", error))?;
        let asset_root = asset
            .source_root
            .as_deref()
            .unwrap_or_else(|| root.parent().unwrap_or(Path::new(".")));
        let scene = crate::scene_asset::load(&path, asset_root).map_err(|error| {
            load_error("E_SCENE_IMPORT", format!("scene asset {id:?}: {error}"))
        })?;
        let canonical =
            fs::canonicalize(&path).map_err(|error| load_error("E_SCENE_IO", error.to_string()))?;
        source_files.insert(canonical.clone(), SourceStamp::read(&canonical));
        for dependency in &scene.dependencies {
            source_files.insert(dependency.clone(), SourceStamp::read(dependency));
        }
        asset.scene_warnings = scene.warnings.clone();
        asset.scene_revision = Some(scene.revision);
        asset.scene = Some(Arc::new(scene));
    }
    if project.shaders.len() > MAX_PROJECT_SURFACE_SHADERS {
        return Err(load_error(
            "E_SHADER_LIMIT",
            format!("project declares more than {MAX_PROJECT_SURFACE_SHADERS} surface shaders"),
        ));
    }
    let source_files = source_files
        .into_keys()
        .map(|path| {
            let stamp = SourceStamp::read(&path);
            (path, stamp)
        })
        .collect();
    Ok(LoadedProject {
        project,
        json: merged,
        source_files,
    })
}

impl Resolver {
    fn collect(&mut self, path: &Path) -> Result<(), LoadError> {
        let canonical = fs::canonicalize(path).map_err(|error| {
            load_error(
                "E_IMPORT_IO",
                format!("cannot resolve import {}: {error}", path.display()),
            )
        })?;
        if self.stack.contains(&canonical) {
            let mut cycle = self
                .stack
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>();
            cycle.push(canonical.display().to_string());
            return Err(load_error(
                "E_IMPORT_CYCLE",
                format!("project import cycle: {}", cycle.join(" -> ")),
            ));
        }
        if self.visited.contains(&canonical) {
            return Ok(());
        }
        let bytes = fs::read(&canonical).map_err(|error| {
            load_error(
                "E_IMPORT_IO",
                format!(
                    "cannot read project/import {}: {error}",
                    canonical.display()
                ),
            )
        })?;
        let project: Project = serde_json::from_slice(&bytes).map_err(|error| {
            let code = match error.classify() {
                serde_json::error::Category::Syntax | serde_json::error::Category::Eof => {
                    "E_JSON_PARSE"
                }
                serde_json::error::Category::Data => "E_PROJECT_SHAPE",
                serde_json::error::Category::Io => "E_PROJECT_IO",
            };
            LoadError {
                code,
                message: format!("invalid project/import {}: {error}", canonical.display()),
                line: Some(error.line()),
                column: Some(error.column()),
            }
        })?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
            let code = match error.classify() {
                serde_json::error::Category::Syntax | serde_json::error::Category::Eof => {
                    "E_JSON_PARSE"
                }
                serde_json::error::Category::Data => "E_PROJECT_SHAPE",
                serde_json::error::Category::Io => "E_PROJECT_IO",
            };
            LoadError {
                code,
                message: format!("invalid JSON in {}: {error}", canonical.display()),
                line: Some(error.line()),
                column: Some(error.column()),
            }
        })?;
        let imports = project
            .imports
            .iter()
            .map(|import| import.uri.clone())
            .collect::<Vec<_>>();
        let assets = project.assets.keys().cloned().collect::<Vec<_>>();
        self.stack.push(canonical.clone());
        for uri in &imports {
            let relative = Path::new(uri);
            if uri.trim().is_empty()
                || uri.contains("://")
                || relative.is_absolute()
                || relative
                    .components()
                    .any(|component| matches!(component, Component::RootDir | Component::Prefix(_)))
            {
                self.stack.pop();
                return Err(load_error(
                    "E_IMPORT_URI",
                    format!(
                        "import URI {uri:?} in {} must be a nonempty relative file path",
                        canonical.display()
                    ),
                ));
            }
            let child = canonical
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(relative);
            if let Err(error) = self.collect(&child) {
                self.stack.pop();
                return Err(error);
            }
        }
        self.stack.pop();
        self.visited.insert(canonical.clone());
        self.documents.push(Document {
            path: canonical,
            value,
            assets,
        });
        Ok(())
    }
}

fn merge_document(target: &mut Value, document: &Document) -> Result<(), LoadError> {
    let target_object = target
        .as_object_mut()
        .ok_or_else(|| load_error("E_IMPORT_MERGE", "merged project root is not an object"))?;
    let source_object = document.value.as_object().ok_or_else(|| {
        load_error(
            "E_IMPORT_MERGE",
            format!("project {} root is not an object", document.path.display()),
        )
    })?;
    for table in TABLES {
        let source_records = source_object
            .get(*table)
            .and_then(Value::as_object)
            .ok_or_else(|| {
                load_error(
                    "E_IMPORT_MERGE",
                    format!(
                        "{} table in {} must be an object",
                        table,
                        document.path.display()
                    ),
                )
            })?;
        let target_records = target_object
            .get_mut(*table)
            .and_then(Value::as_object_mut)
            .ok_or_else(|| {
                load_error(
                    "E_IMPORT_MERGE",
                    format!("merged {table} table is not an object"),
                )
            })?;
        for (id, record) in source_records {
            if target_records.contains_key(id) {
                return Err(load_error(
                    "E_IMPORT_DUPLICATE_ID",
                    format!(
                        "duplicate {table} ID {id:?} while importing {}",
                        document.path.display()
                    ),
                ));
            }
            target_records.insert(id.clone(), record.clone());
        }
    }
    let source_requirements = source_object
        .get("requirements")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            load_error(
                "E_IMPORT_MERGE",
                format!(
                    "requirements in {} must be an array",
                    document.path.display()
                ),
            )
        })?;
    target_object
        .get_mut("requirements")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| load_error("E_IMPORT_MERGE", "merged requirements are not an array"))?
        .extend(source_requirements.iter().cloned());
    Ok(())
}

#[cfg(test)]
mod loadover_tests {
    use super::*;
    use serde_json::json;
    use std::{fs, path::PathBuf, time::SystemTime};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let suffix = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "mogf-scene-loadover-{}-{suffix}",
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
    fn source_only_glb_revision_survives_live_project_rehydration() {
        let temp = TempDir::new();
        let fixture_root =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/glb_scene_harness");
        let active_scene = temp.0.join("scene.glb");
        fs::copy(fixture_root.join("scene_harness.glb"), &active_scene).unwrap();
        let project_path = temp.0.join("project.json");
        let project_document = json!({
            "format": "mogf.project", "version": 1, "project_id": "loadover-test",
            "requirements": [],
            "assets": {"scene": {"uri": "scene.glb", "kind": "scene", "digest": null}},
            "states": {}, "compositions": {}, "objects": {}, "graphs": {},
            "functions": {}, "materials": {}, "shaders": {}, "timelines": {},
            "clips": {}, "media": {}, "effects": {}, "recordings": {},
            "bakes": {}, "metadata": {}, "imports": []
        });
        fs::write(
            &project_path,
            serde_json::to_vec_pretty(&project_document).unwrap(),
        )
        .unwrap();

        let mut previous = load_project(&project_path).unwrap();
        let previous_revision = previous.project.assets["scene"].scene_revision.unwrap();
        let previous_scene = Arc::clone(previous.project.assets["scene"].scene.as_ref().unwrap());
        fs::copy(
            fixture_root.join("scene_harness_loadover.glb"),
            &active_scene,
        )
        .unwrap();
        let next = load_project(&project_path).unwrap();
        let next_revision = next.project.assets["scene"].scene_revision.unwrap();
        let next_scene = Arc::clone(next.project.assets["scene"].scene.as_ref().unwrap());
        assert_ne!(previous_revision, next_revision);
        assert!(
            previous_scene
                .nodes
                .iter()
                .zip(&next_scene.nodes)
                .any(|(old, new)| old.translation != new.translation)
        );

        let previous_json = previous.json.clone();
        let changes = previous
            .project
            .rehydrate(next.project, &previous_json, &next.json);
        assert_eq!(
            previous.project.assets["scene"].scene_revision,
            Some(next_revision)
        );
        assert!(Arc::ptr_eq(
            previous.project.assets["scene"].scene.as_ref().unwrap(),
            &next_scene
        ));
        assert!(changes.iter().any(|change| {
            change.table == "assets" && change.updated == 1 && change.added == 0
        }));
    }
}
