use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use super::{camera, require, typography};

pub(super) type JsonMap = BTreeMap<String, serde_json::Value>;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Project {
    pub(super) format: String,
    pub(super) version: u32,
    pub(super) project_id: String,
    pub(super) requirements: Vec<serde_json::Value>,
    pub(super) assets: BTreeMap<String, AssetRecord>,
    pub(super) states: JsonMap,
    pub(super) compositions: BTreeMap<String, Composition>,
    pub(super) objects: BTreeMap<String, Object>,
    pub(super) graphs: BTreeMap<String, Graph>,
    pub(super) functions: JsonMap,
    pub(super) materials: BTreeMap<String, Material>,
    pub(super) shaders: BTreeMap<String, ShaderRecord>,
    pub(super) timelines: BTreeMap<String, Timeline>,
    pub(super) clips: JsonMap,
    pub(super) media: JsonMap,
    pub(super) effects: BTreeMap<String, EffectRecord>,
    pub(super) recordings: JsonMap,
    pub(super) bakes: JsonMap,
    pub(super) metadata: JsonMap,
    #[serde(default)]
    pub(super) imports: Vec<ImportRecord>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ImportRecord {
    pub(super) uri: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TableChange {
    pub(super) table: &'static str,
    pub(super) added: usize,
    pub(super) updated: usize,
    pub(super) removed: usize,
}

impl Project {
    /// Reuse unchanged records from the live project and move in only changed records.
    pub(super) fn rehydrate(
        &mut self,
        mut next: Project,
        previous_json: &serde_json::Value,
        next_json: &serde_json::Value,
    ) -> Vec<TableChange> {
        let mut changes = Vec::new();
        macro_rules! reconcile_map {
            ($field:ident, $name:literal) => {{
                let previous_records = previous_json
                    .get($name)
                    .and_then(serde_json::Value::as_object)
                    .expect("typed project has an object table");
                let next_records = next_json
                    .get($name)
                    .and_then(serde_json::Value::as_object)
                    .expect("typed project has an object table");
                let mut change = TableChange {
                    table: $name,
                    added: 0,
                    updated: 0,
                    removed: 0,
                };
                let ids: std::collections::BTreeSet<_> =
                    previous_records.keys().chain(next_records.keys()).collect();
                for id in ids {
                    if previous_records.get(id) == next_records.get(id) {
                        continue;
                    }
                    if let Some(record) = next.$field.remove(id) {
                        if self.$field.insert(id.clone(), record).is_some() {
                            change.updated += 1;
                        } else {
                            change.added += 1;
                        }
                    } else if self.$field.remove(id).is_some() {
                        change.removed += 1;
                    }
                }
                changes.push(change);
            }};
        }
        macro_rules! replace_if_changed {
            ($field:ident, $name:literal) => {{
                if previous_json.get($name) != next_json.get($name) {
                    self.$field = next.$field;
                    changes.push(TableChange {
                        table: $name,
                        added: 0,
                        updated: 1,
                        removed: 0,
                    });
                }
            }};
        }

        replace_if_changed!(format, "format");
        replace_if_changed!(version, "version");
        replace_if_changed!(project_id, "project_id");
        replace_if_changed!(requirements, "requirements");
        replace_if_changed!(imports, "imports");
        {
            let previous_records = previous_json
                .get("assets")
                .and_then(serde_json::Value::as_object)
                .expect("typed project has an asset table");
            let next_records = next_json
                .get("assets")
                .and_then(serde_json::Value::as_object)
                .expect("typed project has an asset table");
            let mut change = TableChange {
                table: "assets",
                added: 0,
                updated: 0,
                removed: 0,
            };
            let ids: std::collections::BTreeSet<_> =
                previous_records.keys().chain(next_records.keys()).collect();
            for id in ids {
                let previous_root = self
                    .assets
                    .get(id)
                    .and_then(|asset| asset.source_root.as_ref());
                let next_root = next
                    .assets
                    .get(id)
                    .and_then(|asset| asset.source_root.as_ref());
                let previous_image = self
                    .assets
                    .get(id)
                    .and_then(|a| a.image.as_ref())
                    .map(|i| i.hash);
                let next_image = next
                    .assets
                    .get(id)
                    .and_then(|a| a.image.as_ref())
                    .map(|i| i.hash);
                let previous_scene_revision =
                    self.assets.get(id).and_then(|asset| asset.scene_revision);
                let next_scene_revision =
                    next.assets.get(id).and_then(|asset| asset.scene_revision);
                if previous_records.get(id) == next_records.get(id)
                    && previous_root == next_root
                    && previous_image == next_image
                    && previous_scene_revision == next_scene_revision
                {
                    continue;
                }
                if let Some(record) = next.assets.remove(id) {
                    if self.assets.insert(id.clone(), record).is_some() {
                        change.updated += 1;
                    } else {
                        change.added += 1;
                    }
                } else if self.assets.remove(id).is_some() {
                    change.removed += 1;
                }
            }
            changes.push(change);
        }
        reconcile_map!(states, "states");
        reconcile_map!(compositions, "compositions");
        reconcile_map!(objects, "objects");
        reconcile_map!(graphs, "graphs");
        reconcile_map!(functions, "functions");
        reconcile_map!(materials, "materials");
        reconcile_map!(shaders, "shaders");
        // Shader bytes can change without a project JSON edit. Refresh those records on reload.
        for (id, next_shader) in &next.shaders {
            if let Some(current) = self.shaders.get(id)
                && current.source != next_shader.source
                && let Some(record) = next.shaders.get(id)
            {
                self.shaders.insert(id.clone(), record.clone());
                changes.push(TableChange {
                    table: "shaders",
                    added: 0,
                    updated: 1,
                    removed: 0,
                });
            }
        }
        reconcile_map!(timelines, "timelines");
        reconcile_map!(clips, "clips");
        reconcile_map!(media, "media");
        reconcile_map!(effects, "effects");
        reconcile_map!(recordings, "recordings");
        reconcile_map!(bakes, "bakes");
        reconcile_map!(metadata, "metadata");
        changes.retain(|change| change.added + change.updated + change.removed > 0);
        changes
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Composition {
    pub(super) size: [u32; 2],
    pub(super) fps: Rational,
    pub(super) duration: Rational,
    pub(super) layers: Vec<Layer>,
    pub(super) timelines: Vec<String>,
    pub(super) clips: Vec<String>,
    pub(super) audio: Vec<AudioClip>,
    #[serde(default)]
    pub(super) audio_buses: BTreeMap<String, AudioBus>,
    pub(super) recording: Option<String>,
    pub(super) output_profile: String,
    #[serde(default)]
    pub(super) physics: Option<PhysicsWorld>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PhysicsWorld {
    #[serde(default = "default_gravity")]
    pub(super) gravity: [f32; 3],
    #[serde(default = "default_physics_rate")]
    pub(super) step_hz: u32,
    pub(super) bodies: Vec<super::physics::RigidBodySpec>,
}

fn default_gravity() -> [f32; 3] {
    [0.0, -9.81, 0.0]
}
fn default_physics_rate() -> u32 {
    120
}

/// Audio can be authored as a legacy asset ID or as a scheduled timeline clip.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum AudioClip {
    Asset(String),
    Scheduled(ScheduledAudioClip),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ScheduledAudioClip {
    pub(super) asset: String,
    pub(super) start: f64,
    #[serde(default = "default_audio_bus")]
    pub(super) channel: String,
    #[serde(default)]
    pub(super) source_start: f64,
    #[serde(default)]
    pub(super) duration: Option<f64>,
    #[serde(default = "one_f64")]
    pub(super) gain: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AudioBus {
    #[serde(default = "one_f64")]
    pub(super) gain: f64,
    #[serde(default)]
    pub(super) pan: f64,
}

impl Default for AudioBus {
    fn default() -> Self {
        Self {
            gain: 1.0,
            pan: 0.0,
        }
    }
}

fn default_audio_bus() -> String {
    "main".to_owned()
}

impl AudioClip {
    pub(super) fn asset_id(&self) -> &str {
        match self {
            Self::Asset(asset) => asset,
            Self::Scheduled(clip) => &clip.asset,
        }
    }

    pub(super) fn channel_id(&self) -> &str {
        match self {
            Self::Asset(_) => "main",
            Self::Scheduled(clip) => &clip.channel,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(
    tag = "kind",
    content = "data",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub(super) enum Layer {
    #[serde(rename = "native_3d")]
    Native3d {
        root: String,
        camera: String,
        lights: Vec<String>,
        transparent_sort: String,
        effects: Vec<String>,
        #[serde(default)]
        vertex_shader: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Object {
    pub(super) name: String,
    pub(super) local: Transform,
    pub(super) body: Body,
}

#[derive(Debug, Deserialize)]
#[serde(
    tag = "kind",
    content = "data",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub(super) enum Body {
    Group {
        children: Vec<String>,
        #[serde(default)]
        scene: Option<SceneInstance>,
    },
    Draw {
        batch: GraphOutput,
    },
    Camera {
        operation: String,
        version: u32,
        inputs: JsonMap,
        params: camera::CameraParams,
    },
    Light {
        operation: String,
        version: u32,
        inputs: JsonMap,
        params: LightParams,
    },
}

/// A mounted subscene actor. Playback is sampled from project time and node
/// controls bind MOGF tracks to unique imported source-node selectors.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SceneInstance {
    pub(super) asset: String,
    /// Source scene index; omitted only when the imported file declares a default.
    #[serde(default)]
    pub(super) scene: Option<usize>,
    #[serde(default)]
    pub(super) playback: Option<ScenePlayback>,
    /// Independent clip layers. `playback` remains the legacy single-layer form.
    #[serde(default)]
    pub(super) playbacks: Vec<ScenePlayback>,
    #[serde(default)]
    pub(super) controls: BTreeMap<String, usize>,
    #[serde(default)]
    pub(super) material_overrides: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ScenePlayback {
    pub(super) animation: String,
    #[serde(default = "one_f64")]
    pub(super) speed: f64,
    #[serde(default)]
    pub(super) offset: f64,
    #[serde(default = "default_scene_loop")]
    pub(super) loop_mode: String,
    #[serde(default = "one_f64")]
    pub(super) weight: f64,
    #[serde(default)]
    pub(super) priority: i32,
}

fn one_f64() -> f64 {
    1.0
}
fn default_scene_loop() -> String {
    "repeat".to_owned()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AssetRecord {
    pub(super) uri: String,
    pub(super) kind: String,
    #[serde(rename = "digest")]
    pub(super) _digest: Option<String>,
    #[serde(skip)]
    pub(super) source_root: Option<std::path::PathBuf>,
    #[serde(skip)]
    pub(super) image: Option<Arc<ImageData>>,
    #[serde(skip)]
    pub(super) scene: Option<Arc<crate::scene_asset::SceneAsset>>,
    #[serde(skip)]
    pub(super) scene_revision: Option<[u8; 32]>,
    #[serde(skip)]
    pub(super) scene_warnings: Vec<String>,
}

#[derive(Debug)]
pub(super) struct ImageData {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) rgba: Vec<u8>,
    pub(super) hash: u64,
}

#[cfg(test)]
mod rehydrate_tests {
    use super::*;
    use serde_json::json;

    fn project(revision: u8) -> Project {
        Project {
            format: "mogf.project".to_owned(),
            version: 1,
            project_id: "rehydrate-test".to_owned(),
            requirements: Vec::new(),
            assets: BTreeMap::from([(
                "scene".to_owned(),
                AssetRecord {
                    uri: "scene.glb".to_owned(),
                    kind: "scene".to_owned(),
                    _digest: None,
                    source_root: None,
                    image: None,
                    scene: None,
                    scene_revision: Some([revision; 32]),
                    scene_warnings: Vec::new(),
                },
            )]),
            states: BTreeMap::new(),
            compositions: BTreeMap::new(),
            objects: BTreeMap::new(),
            graphs: BTreeMap::new(),
            functions: BTreeMap::new(),
            materials: BTreeMap::new(),
            shaders: BTreeMap::new(),
            timelines: BTreeMap::new(),
            clips: BTreeMap::new(),
            media: BTreeMap::new(),
            effects: BTreeMap::new(),
            recordings: BTreeMap::new(),
            bakes: BTreeMap::new(),
            metadata: BTreeMap::new(),
            imports: Vec::new(),
        }
    }

    fn document() -> serde_json::Value {
        json!({
            "assets": {"scene": {"uri": "scene.glb", "kind": "scene", "digest": null}},
            "states": {}, "compositions": {}, "objects": {}, "graphs": {},
            "functions": {}, "materials": {}, "shaders": {}, "timelines": {},
            "clips": {}, "media": {}, "effects": {}, "recordings": {},
            "bakes": {}, "metadata": {}
        })
    }

    #[test]
    fn rehydrate_replaces_scene_asset_for_source_only_revision() {
        let mut previous = project(1);
        let next = project(2);
        let document = document();
        let changes = previous.rehydrate(next, &document, &document);

        assert_eq!(previous.assets["scene"].scene_revision, Some([2; 32]));
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].table, "assets");
        assert_eq!(
            (changes[0].added, changes[0].updated, changes[0].removed),
            (0, 1, 0)
        );
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ShaderRecord {
    pub(super) asset: String,
    pub(super) stage: String,
    #[serde(default = "default_shader_abi_version")]
    pub(super) abi_version: u32,
    #[serde(skip)]
    pub(super) source: Option<Arc<str>>,
}

fn default_shader_abi_version() -> u32 {
    1
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EffectRecord {
    pub(super) shader: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Material {
    pub(super) operation: String,
    pub(super) version: u32,
    pub(super) inputs: JsonMap,
    pub(super) params: MaterialParams,
}

impl Material {
    pub(super) fn gpu_shading(&self) -> (ShadingModel, [f64; 4], [f64; 4]) {
        if self.params.image.is_some() {
            return (ShadingModel::Unlit, [0.0; 4], self.params.image_uv_rect);
        }
        match self.operation.as_str() {
            "mogf.material.phong" => (
                ShadingModel::Phong,
                [
                    self.params.ambient,
                    self.params.diffuse,
                    self.params.specular,
                    self.params.shininess,
                ],
                self.params.shader_params,
            ),
            "mogf.material.principled" => (
                ShadingModel::Principled,
                [
                    self.params.metallic.unwrap_or(0.0),
                    self.params.roughness.unwrap_or(0.5),
                    self.params.reflectance.unwrap_or(0.5),
                    self.params.clear_coat.unwrap_or(0.0),
                ],
                self.params.shader_params,
            ),
            _ => (ShadingModel::Unlit, [0.0; 4], self.params.shader_params),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MaterialParams {
    pub(super) color: [f64; 4],
    #[serde(default = "default_ambient")]
    pub(super) ambient: f64,
    #[serde(default = "default_diffuse")]
    pub(super) diffuse: f64,
    #[serde(default = "default_specular")]
    pub(super) specular: f64,
    #[serde(default = "default_shininess")]
    pub(super) shininess: f64,
    #[serde(default)]
    pub(super) metallic: Option<f64>,
    #[serde(default)]
    pub(super) roughness: Option<f64>,
    #[serde(default)]
    pub(super) reflectance: Option<f64>,
    #[serde(default)]
    pub(super) clear_coat: Option<f64>,
    /// Four material-owned scalar values exposed to custom WGSL programs.
    #[serde(default)]
    pub(super) shader_params: [f64; 4],
    /// Optional surface WGSL program ID from the project `shaders` table.
    #[serde(default)]
    pub(super) shader: Option<String>,
    /// An authored PNG on the plane's local XY coordinates, without billboarding.
    #[serde(default)]
    pub(super) image: Option<String>,
    /// Normalized top-left image coordinates: [u_min, v_min, u_max, v_max].
    #[serde(default = "default_image_uv_rect")]
    pub(super) image_uv_rect: [f64; 4],
    /// Whether instances using this material write to the shadow maps.
    #[serde(default = "default_true")]
    pub(super) casts_shadow: bool,
    /// Whether built-in lighting and custom WGSL receive shadow visibility.
    #[serde(default = "default_true")]
    pub(super) receives_shadow: bool,
}

fn default_true() -> bool {
    true
}

fn default_image_uv_rect() -> [f64; 4] {
    [0.0, 0.0, 1.0, 1.0]
}

fn default_ambient() -> f64 {
    0.12
}

fn default_diffuse() -> f64 {
    0.9
}

fn default_specular() -> f64 {
    0.18
}

fn default_shininess() -> f64 {
    32.0
}

fn default_shadow_bias() -> f64 {
    0.0012
}

fn default_shadow_filter_radius() -> f64 {
    0.5
}

fn default_shadow_resolution() -> u32 {
    512
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LightParams {
    pub(super) color: [f64; 3],
    pub(super) intensity: f64,
    #[serde(default)]
    pub(super) range: Option<f64>,
    #[serde(default)]
    pub(super) inner_angle: Option<f64>,
    #[serde(default)]
    pub(super) outer_angle: Option<f64>,
    #[serde(default = "default_shadow_bias")]
    pub(super) shadow_bias: f64,
    #[serde(default = "default_shadow_filter_radius")]
    pub(super) shadow_filter_radius: f64,
    #[serde(default = "default_shadow_resolution")]
    pub(super) shadow_resolution: u32,
    #[serde(default)]
    pub(super) casts_shadow: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GraphOutput {
    pub(super) graph: String,
    pub(super) output: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Transform {
    pub(super) translation: [f64; 3],
    pub(super) rotation: [f64; 4],
    pub(super) scale: [f64; 3],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Graph {
    pub(super) nodes: BTreeMap<String, Node>,
    pub(super) outputs: BTreeMap<String, NodePort>,
}

#[derive(Debug, Deserialize)]
#[serde(
    tag = "kind",
    content = "data",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub(super) enum Node {
    Native {
        operation: String,
        version: u32,
        inputs: BTreeMap<String, Input>,
        params: JsonMap,
    },
}

#[derive(Debug, Deserialize)]
#[serde(
    tag = "kind",
    content = "data",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub(super) enum Input {
    Node { node: String, port: String },
    Record { table: String, record: String },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct NodePort {
    pub(super) node: String,
    pub(super) port: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Timeline {
    pub(super) tracks: Vec<Track>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Track {
    pub(super) target: Target,
    pub(super) clock: Clock,
    pub(super) at: Rational,
    pub(super) duration: Rational,
    pub(super) curve: Curve,
    pub(super) fill: String,
    pub(super) blend: String,
    pub(super) priority: i32,
    pub(super) ordinal: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Target {
    pub(super) table: String,
    pub(super) record: String,
    pub(super) slot: String,
    pub(super) instances: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(
    tag = "kind",
    content = "data",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub(super) enum Clock {
    Composition(String),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Curve {
    pub(super) operation: String,
    pub(super) version: u32,
    pub(super) inputs: JsonMap,
    pub(super) params: CurveParams,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CurveParams {
    pub(super) from: Option<f64>,
    pub(super) to: Option<f64>,
    pub(super) from_quaternion: Option<[f64; 4]>,
    pub(super) to_quaternion: Option<[f64; 4]>,
    pub(super) rotation_unit: Option<String>,
    pub(super) rotation_axis: Option<[f64; 3]>,
    pub(super) spin_turns: Option<i32>,
    pub(super) settle_bounce: Option<String>,
    pub(super) settle_bounce_radians: Option<f64>,
    pub(super) dolly_zoom_focus_z: Option<f64>,
    pub(super) dolly_zoom_start_distance: Option<f64>,
    pub(super) ease: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Rational {
    pub(super) n: String,
    pub(super) d: String,
}

#[derive(Debug, Serialize)]
pub(super) struct Frame<'a> {
    pub(super) format: &'static str,
    pub(super) version: u32,
    pub(super) project_id: &'a str,
    pub(super) composition: &'a str,
    pub(super) frame: u64,
    pub(super) time: Fraction,
    pub(super) output_profile: &'a str,
    pub(super) camera: camera::CameraFrame,
    pub(super) instances: Vec<InstanceFrame>,
    pub(super) lights: Vec<LightFrame>,
    #[serde(skip)]
    pub(super) shader_sources: BTreeMap<String, Arc<str>>,
    #[serde(skip)]
    pub(super) camera_effect_shaders: Vec<Arc<str>>,
    #[serde(skip)]
    pub(super) vertex_shader: Option<Arc<str>>,
    #[serde(skip)]
    pub(super) scene_color_shader_ids: BTreeSet<String>,
}

#[derive(Debug, Serialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub(super) enum ShadingModel {
    Unlit,
    Phong,
    Principled,
}

#[derive(Debug, Serialize, Clone)]
pub(super) struct LightFrame {
    pub(super) kind: String,
    pub(super) position: [f64; 3],
    pub(super) direction: [f64; 3],
    pub(super) color: [f64; 3],
    pub(super) intensity: f64,
    pub(super) range: Option<f64>,
    pub(super) inner_angle: Option<f64>,
    pub(super) outer_angle: Option<f64>,
    pub(super) shadow_bias: f64,
    pub(super) shadow_filter_radius: f64,
    pub(super) shadow_resolution: u32,
    pub(super) casts_shadow: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct Fraction {
    pub(super) n: String,
    pub(super) d: String,
}

#[derive(Debug, Serialize)]
pub(super) struct InstanceFrame {
    pub(super) source_object: String,
    pub(super) source_name: String,
    pub(super) instance_id: String,
    pub(super) prototype: String,
    pub(super) geometry_kind: String,
    pub(super) size: [f64; 3],
    pub(super) material: String,
    #[serde(skip_serializing)]
    pub(super) image: Option<Arc<ImageData>>,
    #[serde(skip_serializing)]
    pub(super) image_sampler: Option<crate::scene_asset::SceneSampler>,
    #[serde(skip_serializing)]
    pub(super) imported_material: Option<crate::scene_asset::SceneMaterial>,
    /// glTF alpha mode: 0 means native material behavior, 1 opaque, 2 mask,
    /// 3 blend. Kept out of serialized frame output.
    #[serde(skip_serializing)]
    pub(super) imported_alpha_mode: u8,
    pub(super) color: [f64; 4],
    pub(super) shading: ShadingModel,
    pub(super) lighting: [f64; 4], // Phong values or metallic, roughness, reflectance, clear coat
    pub(super) shader_params: [f64; 4],
    pub(super) shader_program: Option<String>,
    pub(super) casts_shadow: bool,
    pub(super) receives_shadow: bool,
    pub(super) opacity: f64,
    pub(super) selection_ids: Vec<String>,
    pub(super) character_ids: Vec<String>,
    pub(super) word_id: Option<String>,
    #[serde(skip_serializing)]
    pub(super) glyph_mesh: Option<Arc<[typography::GlyphTriangle]>>,
    #[serde(skip_serializing)]
    pub(super) glyph_hash: Option<u64>,
    #[serde(skip_serializing)]
    pub(super) scene_vertices: Option<Arc<[SceneVertex]>>,
    pub(super) world_matrix: [[f64; 4]; 4],
}

pub(super) struct InstanceSpec {
    pub(super) instance_id: String,
    pub(super) prototype: String,
    pub(super) geometry_kind: String,
    pub(super) size: [f64; 3],
    pub(super) material: String,
    pub(super) image: Option<Arc<ImageData>>,
    pub(super) color: [f64; 4],
    pub(super) shading: ShadingModel,
    pub(super) lighting: [f64; 4],
    pub(super) shader_params: [f64; 4],
    pub(super) shader_program: Option<String>,
    pub(super) casts_shadow: bool,
    pub(super) receives_shadow: bool,
    pub(super) placement: Transform,
    pub(super) selection_ids: Vec<String>,
    pub(super) character_ids: Vec<String>,
    pub(super) word_id: Option<String>,
    pub(super) glyph_mesh: Option<Arc<[typography::GlyphTriangle]>>,
    pub(super) glyph_hash: Option<u64>,
    pub(super) scene_vertices: Option<Arc<[SceneVertex]>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct SceneVertex {
    pub(super) position: [f64; 3],
    pub(super) normal: [f64; 3],
    pub(super) color: [f64; 4],
    /// UVs selected independently by base, normal, MR, occlusion, emissive slots.
    pub(super) texture_uvs: [[f64; 2]; 5],
}

pub(super) enum InstanceTrackValue {
    Scalar(f64),
    Quaternion([f64; 4]),
}

#[derive(Clone, Copy)]
pub(super) struct FractionValue {
    pub(super) n: i64,
    pub(super) d: u64,
}

impl Rational {
    pub(super) fn as_positive(&self) -> Result<FractionValue, String> {
        let n: i64 = self
            .n
            .parse()
            .map_err(|_| "rational numerator is not a canonical i64")?;
        let d: u64 = self
            .d
            .parse()
            .map_err(|_| "rational denominator is not a canonical u64")?;
        require(
            n.to_string() == self.n && d.to_string() == self.d,
            "rational integer strings must be canonical",
        )?;
        require(
            n > 0 && d > 0 && gcd(n.unsigned_abs(), d) == 1,
            "rational must be positive and reduced",
        )?;
        Ok(FractionValue { n, d })
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
pub(super) fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    while b != 0 {
        let r = a % b;
        a = b.abs();
        b = r;
    }
    a.max(1)
}

impl Rational {
    pub(super) fn as_positive_or_zero(&self) -> Result<FractionValue, String> {
        let n: i64 = self
            .n
            .parse()
            .map_err(|_| "rational numerator is not a canonical i64")?;
        let d: u64 = self
            .d
            .parse()
            .map_err(|_| "rational denominator is not a canonical u64")?;
        require(
            n.to_string() == self.n && d.to_string() == self.d,
            "rational integer strings must be canonical",
        )?;
        require(
            n >= 0 && d > 0 && gcd(n.unsigned_abs(), d) == 1,
            "rational must be nonnegative and reduced",
        )?;
        Ok(FractionValue { n, d })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TextParams {
    pub(super) text: String,
    pub(super) size: f64,
    pub(super) token_ids: Vec<String>,
    pub(super) material: String,
    pub(super) tolerance: f64,
    pub(super) quantization: f64,
    pub(super) fill_rule: String,
    pub(super) extrusion_depth: f64,
    #[serde(default)]
    pub(super) max_width: Option<f64>,
    #[serde(default = "default_font_weight")]
    pub(super) font_weight: f64,
}

fn default_font_weight() -> f64 {
    400.0
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BatchRow {
    pub(super) id: String,
    pub(super) placement: Transform,
    pub(super) material: String,
}
