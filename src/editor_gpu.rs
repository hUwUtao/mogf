use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    sync::Arc,
};

use clap::ValueEnum;
use eframe::{
    egui,
    egui_wgpu::{Callback, CallbackResources, CallbackTrait},
    wgpu,
};
use wgpu::util::DeviceExt;

use crate::{
    camera::CameraProjection,
    gpu_abi::{CameraGpu, DrawIndirectGpu, InstanceGpu, LightGpu},
    model::Frame,
    quality::RenderProfile,
    renderer,
};

pub(super) mod offscreen;

const MAX_CACHED_GEOMETRY_BYTES: u64 = 256 * 1024 * 1024;
const MAX_GPU_LIGHTS: usize = 8;
const SHADOW_MAP_SIZE: u32 = 512;
const DEFAULT_GEOMETRY_HOOK: &str = "fn mogf_deform(position: vec3<f32>, shader_params: vec4<f32>) -> vec3<f32> { return position; }";

pub(super) fn validate_surface_shader(source: &str) -> Result<(), String> {
    validate_surface_module(source)
}

pub(super) fn validate_surface_scene_shader(source: &str) -> Result<(), String> {
    validate_surface_module(source)
}

fn validate_surface_module(source: &str) -> Result<(), String> {
    let shader = include_str!("editor_gpu.wgsl")
        .replace(
            "// MOGF_VERTEX_HOOK",
            "fn mogf_vertex(input: VertexOut) -> VertexOut { return input; }",
        )
        .replace("// MOGF_GEOMETRY_HOOK", DEFAULT_GEOMETRY_HOOK)
        .replace("// MOGF_SURFACE_HOOK", source)
        .replace(
            "// MOGF_SHADOW_OPACITY_HOOK",
            shadow_opacity_hook(Some(source)),
        );
    let module =
        naga::front::wgsl::parse_str(&shader).map_err(|error| error.emit_to_string(&shader))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| error.to_string())?;
    crate::gpu_abi::validate_surface_abi(&shader)?;
    Ok(())
}

fn shadow_opacity_hook(surface_source: Option<&str>) -> &'static str {
    let custom_hook = surface_source.is_some_and(|source| {
        let shader = include_str!("editor_gpu.wgsl")
            .replace(
                "// MOGF_VERTEX_HOOK",
                "fn mogf_vertex(input: VertexOut) -> VertexOut { return input; }",
            )
            .replace("// MOGF_GEOMETRY_HOOK", DEFAULT_GEOMETRY_HOOK)
            .replace("// MOGF_SURFACE_HOOK", source)
            .replace("// MOGF_SHADOW_OPACITY_HOOK", "");
        naga::front::wgsl::parse_str(&shader).is_ok_and(|module| {
            module
                .functions
                .iter()
                .any(|(_, function)| function.name.as_deref() == Some("mogf_shadow_opacity"))
        })
    });
    if custom_hook {
        ""
    } else if surface_source.is_some() {
        "fn mogf_shadow_opacity(input: VertexOut) -> f32 { return mogf_surface(input).a; }"
    } else {
        "fn mogf_shadow_opacity(input: VertexOut) -> f32 { return input.color.a; }"
    }
}

pub(super) fn validate_vertex_shader(source: &str) -> Result<(), String> {
    let shader = include_str!("editor_gpu.wgsl")
        .replace(
            "// MOGF_VERTEX_HOOK",
            "fn mogf_vertex(input: VertexOut) -> VertexOut { return input; }",
        )
        .replace("// MOGF_GEOMETRY_HOOK", source)
        .replace(
            "// MOGF_SURFACE_HOOK",
            "fn mogf_surface(input: VertexOut) -> vec4<f32> { return mogf_default_lit_surface(input); }",
        )
        .replace("// MOGF_SHADOW_OPACITY_HOOK", shadow_opacity_hook(None));
    let module =
        naga::front::wgsl::parse_str(&shader).map_err(|error| error.emit_to_string(&shader))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| error.to_string())?;
    crate::gpu_abi::validate_surface_abi(&shader)?;
    Ok(())
}

pub(super) fn validate_camera_effect_shader(source: &str) -> Result<(), String> {
    let shader = include_str!("editor_effect.wgsl").replace("// MOGF_EFFECT_HOOK", source);
    let module =
        naga::front::wgsl::parse_str(&shader).map_err(|error| error.emit_to_string(&shader))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| error.to_string())?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(super) enum BlendMode {
    Alpha,
    Replace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(super) enum DepthMode {
    TestWrite,
    TestOnly,
    Disabled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(super) enum CullMode {
    None,
    Front,
    Back,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RenderSettings {
    pub profile: RenderProfile,
    pub sample_count: u16,
    pub blend: BlendMode,
    pub depth: DepthMode,
    pub cull: CullMode,
    pub surface_shader: Option<String>,
    pub vertex_shader: Option<String>,
    pub effect_shader: Option<String>,
}

impl RenderSettings {
    pub(super) fn new(
        sample_count: u16,
        blend: BlendMode,
        depth: DepthMode,
        cull: CullMode,
        surface_shader: Option<String>,
        vertex_shader: Option<String>,
        effect_shader: Option<String>,
        profile: RenderProfile,
    ) -> Result<Self, String> {
        if !matches!(sample_count, 1 | 4) {
            return Err("MSAA sample count must be one of 1 or 4".into());
        }
        if let Some(source) = &surface_shader {
            let shader = include_str!("editor_gpu.wgsl")
                .replace(
                    "// MOGF_VERTEX_HOOK",
                    "fn mogf_vertex(input: VertexOut) -> VertexOut { return input; }",
                )
                .replace("// MOGF_GEOMETRY_HOOK", DEFAULT_GEOMETRY_HOOK)
                .replace("// MOGF_SURFACE_HOOK", source)
                .replace(
                    "// MOGF_SHADOW_OPACITY_HOOK",
                    shadow_opacity_hook(Some(source)),
                );
            let module = naga::front::wgsl::parse_str(&shader).map_err(|error| {
                format!("custom WGSL parse error: {}", error.emit_to_string(&shader))
            })?;
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .map_err(|error| format!("custom WGSL validation error: {error}"))?;
        }
        if let Some(source) = &vertex_shader {
            let shader = include_str!("editor_gpu.wgsl")
                .replace("// MOGF_VERTEX_HOOK", source)
                .replace("// MOGF_GEOMETRY_HOOK", DEFAULT_GEOMETRY_HOOK)
                .replace("// MOGF_SURFACE_HOOK", "fn mogf_surface(input: VertexOut) -> vec4<f32> { return mogf_default_lit_surface(input); }")
                .replace("// MOGF_SHADOW_OPACITY_HOOK", shadow_opacity_hook(None));
            let module = naga::front::wgsl::parse_str(&shader).map_err(|error| {
                format!(
                    "custom vertex WGSL parse error: {}",
                    error.emit_to_string(&shader)
                )
            })?;
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .map_err(|error| format!("custom vertex WGSL validation error: {error}"))?;
        }
        if let Some(source) = &effect_shader {
            let shader = include_str!("editor_effect.wgsl").replace("// MOGF_EFFECT_HOOK", source);
            let module = naga::front::wgsl::parse_str(&shader).map_err(|error| {
                format!(
                    "custom effect WGSL parse error: {}",
                    error.emit_to_string(&shader)
                )
            })?;
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .map_err(|error| format!("custom effect WGSL validation error: {error}"))?;
        }
        Ok(Self {
            profile,
            sample_count,
            blend,
            depth,
            cull,
            surface_shader,
            vertex_shader,
            effect_shader,
        })
    }
}

#[derive(Clone)]
enum MeshKey {
    Plane,
    Box,
    Glyph {
        mesh: Arc<[crate::typography::GlyphTriangle]>,
        geometry_hash: u64,
    },
    Imported {
        mesh: Arc<[crate::model::SceneVertex]>,
        geometry_hash: u64,
    },
}

impl PartialEq for MeshKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Plane, Self::Plane) | (Self::Box, Self::Box) => true,
            (
                Self::Glyph {
                    mesh: left,
                    geometry_hash: left_hash,
                },
                Self::Glyph {
                    mesh: right,
                    geometry_hash: right_hash,
                },
            ) => {
                left_hash == right_hash
                    && (std::ptr::eq(left.as_ptr(), right.as_ptr())
                        || (left.len() == right.len()
                            && left.iter().zip(right.iter()).all(|(left, right)| {
                                left.shade.to_bits() == right.shade.to_bits()
                                    && left
                                        .vertices
                                        .iter()
                                        .flatten()
                                        .zip(right.vertices.iter().flatten())
                                        .all(|(left, right)| left.to_bits() == right.to_bits())
                            })))
            }
            (
                Self::Imported {
                    mesh: left,
                    geometry_hash: left_hash,
                },
                Self::Imported {
                    mesh: right,
                    geometry_hash: right_hash,
                },
            ) => {
                left_hash == right_hash
                    && (std::ptr::eq(left.as_ptr(), right.as_ptr())
                        || left.as_ref() == right.as_ref())
            }
            _ => false,
        }
    }
}

impl Eq for MeshKey {}

impl Hash for MeshKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Plane => {
                0_u8.hash(state);
            }
            Self::Box => {
                1_u8.hash(state);
            }
            Self::Glyph {
                mesh,
                geometry_hash,
            } => {
                2_u8.hash(state);
                geometry_hash.hash(state);
                mesh.len().hash(state);
            }
            Self::Imported {
                mesh,
                geometry_hash,
            } => {
                3_u8.hash(state);
                geometry_hash.hash(state);
                mesh.len().hash(state);
            }
        }
    }
}

struct MeshData {
    key: MeshKey,
}

impl MeshKey {
    fn vertex_count(&self) -> usize {
        match self {
            Self::Plane => 6,
            Self::Box => 36,
            Self::Glyph { mesh, .. } => mesh.len() * 3,
            Self::Imported { mesh, .. } => mesh.len(),
        }
    }

    fn cache_bytes(&self) -> u64 {
        let key_bytes = match self {
            Self::Glyph { mesh, .. } => {
                std::mem::size_of::<Self>() as u64
                    + (mesh.len() * std::mem::size_of::<crate::typography::GlyphTriangle>()) as u64
                    + (mesh.len() * 3 * std::mem::size_of::<[f32; 21]>()) as u64
            }
            Self::Imported { mesh, .. } => {
                std::mem::size_of::<Self>() as u64
                    + (mesh.len() * std::mem::size_of::<crate::model::SceneVertex>()) as u64
                    + (mesh.len() * std::mem::size_of::<[f32; 21]>()) as u64
            }
            Self::Plane | Self::Box => std::mem::size_of::<Self>() as u64,
        };
        key_bytes + (self.vertex_count() * std::mem::size_of::<[f32; 21]>()) as u64
    }

    fn create_buffer(&self, device: &wgpu::Device, label: &'static str) -> wgpu::Buffer {
        match self {
            Self::Glyph { mesh, .. } => {
                let vertices: Vec<[f32; 21]> = mesh
                    .iter()
                    .flat_map(|triangle| {
                        triangle.vertices.map(|point| {
                            [
                                point[0] as f32,
                                point[1] as f32,
                                point[2] as f32,
                                triangle.shade as f32,
                                point[0] as f32 + 0.5,
                                0.5 - point[1] as f32,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                            ]
                        })
                    })
                    .collect();
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
            Self::Imported { mesh, .. } => {
                let vertices: Vec<[f32; 21]> = mesh
                    .iter()
                    .map(|vertex| {
                        let mut output = [0.0; 21];
                        output[..3].copy_from_slice(&vertex.position.map(|value| value as f32));
                        output[3] = 1.0;
                        output[4..6].copy_from_slice(&vertex.texture_uvs[0].map(|v| v as f32));
                        output[6..9].copy_from_slice(&vertex.normal.map(|value| value as f32));
                        for slot in 1..5 {
                            let start = 9 + (slot - 1) * 2;
                            output[start..start + 2]
                                .copy_from_slice(&vertex.texture_uvs[slot].map(|v| v as f32));
                        }
                        output[17..21].copy_from_slice(&vertex.color.map(|value| value as f32));
                        output
                    })
                    .collect();
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
            Self::Plane => {
                let triangles = renderer::plane_triangles([1.0, 1.0, 0.0]);
                let positions = triangles
                    .into_iter()
                    .flatten()
                    .map(|point| {
                        [
                            point[0] as f32,
                            point[1] as f32,
                            point[2] as f32,
                            1.0,
                            point[0] as f32 + 0.5,
                            0.5 - point[1] as f32,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ]
                    })
                    .collect::<Vec<_>>();
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(&positions),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
            Self::Box => {
                let triangles = renderer::box_triangles([1.0, 1.0, 1.0]);
                let positions = triangles
                    .into_iter()
                    .flatten()
                    .map(|point| {
                        [
                            point[0] as f32,
                            point[1] as f32,
                            point[2] as f32,
                            1.0,
                            point[0] as f32 + 0.5,
                            0.5 - point[1] as f32,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ]
                    })
                    .collect::<Vec<_>>();
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(&positions),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
        }
    }
}

#[derive(Clone)]
struct DrawInstance {
    mesh: usize,
    gpu: InstanceGpu,
    opaque: bool,
    shader_program: Option<String>,
    scene_color: bool,
    casts_shadow: bool,
    receives_shadow: bool,
    image: Option<Arc<crate::model::ImageData>>,
    image_sampler: Option<crate::scene_asset::SceneSampler>,
    imported_material: Option<crate::scene_asset::SceneMaterial>,
    imported_alpha_mode: u8,
}

struct DrawBatch {
    mesh: usize,
    first_instance: usize,
    instance_count: usize,
    compact_base: usize,
    shader_program: Option<String>,
    scene_color: bool,
    material_texture_key: MaterialTextureKey,
    imported_alpha_mode: u8,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct MaterialTextureKey {
    base: Option<(u64, Option<crate::scene_asset::SceneSampler>)>,
    normal: Option<(u64, crate::scene_asset::SceneSampler)>,
    metallic_roughness: Option<(u64, crate::scene_asset::SceneSampler)>,
    occlusion: Option<(u64, crate::scene_asset::SceneSampler)>,
    emissive: Option<(u64, crate::scene_asset::SceneSampler)>,
}

fn material_texture_key(draw: &DrawInstance) -> MaterialTextureKey {
    let key = |binding: &Option<crate::scene_asset::SceneTextureBinding>| {
        binding
            .as_ref()
            .map(|binding| (binding.image.hash, binding.sampler))
    };
    MaterialTextureKey {
        base: draw
            .image
            .as_ref()
            .map(|image| (image.hash, draw.image_sampler)),
        normal: draw
            .imported_material
            .as_ref()
            .and_then(|m| key(&m.normal_texture)),
        metallic_roughness: draw
            .imported_material
            .as_ref()
            .and_then(|m| key(&m.metallic_roughness_texture)),
        occlusion: draw
            .imported_material
            .as_ref()
            .and_then(|m| key(&m.occlusion_texture)),
        emissive: draw
            .imported_material
            .as_ref()
            .and_then(|m| key(&m.emissive_texture)),
    }
}

fn create_imported_sampler(
    device: &wgpu::Device,
    source: crate::scene_asset::SceneSampler,
) -> wgpu::Sampler {
    let address = |mode| match mode {
        33_071 => wgpu::AddressMode::ClampToEdge,
        33_648 => wgpu::AddressMode::MirrorRepeat,
        _ => wgpu::AddressMode::Repeat,
    };
    let (min_filter, mipmap_filter) = match source.min_filter {
        Some(9728 | 9984) => (wgpu::FilterMode::Nearest, wgpu::MipmapFilterMode::Nearest),
        Some(9985) => (wgpu::FilterMode::Linear, wgpu::MipmapFilterMode::Nearest),
        Some(9986) => (wgpu::FilterMode::Nearest, wgpu::MipmapFilterMode::Linear),
        Some(9987) => (wgpu::FilterMode::Linear, wgpu::MipmapFilterMode::Linear),
        _ => (wgpu::FilterMode::Linear, wgpu::MipmapFilterMode::Nearest),
    };
    let mag_filter = match source.mag_filter {
        Some(9728) => wgpu::FilterMode::Nearest,
        _ => wgpu::FilterMode::Linear,
    };
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("mogf-imported-scene-sampler"),
        address_mode_u: address(source.wrap_s),
        address_mode_v: address(source.wrap_t),
        mag_filter,
        min_filter,
        mipmap_filter,
        ..Default::default()
    })
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.max(0.0).powf(1.0 / 2.4) - 0.055
    }
}

/// Build complete mip chains in the texture's sampling color space. Normal maps
/// average as vectors and renormalize so coarse levels retain a unit direction.
fn imported_texture_mips(
    image: &crate::model::ImageData,
    srgb: bool,
    normal_map: bool,
) -> Vec<(u32, u32, Vec<u8>)> {
    let mut levels = vec![(image.width, image.height, image.rgba.clone())];
    while levels
        .last()
        .is_some_and(|(width, height, _)| *width > 1 || *height > 1)
    {
        let (width, height, source) = levels.last().expect("base image mip exists");
        let next_width = (*width / 2).max(1);
        let next_height = (*height / 2).max(1);
        let mut next = vec![0u8; next_width as usize * next_height as usize * 4];
        for y in 0..next_height {
            let y0 = y as f32 * *height as f32 / next_height as f32;
            let y1 = (y + 1) as f32 * *height as f32 / next_height as f32;
            for x in 0..next_width {
                let x0 = x as f32 * *width as f32 / next_width as f32;
                let x1 = (x + 1) as f32 * *width as f32 / next_width as f32;
                let mut sum = [0.0f32; 4];
                let mut total = 0.0f32;
                for source_y in y0.floor() as u32..y1.ceil() as u32 {
                    let weight_y =
                        (y1.min(source_y as f32 + 1.0) - y0.max(source_y as f32)).max(0.0);
                    for source_x in x0.floor() as u32..x1.ceil() as u32 {
                        let weight_x =
                            (x1.min(source_x as f32 + 1.0) - x0.max(source_x as f32)).max(0.0);
                        let weight = weight_x * weight_y;
                        let index = ((source_y * *width + source_x) * 4) as usize;
                        for channel in 0..4 {
                            let mut value = source[index + channel] as f32 / 255.0;
                            if channel < 3 && srgb {
                                value = srgb_to_linear(value);
                            }
                            if channel < 3 && normal_map {
                                value = value * 2.0 - 1.0;
                            }
                            sum[channel] += value * weight;
                        }
                        total += weight;
                    }
                }
                for value in &mut sum {
                    *value /= total.max(1.0e-8);
                }
                if normal_map {
                    let length = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
                    if length > 1.0e-8 {
                        for channel in 0..3 {
                            sum[channel] /= length;
                        }
                    } else {
                        sum[..3].copy_from_slice(&[0.0, 0.0, 1.0]);
                    }
                    for value in &mut sum[..3] {
                        *value = *value * 0.5 + 0.5;
                    }
                } else if srgb {
                    for value in &mut sum[..3] {
                        *value = linear_to_srgb(*value);
                    }
                }
                let destination = ((y * next_width + x) * 4) as usize;
                for channel in 0..4 {
                    next[destination + channel] =
                        (sum[channel].clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        levels.push((next_width, next_height, next));
    }
    levels
}

#[cfg(test)]
mod imported_texture_mip_tests {
    use super::imported_texture_mips;
    use crate::model::ImageData;

    #[test]
    fn generates_complete_srgb_and_normal_mip_chains() {
        let checker = ImageData {
            width: 2,
            height: 2,
            rgba: vec![
                0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255,
            ],
            hash: 0,
        };
        let srgb = imported_texture_mips(&checker, true, false);
        let linear = imported_texture_mips(&checker, false, false);
        assert_eq!(
            srgb.iter().map(|(w, h, _)| (*w, *h)).collect::<Vec<_>>(),
            [(2, 2), (1, 1)]
        );
        assert_eq!(&srgb[1].2[..3], &[188, 188, 188]);
        assert_eq!(&linear[1].2[..3], &[128, 128, 128]);

        let normals = ImageData {
            width: 2,
            height: 2,
            rgba: vec![
                255, 128, 128, 255, 255, 128, 128, 255, 128, 255, 128, 255, 128, 255, 128, 255,
            ],
            hash: 1,
        };
        let normal_mips = imported_texture_mips(&normals, false, true);
        assert_eq!(&normal_mips[1].2[..3], &[218, 218, 128]);
    }
}

pub(super) struct SceneFrame {
    camera: CameraGpu,
    shadow_camera: Option<CameraGpu>,
    shadow_resolution: Option<u32>,
    shadow_signature: Option<u64>,
    has_mask_shadow_casters: bool,
    camera_effect_shaders: Vec<Arc<str>>,
    requires_scene_color: bool,
    vertex_shader: Option<Arc<str>>,
    lights: Vec<LightGpu>,
    meshes: Vec<MeshData>,
    draws: Vec<DrawInstance>,
    batches: Vec<DrawBatch>,
    shader_sources: HashMap<String, Arc<str>>,
}

struct CachedMesh {
    buffer: wgpu::Buffer,
    cache_bytes: u64,
    last_used: u64,
}

struct SurfacePipelines {
    opaque: wgpu::RenderPipeline,
    transparent: wgpu::RenderPipeline,
}

struct ShadowPipelines {
    opaque: wgpu::RenderPipeline,
    alpha: wgpu::RenderPipeline,
    mask: wgpu::RenderPipeline,
}

struct ProjectSurfacePipelines {
    source: Arc<str>,
    vertex_source: Option<Arc<str>>,
    pipelines: SurfacePipelines,
    shadow_alpha: wgpu::RenderPipeline,
}

struct ProjectVertexPipelines {
    source: Arc<str>,
    pipelines: SurfacePipelines,
    shadow_opaque: wgpu::RenderPipeline,
    shadow_alpha: wgpu::RenderPipeline,
    shadow_mask: wgpu::RenderPipeline,
}

struct GpuResources {
    target_format: wgpu::TextureFormat,
    settings: RenderSettings,
    prepared_frame: Option<Arc<SceneFrame>>,
    cull_bind_group: wgpu::BindGroup,
    shadow_cull_bind_group: wgpu::BindGroup,
    cull_bind_group_layout: wgpu::BindGroupLayout,
    render_bind_group: wgpu::BindGroup,
    render_bind_group_layout: wgpu::BindGroupLayout,
    render_pipeline_layout: wgpu::PipelineLayout,
    scene_color_bind_group_layout: wgpu::BindGroupLayout,
    scene_color_sampler: wgpu::Sampler,
    default_scene_color_bind_group: wgpu::BindGroup,
    default_scene_color_view: wgpu::TextureView,
    image_bind_group_layout: wgpu::BindGroupLayout,
    image_sampler: wgpu::Sampler,
    default_image_bind_group: wgpu::BindGroup,
    image_cache: HashMap<MaterialTextureKey, wgpu::BindGroup>,
    shadow_bind_group: wgpu::BindGroup,
    shadow_bind_group_layout: wgpu::BindGroupLayout,
    shadow_sampler: wgpu::Sampler,
    shadow_filter_sampler: wgpu::Sampler,
    shadow_transmittance_sampler: wgpu::Sampler,
    shadow_alpha_bind_group: wgpu::BindGroup,
    shadow_caster_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    shadow_camera_buffer: wgpu::Buffer,
    shadow_camera_bytes: Vec<u8>,
    light_buffer: wgpu::Buffer,
    light_capacity: usize,
    cull_pipeline: wgpu::ComputePipeline,
    opaque_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    project_surface_pipelines: HashMap<String, ProjectSurfacePipelines>,
    project_vertex_pipelines: Option<ProjectVertexPipelines>,
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_alpha_pipeline: wgpu::RenderPipeline,
    shadow_mask_pipeline: wgpu::RenderPipeline,
    shadow_depth_view: wgpu::TextureView,
    shadow_transmittance_view: wgpu::TextureView,
    shadow_resolution: u32,
    shadow_signature: Option<u64>,
    effect_pipelines: HashMap<Arc<str>, wgpu::RenderPipeline>,
    effect_layout: Option<wgpu::BindGroupLayout>,
    effect_sampler: Option<wgpu::Sampler>,
    scene_target: Option<SceneTarget>,
    mesh_cache: HashMap<MeshKey, CachedMesh>,
    frame_meshes: HashMap<MeshKey, CachedMesh>,
    mesh_cache_bytes: u64,
    frame_serial: u64,
    instance_buffer: wgpu::Buffer,
    instance_capacity: u64,
    compact_buffer: wgpu::Buffer,
    compact_capacity: u64,
    shadow_compact_buffer: wgpu::Buffer,
    shadow_compact_capacity: u64,
    indirect_buffer: wgpu::Buffer,
    indirect_capacity: u64,
    shadow_indirect_buffer: wgpu::Buffer,
    shadow_indirect_capacity: u64,
    instance_data: Vec<InstanceGpu>,
    instance_bytes: Vec<u8>,
    camera_bytes: Vec<u8>,
    stats_started: std::time::Instant,
    stats_prepare_ms: f64,
    stats_prepares: u64,
    stats_mesh_hits: u64,
    stats_mesh_misses: u64,
    stats_batches: u64,
    stats_instances: u64,
    stats_shadow_renders: u64,
    stats_shadow_reuses: u64,
}

struct SceneTarget {
    size: [u32; 2],
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    scene_color_snapshot: Option<wgpu::Texture>,
    scene_color_bind_group: Option<wgpu::BindGroup>,
    multisampled_color_view: Option<wgpu::TextureView>,
    depth_view: Option<wgpu::TextureView>,
    effect_depth_view: wgpu::TextureView,
    effect_bind_groups: Vec<wgpu::BindGroup>,
    effect_outputs: Vec<EffectTarget>,
}

struct EffectTarget {
    color_view: wgpu::TextureView,
    multisampled_color_view: Option<wgpu::TextureView>,
}

pub(super) fn vulkan_wgpu_configuration() -> Result<eframe::egui_wgpu::WgpuConfiguration, String> {
    let mut config = eframe::egui_wgpu::WgpuConfiguration::default();
    match &mut config.wgpu_setup {
        eframe::egui_wgpu::WgpuSetup::CreateNew(setup) => {
            setup.instance_descriptor.backends = wgpu::Backends::VULKAN;
            setup.power_preference = wgpu::PowerPreference::HighPerformance;
        }
        eframe::egui_wgpu::WgpuSetup::Existing(_) => {
            return Err("MOGF requires its own Vulkan-backed wgpu surface".into());
        }
    }
    Ok(config)
}

pub(super) fn callback_shape(
    frame: Arc<SceneFrame>,
    rect: egui::Rect,
    target_format: wgpu::TextureFormat,
    settings: Arc<RenderSettings>,
    emit_stats: bool,
) -> egui::Shape {
    let mut effect_sources = if frame.camera_effect_shaders.is_empty() {
        settings
            .effect_shader
            .as_deref()
            .map(Arc::from)
            .into_iter()
            .collect()
    } else {
        frame.camera_effect_shaders.clone()
    };
    if frame.requires_scene_color && effect_sources.is_empty() {
        effect_sources.push(Arc::from(
            "fn mogf_effect(uv: vec2<f32>, color: vec4<f32>, resolution: vec2<f32>) -> vec4<f32> { return color; }",
        ));
    }
    let callback = GpuCallback {
        frame,
        effect_sources,
        rect,
        target_format,
        settings,
        emit_stats,
    };
    egui::Shape::Callback(Callback::new_paint_callback(rect, callback))
}

fn shadow_bounds_points(draws: &[DrawInstance]) -> Vec<[f64; 3]> {
    draws
        .iter()
        .filter(|draw| draw.casts_shadow || draw.receives_shadow)
        .flat_map(|draw| {
            (0..8).map(move |corner| {
                let local = std::array::from_fn(|axis| {
                    let min = draw.gpu.bounds_min[axis] as f64;
                    let max = draw.gpu.bounds_max[axis] as f64;
                    if corner & (1 << axis) == 0 { min } else { max }
                });
                renderer::transform_point(&draw.gpu.rows.map(|row| row.map(f64::from)), local)
            })
        })
        .collect()
}

fn directional_shadow_camera(draws: &[DrawInstance], direction: [f64; 3]) -> Option<CameraGpu> {
    let points = shadow_bounds_points(draws);
    if points.is_empty() {
        return None;
    }
    let bounds_min = std::array::from_fn::<_, 3, _>(|axis| {
        points
            .iter()
            .map(|point| point[axis])
            .fold(f64::INFINITY, f64::min)
    });
    let bounds_max = std::array::from_fn::<_, 3, _>(|axis| {
        points
            .iter()
            .map(|point| point[axis])
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let center: [f64; 3] = std::array::from_fn(|axis| (bounds_min[axis] + bounds_max[axis]) * 0.5);
    let diagonal = bounds_min
        .iter()
        .zip(bounds_max.iter())
        .map(|(min, max)| (max - min) * (max - min))
        .sum::<f64>()
        .sqrt();
    let padding = (diagonal * 0.08).max(0.05);
    let direction_length = direction
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !direction_length.is_finite() || direction_length < 1e-8 {
        return None;
    }
    let forward = direction.map(|value| value / direction_length);
    let up_hint = if forward[1].abs() > 0.95 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let normalize = |value: [f64; 3]| {
        let length = value.iter().map(|axis| axis * axis).sum::<f64>().sqrt();
        value.map(|axis| axis / length.max(1e-12))
    };
    let cross = |left: [f64; 3], right: [f64; 3]| {
        [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]
    };
    let dot =
        |left: [f64; 3], right: [f64; 3]| left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
    let right = normalize(cross(forward, up_hint));
    let up = cross(right, forward);
    let backward = forward.map(|value| -value);
    let camera_distance = diagonal + padding * 2.0;
    let position = std::array::from_fn(|axis| center[axis] - forward[axis] * camera_distance);
    let view_rows =
        [right, up, backward].map(|basis| [basis[0], basis[1], basis[2], -dot(basis, position)]);
    let depths = points
        .iter()
        .map(|point| {
            -((view_rows[2][0] * point[0])
                + (view_rows[2][1] * point[1])
                + (view_rows[2][2] * point[2])
                + view_rows[2][3])
        })
        .collect::<Vec<_>>();
    let min_depth = depths.iter().copied().fold(f64::INFINITY, f64::min);
    let max_depth = depths.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let near = (min_depth - padding).max(0.01);
    let far = (max_depth + padding).max(near + 0.1);
    let projected_x = points
        .iter()
        .map(|point| {
            dot(
                right,
                std::array::from_fn(|axis| point[axis] - center[axis]),
            )
        })
        .collect::<Vec<_>>();
    let projected_y = points
        .iter()
        .map(|point| dot(up, std::array::from_fn(|axis| point[axis] - center[axis])))
        .collect::<Vec<_>>();
    let half_width = (projected_x
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - projected_x.iter().copied().fold(f64::INFINITY, f64::min))
        * 0.5
        + padding;
    let half_height = (projected_y
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - projected_y.iter().copied().fold(f64::INFINITY, f64::min))
        * 0.5
        + padding;
    let shadow_aspect = 1.0_f64;
    let ortho_height = (half_height.max(half_width / shadow_aspect) * 2.0).max(0.1);
    Some(CameraGpu {
        view_rows: [
            view_rows[0].map(|value| value as f32),
            view_rows[1].map(|value| value as f32),
            view_rows[2].map(|value| value as f32),
            [0.0, 0.0, 0.0, 1.0],
        ],
        projection: [
            ortho_height as f32,
            shadow_aspect as f32,
            near as f32,
            far as f32,
        ],
        flags: [1, 0, draws.len() as u32, 0],
        shadow: [0, 0, 0, 1],
    })
}

fn spot_shadow_camera(
    draws: &[DrawInstance],
    light: &crate::model::LightFrame,
) -> Option<CameraGpu> {
    let points = shadow_bounds_points(draws);
    if points.is_empty() {
        return None;
    }
    let direction_length = light
        .direction
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let outer_angle = light.outer_angle?;
    if !direction_length.is_finite()
        || direction_length < 1e-8
        || !outer_angle.is_finite()
        || !(0.0..std::f64::consts::FRAC_PI_2).contains(&outer_angle)
    {
        return None;
    }
    let forward = light.direction.map(|value| value / direction_length);
    let up_hint = if forward[1].abs() > 0.95 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let normalize = |value: [f64; 3]| {
        let length = value.iter().map(|axis| axis * axis).sum::<f64>().sqrt();
        value.map(|axis| axis / length.max(1e-12))
    };
    let cross = |left: [f64; 3], right: [f64; 3]| {
        [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]
    };
    let dot =
        |left: [f64; 3], right: [f64; 3]| left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
    let right = normalize(cross(forward, up_hint));
    let up = cross(right, forward);
    let backward = forward.map(|value| -value);
    let position = light.position;
    let view_rows =
        [right, up, backward].map(|basis| [basis[0], basis[1], basis[2], -dot(basis, position)]);
    let depths = points
        .iter()
        .map(|point| {
            -((view_rows[2][0] * point[0])
                + (view_rows[2][1] * point[1])
                + (view_rows[2][2] * point[2])
                + view_rows[2][3])
        })
        .filter(|depth| depth.is_finite() && *depth > 0.0)
        .collect::<Vec<_>>();
    let min_depth = depths.iter().copied().fold(f64::INFINITY, f64::min);
    let max_depth = depths.iter().copied().fold(0.0, f64::max);
    if !min_depth.is_finite() || !max_depth.is_finite() || max_depth <= 0.0 {
        return None;
    }
    let depth_padding = ((max_depth - min_depth) * 0.02).max(0.05);
    let near = (min_depth - depth_padding).max(0.01);
    let content_far = max_depth + depth_padding;
    let far = light
        .range
        .map_or(content_far, |range| content_far.min(range));
    if !far.is_finite() || far <= near + 0.01 {
        return None;
    }
    Some(CameraGpu {
        view_rows: [
            view_rows[0].map(|value| value as f32),
            view_rows[1].map(|value| value as f32),
            view_rows[2].map(|value| value as f32),
            [0.0, 0.0, 0.0, 1.0],
        ],
        projection: [(outer_angle * 2.0) as f32, 1.0, near as f32, far as f32],
        flags: [0, 0, draws.len() as u32, 0],
        shadow: [0, 0, 0, 1],
    })
}

fn shadow_camera(draws: &[DrawInstance], light: &crate::model::LightFrame) -> Option<CameraGpu> {
    match light.kind.as_str() {
        "directional" => directional_shadow_camera(draws, light.direction),
        "spot" => spot_shadow_camera(draws, light),
        _ => None,
    }
}

fn shadow_signature(
    shadow_camera: Option<&CameraGpu>,
    draws: &[DrawInstance],
    meshes: &[MeshData],
    vertex_shader: Option<&str>,
    shader_sources: &HashMap<String, Arc<str>>,
) -> Option<u64> {
    let camera = shadow_camera?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytemuck::bytes_of(camera).hash(&mut hasher);
    vertex_shader.hash(&mut hasher);
    for draw in draws {
        if !draw.casts_shadow {
            continue;
        }
        meshes[draw.mesh].key.hash(&mut hasher);
        bytemuck::bytes_of(&draw.gpu.rows).hash(&mut hasher);
        bytemuck::bytes_of(&draw.gpu.shader_params).hash(&mut hasher);
        draw.gpu.color[3].to_bits().hash(&mut hasher);
        draw.opaque.hash(&mut hasher);
        draw.imported_alpha_mode.hash(&mut hasher);
        draw.shader_program.hash(&mut hasher);
        draw.image
            .as_ref()
            .map(|image| image.hash)
            .hash(&mut hasher);
        draw.image_sampler.hash(&mut hasher);
        if let Some(source) = draw
            .shader_program
            .as_ref()
            .and_then(|id| shader_sources.get(id))
        {
            source.hash(&mut hasher);
        }
    }
    Some(hasher.finish())
}

pub(super) fn build_frame(
    frame: &Frame<'_>,
    aspect: f32,
    target_format: wgpu::TextureFormat,
) -> Result<SceneFrame, String> {
    if frame.lights.len() > MAX_GPU_LIGHTS {
        return Err(format!(
            "mobile forward renderer supports at most {MAX_GPU_LIGHTS} lights per frame"
        ));
    }
    let view = renderer::inverse(&frame.camera.world_matrix)?;
    let (projection, projection_kind) = match frame.camera.projection {
        CameraProjection::Perspective { fov_y } => (
            [
                fov_y as f32,
                aspect,
                frame.camera.near as f32,
                frame.camera.far as f32,
            ],
            0,
        ),
        CameraProjection::Orthographic { height } => (
            [
                height as f32,
                aspect,
                frame.camera.near as f32,
                frame.camera.far as f32,
            ],
            1,
        ),
    };
    let lights = frame
        .lights
        .iter()
        .map(|light| {
            let position = renderer::transform_point(&view, light.position);
            let direction_end =
                std::array::from_fn(|axis| light.position[axis] + light.direction[axis]);
            let mut direction = renderer::transform_point(&view, direction_end);
            for axis in 0..3 {
                direction[axis] -= position[axis];
            }
            let length = direction
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if length > 1e-12 {
                direction = direction.map(|value| value / length);
            }
            let kind = match light.kind.as_str() {
                "ambient" => 0,
                "directional" => 1,
                "point" => 2,
                "spot" => 3,
                _ => 0,
            };
            LightGpu {
                position: [
                    position[0] as f32,
                    position[1] as f32,
                    position[2] as f32,
                    1.0,
                ],
                direction: [
                    direction[0] as f32,
                    direction[1] as f32,
                    direction[2] as f32,
                    light.range.map_or(-1.0, |range| range as f32),
                ],
                color_intensity: [
                    light.color[0] as f32,
                    light.color[1] as f32,
                    light.color[2] as f32,
                    light.intensity as f32,
                ],
                kind: [kind, u32::from(light.casts_shadow), 0, 0],
                cone_cosines: [
                    light.inner_angle.map_or(1.0, |angle| angle.cos() as f32),
                    light.outer_angle.map_or(1.0, |angle| angle.cos() as f32),
                    0.0,
                    0.0,
                ],
                shadow_params: [
                    light.shadow_bias as f32,
                    light.shadow_filter_radius as f32,
                    0.0,
                    0.0,
                ],
            }
        })
        .collect::<Vec<_>>();
    let mut mesh_indices = HashMap::<MeshKey, usize>::new();
    let mut meshes = Vec::<MeshData>::new();
    let mut draws = Vec::with_capacity(frame.instances.len());

    for instance in &frame.instances {
        let local_bounds = match instance.geometry_kind.as_str() {
            "plane" => ([-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]),
            "box" => ([-0.5; 3], [0.5; 3]),
            "glyph_mesh" => {
                let triangles = instance
                    .glyph_mesh
                    .as_ref()
                    .ok_or("glyph instance has no fill mesh")?;
                if triangles.is_empty() {
                    continue;
                }
                let half = instance.size.map(|dimension| dimension * 0.5);
                (half.map(|value| -value), half)
            }
            "scene_mesh" => {
                let vertices = instance
                    .scene_vertices
                    .as_ref()
                    .ok_or("scene mesh instance has no vertex data")?;
                if vertices.is_empty() {
                    continue;
                }
                let mut low = [f64::INFINITY; 3];
                let mut high = [f64::NEG_INFINITY; 3];
                for vertex in vertices.iter() {
                    for axis in 0..3 {
                        low[axis] = low[axis].min(vertex.position[axis]);
                        high[axis] = high[axis].max(vertex.position[axis]);
                    }
                }
                (low, high)
            }
            other => return Err(format!("GPU editor does not support geometry {other:?}")),
        };
        let key = match instance.geometry_kind.as_str() {
            "plane" => MeshKey::Plane,
            "box" => MeshKey::Box,
            "glyph_mesh" => {
                let triangles = instance
                    .glyph_mesh
                    .as_ref()
                    .ok_or("glyph instance has no fill mesh")?;
                MeshKey::Glyph {
                    mesh: Arc::clone(triangles),
                    geometry_hash: instance
                        .glyph_hash
                        .ok_or("glyph instance has no mesh fingerprint")?,
                }
            }
            "scene_mesh" => MeshKey::Imported {
                mesh: Arc::clone(
                    instance
                        .scene_vertices
                        .as_ref()
                        .ok_or("scene mesh instance has no vertex data")?,
                ),
                geometry_hash: instance
                    .glyph_hash
                    .ok_or("scene mesh instance has no fingerprint")?,
            },
            other => return Err(format!("GPU editor does not support geometry {other:?}")),
        };
        let mesh = if let Some(index) = mesh_indices.get(&key) {
            *index
        } else {
            let index = meshes.len();
            mesh_indices.insert(key.clone(), index);
            meshes.push(MeshData { key });
            index
        };
        let rgb = instance.color.map(|channel| channel.clamp(0.0, 1.0));
        let color = [
            rgb[0] as f32,
            rgb[1] as f32,
            rgb[2] as f32,
            (rgb[3] * instance.opacity.clamp(0.0, 1.0)) as f32,
        ];
        let shading_mode = match instance.shading {
            crate::model::ShadingModel::Unlit => 0,
            crate::model::ShadingModel::Phong => 1,
            crate::model::ShadingModel::Principled => 2,
        };
        let scene_color = instance
            .shader_program
            .as_ref()
            .is_some_and(|shader| frame.scene_color_shader_ids.contains(shader));
        let mut rows = instance
            .world_matrix
            .map(|row| row.map(|value| value as f32));
        let scaled_axes = match instance.geometry_kind.as_str() {
            "plane" => 2,
            "box" => 3,
            _ => 0,
        };
        // Plane and box dimensions live in the instance matrix so their unit meshes batch together.
        for row in &mut rows {
            for axis in 0..scaled_axes {
                row[axis] *= instance.size[axis] as f32;
            }
        }
        let imported_alpha_mode = instance.imported_alpha_mode;
        let imported_texture_flags = instance.imported_material.as_ref().map_or(0, |material| {
            u32::from(material.normal_texture.is_some())
                | (u32::from(material.metallic_roughness_texture.is_some()) << 1)
                | (u32::from(material.occlusion_texture.is_some()) << 2)
                | (u32::from(material.emissive_texture.is_some()) << 3)
                | 16
        });
        draws.push(DrawInstance {
            mesh,
            gpu: InstanceGpu {
                rows,
                color,
                bounds_min: [
                    local_bounds.0[0] as f32,
                    local_bounds.0[1] as f32,
                    local_bounds.0[2] as f32,
                    0.0,
                ],
                bounds_max: [
                    local_bounds.1[0] as f32,
                    local_bounds.1[1] as f32,
                    local_bounds.1[2] as f32,
                    0.0,
                ],
                batch: [0, 0, shading_mode, 0],
                material: instance.lighting.map(|value| value as f32),
                shader_params: instance.shader_params.map(|value| value as f32),
                imported_params: instance
                    .imported_material
                    .as_ref()
                    .map_or([0.0; 4], |material| {
                        [
                            material.normal_scale as f32,
                            material.occlusion_strength as f32,
                            material.emissive_strength as f32,
                            f32::from_bits(imported_texture_flags),
                        ]
                    }),
                shadow_flags: [
                    u32::from(instance.casts_shadow),
                    u32::from(instance.receives_shadow),
                    u32::from(instance.image.is_some()),
                    u32::from(instance.scene_vertices.is_some()),
                ],
            },
            opaque: match imported_alpha_mode {
                1 | 2 => true,
                3 => false,
                _ => color[3] >= 0.999 && !scene_color && instance.image.is_none(),
            },
            shader_program: instance.shader_program.clone(),
            scene_color,
            casts_shadow: instance.casts_shadow,
            receives_shadow: instance.receives_shadow,
            image: instance.image.clone(),
            image_sampler: instance.image_sampler,
            imported_material: instance.imported_material.clone(),
            imported_alpha_mode,
        });
    }

    // Keep opaque surfaces first for stable early depth writes; translucent surfaces retain
    // authored order and test against opaque depth without writing their own depth.
    // Opaque instances can be reordered by mesh and drawn in one instanced call. Transparent
    // instances retain their authored order and remain individual draws.
    for draw in &draws {
        if let Some(shader) = &draw.shader_program {
            if !frame.shader_sources.contains_key(shader) {
                return Err(format!(
                    "material references unavailable surface shader {shader:?}"
                ));
            }
        }
    }
    draws.sort_by_key(|draw| {
        (
            !draw.opaque,
            if draw.opaque { draw.mesh } else { 0 },
            if draw.opaque {
                draw.shader_program.clone()
            } else {
                None
            },
        )
    });
    let mut batches = Vec::new();
    let mut start = 0;
    let mut compact_base = 0;
    while start < draws.len() {
        let draw = &draws[start];
        let batch_mesh = draw.mesh;
        let opaque = draw.opaque;
        let shader_program = draw.shader_program.clone();
        let scene_color = draw.scene_color;
        let texture_key = material_texture_key(draw);
        let imported_alpha_mode = draw.imported_alpha_mode;
        let mut end = start + 1;
        while opaque
            && end < draws.len()
            && draws[end].opaque
            && draws[end].mesh == batch_mesh
            && draws[end].shader_program == shader_program
            && draws[end].imported_alpha_mode == imported_alpha_mode
            && material_texture_key(&draws[end]) == texture_key
        {
            end += 1;
        }
        let batch_index =
            u32::try_from(batches.len()).map_err(|_| "GPU editor batch count exceeds u32")?;
        let base = u32::try_from(compact_base)
            .map_err(|_| "GPU editor compact instance count exceeds u32")?;
        for instance in &mut draws[start..end] {
            let shading_mode = instance.gpu.batch[2];
            let shadow_flags = instance.gpu.shadow_flags;
            let packed_flags = u32::from(frame.vertex_shader.is_some())
                | (shadow_flags[0] << 1)
                | (shadow_flags[1] << 2)
                | (shadow_flags[2] << 3)
                | (shadow_flags[3] << 4)
                | (u32::from(instance.imported_alpha_mode) << 5)
                | (u32::from(instance.imported_material.is_some()) << 7);
            instance.gpu.batch = [batch_index, base, shading_mode, packed_flags];
        }
        batches.push(DrawBatch {
            mesh: batch_mesh,
            first_instance: start,
            instance_count: end - start,
            compact_base,
            shader_program,
            scene_color,
            material_texture_key: texture_key,
            imported_alpha_mode,
        });
        compact_base += end - start;
        start = end;
    }
    let instance_count =
        u32::try_from(draws.len()).map_err(|_| "GPU editor instance count exceeds u32")?;
    let shadow_light_index = frame
        .lights
        .iter()
        .position(|light| light.casts_shadow)
        .map(|index| index as u32);
    if frame
        .lights
        .iter()
        .filter(|light| light.casts_shadow)
        .count()
        > 1
    {
        return Err("mobile renderer supports one directional or spot shadow-casting light".into());
    }
    let shadow_camera =
        shadow_light_index.and_then(|index| shadow_camera(&draws, &frame.lights[index as usize]));
    let shadow_resolution = shadow_camera.as_ref().and_then(|_| {
        shadow_light_index.map(|index| frame.lights[index as usize].shadow_resolution)
    });
    let vertex_shader = frame.vertex_shader.clone();
    let shader_sources: HashMap<_, _> = frame.shader_sources.clone().into_iter().collect();
    let shadow_signature = shadow_signature(
        shadow_camera.as_ref(),
        &draws,
        &meshes,
        vertex_shader.as_deref(),
        &shader_sources,
    );
    let has_transparent_shadow_casters = draws
        .iter()
        .any(|draw| draw.casts_shadow && (!draw.opaque || draw.shader_program.is_some()));
    let has_mask_shadow_casters = draws
        .iter()
        .any(|draw| draw.casts_shadow && draw.imported_alpha_mode == 2);
    let requires_scene_color = draws.iter().any(|draw| draw.scene_color);
    Ok(SceneFrame {
        camera: CameraGpu {
            view_rows: view.map(|row| row.map(|value| value as f32)),
            projection,
            flags: [
                projection_kind,
                u32::from(target_format.is_srgb()),
                instance_count,
                lights.len() as u32,
            ],
            shadow: [
                shadow_light_index.unwrap_or(0),
                u32::from(shadow_camera.is_some()),
                u32::from(has_transparent_shadow_casters),
                0,
            ],
        },
        shadow_camera,
        shadow_resolution,
        shadow_signature,
        has_mask_shadow_casters,
        camera_effect_shaders: frame.camera_effect_shaders.clone(),
        requires_scene_color,
        vertex_shader,
        lights,
        meshes,
        draws,
        batches,
        shader_sources,
    })
}

struct GpuCallback {
    frame: Arc<SceneFrame>,
    effect_sources: Vec<Arc<str>>,
    rect: egui::Rect,
    target_format: wgpu::TextureFormat,
    settings: Arc<RenderSettings>,
    emit_stats: bool,
}

impl CallbackTrait for GpuCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if callback_resources.get::<GpuResources>().is_none() {
            callback_resources.insert(GpuResources::new(
                device,
                self.target_format,
                &self.settings,
            ));
        }
        let resources = callback_resources.get_mut::<GpuResources>().unwrap();
        if resources.target_format != self.target_format || resources.settings != *self.settings {
            *resources = GpuResources::new(device, self.target_format, &self.settings);
        }
        if !self.effect_sources.is_empty() {
            resources.ensure_camera_effect_pipelines(device, &self.effect_sources);
            let pixels_per_point = screen_descriptor.pixels_per_point;
            let size = [
                (self.rect.width() * pixels_per_point).round().max(1.0) as u32,
                (self.rect.height() * pixels_per_point).round().max(1.0) as u32,
            ];
            resources.ensure_scene_target(
                device,
                size,
                self.effect_sources.len(),
                self.frame.requires_scene_color,
            );
        }
        if let Some(resolution) = self.frame.shadow_resolution {
            resources.ensure_shadow_resolution(device, resolution);
        }
        let same_frame = resources
            .prepared_frame
            .as_ref()
            .is_some_and(|prepared| Arc::ptr_eq(prepared, &self.frame));
        let shadow_cache_hit = self.frame.shadow_signature.is_some()
            && self.frame.shadow_resolution == Some(resources.shadow_resolution)
            && resources.shadow_signature == self.frame.shadow_signature;
        if !same_frame {
            let started = std::time::Instant::now();
            let (cache_hits, cache_misses) = resources.prepare_frame(device, queue, &self.frame);
            if !self.frame.draws.is_empty() {
                let mut pass = egui_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mogf-editor-cull-and-compact"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&resources.cull_pipeline);
                pass.set_bind_group(0, &resources.cull_bind_group, &[]);
                pass.dispatch_workgroups((self.frame.draws.len() as u32).div_ceil(64), 1, 1);
            }
            resources.prepared_frame = Some(self.frame.clone());
            if self.emit_stats {
                resources.stats_prepare_ms += started.elapsed().as_secs_f64() * 1000.0;
                resources.stats_prepares += 1;
                resources.stats_mesh_hits += cache_hits as u64;
                resources.stats_mesh_misses += cache_misses as u64;
                resources.stats_batches += self.frame.batches.len() as u64;
                resources.stats_instances += self.frame.draws.len() as u64;
                if self.frame.shadow_signature.is_some() {
                    if shadow_cache_hit {
                        resources.stats_shadow_reuses += 1;
                    } else {
                        resources.stats_shadow_renders += 1;
                    }
                }
                if resources.stats_started.elapsed().as_secs_f64() >= 1.0 {
                    let samples = resources.stats_prepares.max(1) as f64;
                    eprintln!(
                        "live GPU prepare: {:.2} ms avg across {} frames · meshes {} hit / {} upload · {:.1} avg batches / {:.1} avg instances · shadows {} render / {} reuse",
                        resources.stats_prepare_ms / samples,
                        resources.stats_prepares,
                        resources.stats_mesh_hits,
                        resources.stats_mesh_misses,
                        resources.stats_batches as f64 / samples,
                        resources.stats_instances as f64 / samples,
                        resources.stats_shadow_renders,
                        resources.stats_shadow_reuses,
                    );
                    resources.stats_started = std::time::Instant::now();
                    resources.stats_prepare_ms = 0.0;
                    resources.stats_prepares = 0;
                    resources.stats_mesh_hits = 0;
                    resources.stats_mesh_misses = 0;
                    resources.stats_batches = 0;
                    resources.stats_instances = 0;
                    resources.stats_shadow_renders = 0;
                    resources.stats_shadow_reuses = 0;
                }
            }
        }
        if !same_frame && self.frame.shadow_signature.is_some() && !shadow_cache_hit {
            {
                let mut pass = egui_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mogf-editor-shadow-cull-and-compact"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&resources.cull_pipeline);
                pass.set_bind_group(0, &resources.shadow_cull_bind_group, &[]);
                pass.dispatch_workgroups((self.frame.draws.len() as u32).div_ceil(64), 1, 1);
            }
            let mut pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mogf-editor-directional-shadow-pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &resources.shadow_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            draw_shadow(&self.frame, resources, &mut pass, ShadowPassKind::Opaque);
            drop(pass);
            if self.frame.camera.shadow[2] != 0 {
                let mut alpha_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mogf-editor-directional-shadow-alpha-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &resources.shadow_transmittance_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &resources.shadow_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                draw_shadow(
                    &self.frame,
                    resources,
                    &mut alpha_pass,
                    ShadowPassKind::Alpha,
                );
            }
            if self.frame.has_mask_shadow_casters {
                let mut mask_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mogf-editor-directional-shadow-mask-pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &resources.shadow_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                draw_shadow(&self.frame, resources, &mut mask_pass, ShadowPassKind::Mask);
            }
            resources.shadow_signature = self.frame.shadow_signature;
        }
        if !self.effect_sources.is_empty() {
            let target = resources.scene_target.as_ref().unwrap();
            if self.frame.requires_scene_color {
                {
                    let mut opaque_pass =
                        egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("mogf-editor-scene-color-opaque-pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: target
                                    .multisampled_color_view
                                    .as_ref()
                                    .unwrap_or(&target.color_view),
                                resolve_target: target
                                    .multisampled_color_view
                                    .as_ref()
                                    .map(|_| &target.color_view),
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: target.depth_view.as_ref().map(|view| {
                                wgpu::RenderPassDepthStencilAttachment {
                                    view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                }
                            }),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                    draw_scene(
                        &self.frame,
                        resources,
                        &mut opaque_pass,
                        SceneDrawFilter::Opaque,
                        &resources.default_scene_color_bind_group,
                    );
                }
                let snapshot = target
                    .scene_color_snapshot
                    .as_ref()
                    .expect("scene snapshot exists");
                egui_encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &target.color,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: snapshot,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: target.size[0],
                        height: target.size[1],
                        depth_or_array_layers: 1,
                    },
                );
                let scene_bind_group = target
                    .scene_color_bind_group
                    .as_ref()
                    .expect("scene snapshot binding exists");
                let mut transparent_pass =
                    egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("mogf-editor-scene-color-transparent-pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: target
                                .multisampled_color_view
                                .as_ref()
                                .unwrap_or(&target.color_view),
                            resolve_target: target
                                .multisampled_color_view
                                .as_ref()
                                .map(|_| &target.color_view),
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: target.depth_view.as_ref().map(|view| {
                            wgpu::RenderPassDepthStencilAttachment {
                                view,
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                draw_scene(
                    &self.frame,
                    resources,
                    &mut transparent_pass,
                    SceneDrawFilter::Transparent,
                    scene_bind_group,
                );
            } else {
                let mut pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mogf-editor-effect-scene-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target
                            .multisampled_color_view
                            .as_ref()
                            .unwrap_or(&target.color_view),
                        resolve_target: target
                            .multisampled_color_view
                            .as_ref()
                            .map(|_| &target.color_view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: target.depth_view.as_ref().map(|view| {
                        wgpu::RenderPassDepthStencilAttachment {
                            view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                draw_scene(
                    &self.frame,
                    resources,
                    &mut pass,
                    SceneDrawFilter::All,
                    &resources.default_scene_color_bind_group,
                );
            }
            for effect_index in 0..self.effect_sources.len().saturating_sub(1) {
                let target = resources.scene_target.as_ref().unwrap();
                let output = &target.effect_outputs[effect_index];
                let pipeline = &resources.effect_pipelines[&self.effect_sources[effect_index]];
                let mut effect_pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mogf-editor-camera-effect-chain-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output
                            .multisampled_color_view
                            .as_ref()
                            .unwrap_or(&output.color_view),
                        resolve_target: output
                            .multisampled_color_view
                            .as_ref()
                            .map(|_| &output.color_view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &target.effect_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                effect_pass.set_pipeline(pipeline);
                effect_pass.set_bind_group(0, &target.effect_bind_groups[effect_index], &[]);
                effect_pass.draw(0..3, 0..1);
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        let Some(resources) = callback_resources.get::<GpuResources>() else {
            return;
        };
        if !self.effect_sources.is_empty()
            && let Some(target) = &resources.scene_target
        {
            let effect_index = self.effect_sources.len() - 1;
            if let Some(pipeline) = resources
                .effect_pipelines
                .get(&self.effect_sources[effect_index])
            {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &target.effect_bind_groups[effect_index], &[]);
            } else {
                return;
            }
            render_pass.draw(0..3, 0..1);
        } else {
            draw_scene(
                &self.frame,
                resources,
                render_pass,
                SceneDrawFilter::All,
                &resources.default_scene_color_bind_group,
            );
        }
    }
}

#[derive(Clone, Copy)]
enum SceneDrawFilter {
    All,
    Opaque,
    Transparent,
}

fn draw_scene<'a>(
    frame: &SceneFrame,
    resources: &GpuResources,
    render_pass: &mut wgpu::RenderPass<'a>,
    filter: SceneDrawFilter,
    scene_color_bind_group: &wgpu::BindGroup,
) {
    render_pass.set_bind_group(0, &resources.render_bind_group, &[]);
    render_pass.set_bind_group(1, &resources.shadow_bind_group, &[]);
    for (batch_index, batch) in frame.batches.iter().enumerate() {
        let opaque = frame.draws[batch.first_instance].opaque;
        if matches!(filter, SceneDrawFilter::Opaque) && !opaque
            || matches!(filter, SceneDrawFilter::Transparent) && opaque
        {
            continue;
        }
        render_pass.set_bind_group(
            2,
            if batch.scene_color {
                scene_color_bind_group
            } else {
                &resources.default_scene_color_bind_group
            },
            &[],
        );
        render_pass.set_bind_group(
            3,
            resources
                .image_cache
                .get(&batch.material_texture_key)
                .unwrap_or(&resources.default_image_bind_group),
            &[],
        );
        let mesh = &frame.meshes[batch.mesh];
        let Some(cached) = resources
            .mesh_cache
            .get(&mesh.key)
            .or_else(|| resources.frame_meshes.get(&mesh.key))
        else {
            continue;
        };
        let pipelines = batch
            .shader_program
            .as_ref()
            .and_then(|id| resources.project_surface_pipelines.get(id))
            .map(|program| &program.pipelines);
        let vertex_pipelines = frame
            .vertex_shader
            .as_ref()
            .and_then(|source| {
                resources
                    .project_vertex_pipelines
                    .as_ref()
                    .filter(|p| p.source == *source)
            })
            .map(|program| &program.pipelines);
        let selected = pipelines.or(vertex_pipelines);
        render_pass.set_pipeline(if opaque {
            selected.map_or(&resources.opaque_pipeline, |pipelines| &pipelines.opaque)
        } else {
            selected.map_or(&resources.transparent_pipeline, |pipelines| {
                &pipelines.transparent
            })
        });
        render_pass.set_vertex_buffer(0, cached.buffer.slice(..));
        let byte_offset = (batch.compact_base * std::mem::size_of::<InstanceGpu>()) as u64;
        let byte_size = (batch.instance_count * std::mem::size_of::<InstanceGpu>()) as u64;
        render_pass.set_vertex_buffer(
            1,
            resources
                .compact_buffer
                .slice(byte_offset..byte_offset + byte_size),
        );
        render_pass.draw_indirect(
            &resources.indirect_buffer,
            (batch_index * std::mem::size_of::<DrawIndirectGpu>()) as u64,
        );
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShadowPassKind {
    Opaque,
    Alpha,
    Mask,
}

fn draw_shadow<'a>(
    frame: &SceneFrame,
    resources: &GpuResources,
    render_pass: &mut wgpu::RenderPass<'a>,
    pass_kind: ShadowPassKind,
) {
    let project_pipelines = frame.vertex_shader.as_ref().and_then(|source| {
        resources
            .project_vertex_pipelines
            .as_ref()
            .filter(|p| p.source == *source)
    });
    let default_pipeline = match pass_kind {
        ShadowPassKind::Opaque => {
            project_pipelines.map_or(&resources.shadow_pipeline, |program| &program.shadow_opaque)
        }
        ShadowPassKind::Alpha => project_pipelines
            .map_or(&resources.shadow_alpha_pipeline, |program| {
                &program.shadow_alpha
            }),
        ShadowPassKind::Mask => project_pipelines
            .map_or(&resources.shadow_mask_pipeline, |program| {
                &program.shadow_mask
            }),
    };
    render_pass.set_pipeline(default_pipeline);
    render_pass.set_bind_group(0, &resources.shadow_caster_bind_group, &[]);
    if pass_kind != ShadowPassKind::Opaque {
        render_pass.set_bind_group(1, &resources.shadow_alpha_bind_group, &[]);
        render_pass.set_bind_group(2, &resources.default_scene_color_bind_group, &[]);
        render_pass.set_bind_group(3, &resources.default_image_bind_group, &[]);
    }
    for (batch_index, batch) in frame.batches.iter().enumerate() {
        let opaque = frame.draws[batch.first_instance].opaque;
        let masked = batch.imported_alpha_mode == 2;
        let skip = match pass_kind {
            ShadowPassKind::Opaque => !opaque || masked || batch.shader_program.is_some(),
            ShadowPassKind::Alpha => masked || (opaque && batch.shader_program.is_none()),
            ShadowPassKind::Mask => !masked,
        };
        if skip {
            continue;
        }
        if pass_kind != ShadowPassKind::Opaque {
            render_pass.set_bind_group(
                3,
                resources
                    .image_cache
                    .get(&batch.material_texture_key)
                    .unwrap_or(&resources.default_image_bind_group),
                &[],
            );
            let material_pipeline = if pass_kind == ShadowPassKind::Alpha {
                batch
                    .shader_program
                    .as_ref()
                    .and_then(|id| resources.project_surface_pipelines.get(id))
                    .map(|program| &program.shadow_alpha)
            } else {
                None
            };
            render_pass.set_pipeline(material_pipeline.unwrap_or(default_pipeline));
        }
        let mesh = &frame.meshes[batch.mesh];
        let Some(cached) = resources
            .mesh_cache
            .get(&mesh.key)
            .or_else(|| resources.frame_meshes.get(&mesh.key))
        else {
            continue;
        };
        render_pass.set_vertex_buffer(0, cached.buffer.slice(..));
        let byte_offset = (batch.compact_base * std::mem::size_of::<InstanceGpu>()) as u64;
        let byte_size = (batch.instance_count * std::mem::size_of::<InstanceGpu>()) as u64;
        render_pass.set_vertex_buffer(
            1,
            resources
                .shadow_compact_buffer
                .slice(byte_offset..byte_offset + byte_size),
        );
        render_pass.draw_indirect(
            &resources.shadow_indirect_buffer,
            (batch_index * std::mem::size_of::<DrawIndirectGpu>()) as u64,
        );
    }
}

fn create_surface_pipelines(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    settings: &RenderSettings,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label_prefix: &str,
) -> SurfacePipelines {
    let vertex_attributes = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 16,
            shader_location: 9,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 24,
            shader_location: 10,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 36,
            shader_location: 11,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 44,
            shader_location: 12,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 52,
            shader_location: 13,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 60,
            shader_location: 14,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 68,
            shader_location: 8,
        },
    ];
    let float4 = wgpu::VertexFormat::Float32x4;
    let instance_attributes = [
        wgpu::VertexAttribute {
            format: float4,
            offset: std::mem::offset_of!(InstanceGpu, rows) as u64,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: float4,
            offset: std::mem::offset_of!(InstanceGpu, rows) as u64 + 16,
            shader_location: 2,
        },
        wgpu::VertexAttribute {
            format: float4,
            offset: std::mem::offset_of!(InstanceGpu, rows) as u64 + 32,
            shader_location: 3,
        },
        wgpu::VertexAttribute {
            format: float4,
            offset: std::mem::offset_of!(InstanceGpu, rows) as u64 + 48,
            shader_location: 4,
        },
        wgpu::VertexAttribute {
            format: float4,
            offset: std::mem::offset_of!(InstanceGpu, color) as u64,
            shader_location: 5,
        },
        wgpu::VertexAttribute {
            format: float4,
            offset: std::mem::offset_of!(InstanceGpu, material) as u64,
            shader_location: 6,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32x4,
            offset: std::mem::offset_of!(InstanceGpu, batch) as u64,
            shader_location: 7,
        },
        wgpu::VertexAttribute {
            format: float4,
            offset: std::mem::offset_of!(InstanceGpu, imported_params) as u64,
            shader_location: 15,
        },
    ];
    let buffers = [
        wgpu::VertexBufferLayout {
            array_stride: (std::mem::size_of::<f32>() * 21) as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attributes,
        },
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceGpu>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &instance_attributes,
        },
    ];
    let sample_count = u32::from(settings.sample_count);
    let make_pipeline = |opaque: bool| {
        let label = format!(
            "{label_prefix}-{}",
            if opaque { "opaque" } else { "transparent" }
        );
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: match settings.cull {
                    CullMode::None => None,
                    CullMode::Front => Some(wgpu::Face::Front),
                    CullMode::Back => Some(wgpu::Face::Back),
                },
                ..Default::default()
            },
            depth_stencil: (settings.depth != DepthMode::Disabled).then_some(
                wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: Some(settings.depth == DepthMode::TestWrite && opaque),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                },
            ),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: (settings.blend == BlendMode::Alpha)
                        .then_some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    SurfacePipelines {
        opaque: make_pipeline(true),
        transparent: make_pipeline(false),
    }
}

fn create_shadow_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    render_bind_group_layout: &wgpu::BindGroupLayout,
    alpha_pipeline_layout: &wgpu::PipelineLayout,
) -> ShadowPipelines {
    const FLOAT4: wgpu::VertexFormat = wgpu::VertexFormat::Float32x4;
    let vertex_attributes = [
        wgpu::VertexAttribute {
            format: FLOAT4,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 16,
            shader_location: 9,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 24,
            shader_location: 10,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 36,
            shader_location: 11,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 44,
            shader_location: 12,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 52,
            shader_location: 13,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 60,
            shader_location: 14,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 68,
            shader_location: 8,
        },
    ];
    let instance_attributes = [
        wgpu::VertexAttribute {
            format: FLOAT4,
            offset: std::mem::offset_of!(InstanceGpu, rows) as u64,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: FLOAT4,
            offset: std::mem::offset_of!(InstanceGpu, rows) as u64 + 16,
            shader_location: 2,
        },
        wgpu::VertexAttribute {
            format: FLOAT4,
            offset: std::mem::offset_of!(InstanceGpu, rows) as u64 + 32,
            shader_location: 3,
        },
        wgpu::VertexAttribute {
            format: FLOAT4,
            offset: std::mem::offset_of!(InstanceGpu, rows) as u64 + 48,
            shader_location: 4,
        },
        wgpu::VertexAttribute {
            format: FLOAT4,
            offset: std::mem::offset_of!(InstanceGpu, color) as u64,
            shader_location: 5,
        },
        wgpu::VertexAttribute {
            format: FLOAT4,
            offset: std::mem::offset_of!(InstanceGpu, material) as u64,
            shader_location: 6,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32x4,
            offset: std::mem::offset_of!(InstanceGpu, batch) as u64,
            shader_location: 7,
        },
        wgpu::VertexAttribute {
            format: FLOAT4,
            offset: std::mem::offset_of!(InstanceGpu, imported_params) as u64,
            shader_location: 15,
        },
    ];
    let buffers = [
        wgpu::VertexBufferLayout {
            array_stride: (std::mem::size_of::<f32>() * 21) as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attributes,
        },
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceGpu>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &instance_attributes,
        },
    ];
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mogf-project-directional-shadow-pipeline-layout"),
        bind_group_layouts: &[Some(render_bind_group_layout)],
        immediate_size: 0,
    });
    let make_pipeline = |kind: u8| {
        let alpha = kind == 1;
        let mask = kind == 2;
        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let pipeline_layout = if alpha || mask {
            alpha_pipeline_layout
        } else {
            &layout
        };
        let alpha_targets = [Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::R8Unorm,
            blend: Some(blend),
            write_mask: wgpu::ColorWrites::RED,
        })];
        let targets: &[Option<wgpu::ColorTargetState>] = if alpha { &alpha_targets } else { &[] };
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(match kind {
                1 => "mogf-project-directional-shadow-alpha-pipeline",
                2 => "mogf-project-directional-shadow-mask-pipeline",
                _ => "mogf-project-directional-shadow-pipeline",
            }),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(!alpha),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: if alpha {
                    Default::default()
                } else {
                    wgpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    }
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: (alpha || mask).then_some(wgpu::FragmentState {
                module,
                entry_point: Some(if mask {
                    "fs_shadow_mask"
                } else {
                    "fs_shadow_alpha"
                }),
                compilation_options: Default::default(),
                targets,
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    ShadowPipelines {
        opaque: make_pipeline(0),
        alpha: make_pipeline(1),
        mask: make_pipeline(2),
    }
}

fn create_effect_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mogf-editor-effect-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(8),
                },
                count: None,
            },
        ],
    })
}

fn create_effect_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("mogf-editor-effect-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

fn create_camera_effect_pipeline(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    sample_count: u32,
    source: &str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader_source = include_str!("editor_effect.wgsl").replace("// MOGF_EFFECT_HOOK", source);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mogf-editor-camera-effect-wgsl"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mogf-editor-effect-pipeline-layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mogf-editor-camera-effect-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("mogf_effect_vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("mogf_effect_fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

impl GpuResources {
    fn ensure_shadow_resolution(&mut self, device: &wgpu::Device, resolution: u32) {
        if self.shadow_resolution == resolution {
            return;
        }
        let size = wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 1,
        };
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mogf-editor-shadow-map-resized"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_depth_view = shadow_texture.create_view(&Default::default());
        let transmittance_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mogf-editor-shadow-transmittance-map-resized"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_transmittance_view = transmittance_texture.create_view(&Default::default());
        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mogf-editor-shadow-receiver-bindings-resized"),
            layout: &self.shadow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.shadow_camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&shadow_transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_filter_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_transmittance_sampler),
                },
            ],
        });
        self.shadow_depth_view = shadow_depth_view;
        self.shadow_transmittance_view = shadow_transmittance_view;
        self.shadow_bind_group = shadow_bind_group;
        self.shadow_resolution = resolution;
        self.shadow_signature = None;
    }

    fn ensure_camera_effect_pipelines(&mut self, device: &wgpu::Device, sources: &[Arc<str>]) {
        if self.effect_layout.is_none() {
            self.effect_layout = Some(create_effect_bind_group_layout(device));
        }
        if self.effect_sampler.is_none() {
            self.effect_sampler = Some(create_effect_sampler(device));
        }
        self.effect_pipelines.retain(|source, _| {
            sources
                .iter()
                .any(|active| active.as_ref() == source.as_ref())
                || self
                    .settings
                    .effect_shader
                    .as_deref()
                    .is_some_and(|fallback| fallback == source.as_ref())
        });
        for source in sources {
            if self.effect_pipelines.contains_key(source) {
                continue;
            }
            let pipeline = create_camera_effect_pipeline(
                device,
                self.target_format,
                u32::from(self.settings.sample_count),
                source,
                self.effect_layout
                    .as_ref()
                    .expect("effect layout was created"),
            );
            self.effect_pipelines.insert(Arc::clone(source), pipeline);
        }
    }

    fn ensure_scene_target(
        &mut self,
        device: &wgpu::Device,
        size: [u32; 2],
        effect_count: usize,
        capture_scene_color: bool,
    ) {
        if self.scene_target.as_ref().is_some_and(|target| {
            target.size == size
                && target.effect_outputs.len() + 1 == effect_count
                && target.scene_color_snapshot.is_some() == capture_scene_color
        }) {
            return;
        }
        let layout = self.effect_layout.as_ref().expect("effect layout exists");
        let sampler = self.effect_sampler.as_ref().expect("effect sampler exists");
        let extent = wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mogf-editor-effect-scene-color"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | if capture_scene_color {
                    wgpu::TextureUsages::COPY_SRC
                } else {
                    wgpu::TextureUsages::empty()
                },
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());
        let scene_color_snapshot = capture_scene_color.then(|| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("mogf-editor-scene-color-snapshot"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.target_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        });
        let scene_color_bind_group = scene_color_snapshot.as_ref().map(|texture| {
            let view = texture.create_view(&Default::default());
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mogf-editor-scene-color-snapshot-bindings"),
                layout: &self.scene_color_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.scene_color_sampler),
                    },
                ],
            })
        });
        let multisampled_color_view = (self.settings.sample_count > 1).then(|| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("mogf-editor-effect-scene-msaa"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: u32::from(self.settings.sample_count),
                    dimension: wgpu::TextureDimension::D2,
                    format: self.target_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        });
        let depth_view = (self.settings.depth != DepthMode::Disabled).then(|| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("mogf-editor-effect-scene-depth"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: u32::from(self.settings.sample_count),
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth24Plus,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        });
        let effect_depth_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("mogf-editor-effect-chain-depth"),
                size: extent,
                mip_level_count: 1,
                sample_count: u32::from(self.settings.sample_count),
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24Plus,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let resolution_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mogf-editor-effect-resolution"),
            contents: bytemuck::cast_slice(&[size[0] as f32, size[1] as f32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let effect_outputs = (0..effect_count.saturating_sub(1))
            .map(|_| {
                let color = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("mogf-editor-effect-chain-output"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.target_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let color_view = color.create_view(&Default::default());
                let multisampled_color_view = (self.settings.sample_count > 1).then(|| {
                    device
                        .create_texture(&wgpu::TextureDescriptor {
                            label: Some("mogf-editor-effect-chain-msaa"),
                            size: extent,
                            mip_level_count: 1,
                            sample_count: u32::from(self.settings.sample_count),
                            dimension: wgpu::TextureDimension::D2,
                            format: self.target_format,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                            view_formats: &[],
                        })
                        .create_view(&Default::default())
                });
                EffectTarget {
                    color_view,
                    multisampled_color_view,
                }
            })
            .collect::<Vec<_>>();
        let mut effect_bind_groups = Vec::with_capacity(effect_count);
        for index in 0..effect_count {
            let input_view = if index == 0 {
                &color_view
            } else {
                &effect_outputs[index - 1].color_view
            };
            effect_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mogf-editor-effect-chain-input"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: resolution_buffer.as_entire_binding(),
                    },
                ],
            }));
        }
        self.scene_target = Some(SceneTarget {
            size,
            color,
            color_view,
            scene_color_snapshot,
            scene_color_bind_group,
            multisampled_color_view,
            depth_view,
            effect_depth_view,
            effect_bind_groups,
            effect_outputs,
        });
    }

    fn create_cull_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        camera_buffer: &wgpu::Buffer,
        indirect_buffer: &wgpu::Buffer,
        instance_buffer: &wgpu::Buffer,
        compact_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mogf-editor-instance-culling-bindings"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indirect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: compact_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn create_render_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        camera_buffer: &wgpu::Buffer,
        light_buffer: &wgpu::Buffer,
        compact_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mogf-editor-render-bindings"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: compact_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        settings: &RenderSettings,
    ) -> Self {
        let sample_count = u32::from(settings.sample_count);
        let default_surface = "fn mogf_surface(input: VertexOut) -> vec4<f32> { return mogf_default_lit_surface(input); }";
        let default_vertex = "fn mogf_vertex(input: VertexOut) -> VertexOut { return input; }";
        let shader_source = include_str!("editor_gpu.wgsl")
            .replace(
                "// MOGF_VERTEX_HOOK",
                settings.vertex_shader.as_deref().unwrap_or(default_vertex),
            )
            .replace("// MOGF_GEOMETRY_HOOK", DEFAULT_GEOMETRY_HOOK)
            .replace(
                "// MOGF_SURFACE_HOOK",
                settings
                    .surface_shader
                    .as_deref()
                    .unwrap_or(default_surface),
            )
            .replace(
                "// MOGF_SHADOW_OPACITY_HOOK",
                shadow_opacity_hook(settings.surface_shader.as_deref()),
            );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mogf-editor-instanced-wgsl"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let camera_layout_entry = |visibility| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<CameraGpu>() as u64),
            },
            count: None,
        };
        let indirect_write_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let instance_read_entry = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let compact_write_entry = wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let cull_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mogf-editor-instance-culling-layout"),
                entries: &[
                    camera_layout_entry(wgpu::ShaderStages::COMPUTE),
                    indirect_write_entry,
                    instance_read_entry,
                    compact_write_entry,
                ],
            });
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mogf-editor-render-layout"),
                entries: &[
                    camera_layout_entry(wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let shadow_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mogf-editor-shadow-sampling-layout"),
                entries: &[
                    camera_layout_entry(wgpu::ShaderStages::FRAGMENT),
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let scene_color_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mogf-editor-scene-color-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mogf-editor-camera"),
            size: std::mem::size_of::<CameraGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mogf-editor-shadow-camera"),
            size: std::mem::size_of::<CameraGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_capacity = MAX_GPU_LIGHTS;
        let light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mogf-editor-lights"),
            size: (light_capacity * std::mem::size_of::<LightGpu>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mogf-editor-directional-shadow-map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_depth_view = shadow_texture.create_view(&Default::default());
        let shadow_transmittance_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mogf-editor-shadow-transmittance-map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_transmittance_view =
            shadow_transmittance_texture.create_view(&Default::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mogf-editor-shadow-comparison-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let shadow_filter_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mogf-editor-shadow-filtered-comparison-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let shadow_transmittance_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mogf-editor-shadow-transmittance-filter-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let initial_instances = 256_u64;
        let instance_capacity = initial_instances * std::mem::size_of::<InstanceGpu>() as u64;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mogf-editor-instance-buffer"),
            size: instance_capacity,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let compact_capacity = instance_capacity;
        let compact_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mogf-editor-compacted-instances"),
            size: compact_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let shadow_compact_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mogf-editor-shadow-compacted-instances"),
            size: compact_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let indirect_capacity = initial_instances * std::mem::size_of::<DrawIndirectGpu>() as u64;
        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mogf-editor-indirect-draws"),
            size: indirect_capacity,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mogf-editor-shadow-indirect-draws"),
            size: indirect_capacity,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_bind_group = Self::create_cull_bind_group(
            device,
            &cull_bind_group_layout,
            &camera_buffer,
            &indirect_buffer,
            &instance_buffer,
            &compact_buffer,
        );
        let shadow_cull_bind_group = Self::create_cull_bind_group(
            device,
            &cull_bind_group_layout,
            &shadow_camera_buffer,
            &shadow_indirect_buffer,
            &instance_buffer,
            &shadow_compact_buffer,
        );
        let render_bind_group = Self::create_render_bind_group(
            device,
            &render_bind_group_layout,
            &camera_buffer,
            &light_buffer,
            &compact_buffer,
        );
        let shadow_caster_bind_group = Self::create_render_bind_group(
            device,
            &render_bind_group_layout,
            &shadow_camera_buffer,
            &light_buffer,
            &shadow_compact_buffer,
        );
        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mogf-editor-shadow-receiver-bindings"),
            layout: &shadow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&shadow_transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&shadow_filter_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&shadow_transmittance_sampler),
                },
            ],
        });
        let shadow_alpha_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mogf-editor-shadow-alpha-dummy-depth"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_alpha_transmittance = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mogf-editor-shadow-alpha-dummy-transmittance"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_alpha_depth_view = shadow_alpha_depth.create_view(&Default::default());
        let shadow_alpha_transmittance_view =
            shadow_alpha_transmittance.create_view(&Default::default());
        let shadow_alpha_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mogf-editor-shadow-alpha-bindings"),
            layout: &shadow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_alpha_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&shadow_alpha_transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&shadow_filter_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&shadow_transmittance_sampler),
                },
            ],
        });
        let scene_color_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mogf-editor-scene-color-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let default_scene_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mogf-editor-default-scene-color"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let default_scene_color_view = default_scene_color_texture.create_view(&Default::default());
        let default_scene_color_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mogf-editor-default-scene-color-bindings"),
            layout: &scene_color_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&default_scene_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&scene_color_sampler),
                },
            ],
        });
        let cull_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mogf-editor-culling-pipeline-layout"),
            bind_group_layouts: &[Some(&cull_bind_group_layout)],
            immediate_size: 0,
        });
        let image_bind_group_layout_entries = (0..10)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: if binding % 2 == 0 {
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }
                } else {
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let image_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mogf-image-layout"),
                entries: &image_bind_group_layout_entries,
            });
        let image_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mogf-image-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let default_image_entries = (0..5)
            .flat_map(|slot| {
                [
                    wgpu::BindGroupEntry {
                        binding: slot * 2,
                        resource: wgpu::BindingResource::TextureView(&default_scene_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: slot * 2 + 1,
                        resource: wgpu::BindingResource::Sampler(&image_sampler),
                    },
                ]
            })
            .collect::<Vec<_>>();
        let default_image_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mogf-default-image-bindings"),
            layout: &image_bind_group_layout,
            entries: &default_image_entries,
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mogf-editor-pipeline-layout"),
                bind_group_layouts: &[
                    Some(&render_bind_group_layout),
                    Some(&shadow_bind_group_layout),
                    Some(&scene_color_bind_group_layout),
                    Some(&image_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mogf-editor-instance-culling-pipeline"),
            layout: Some(&cull_pipeline_layout),
            module: &shader,
            entry_point: Some("cull_instances"),
            compilation_options: Default::default(),
            cache: None,
        });
        let float4 = wgpu::VertexFormat::Float32x4;
        let vertex_attributes = [
            wgpu::VertexAttribute {
                format: float4,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 16,
                shader_location: 9,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 24,
                shader_location: 10,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 36,
                shader_location: 11,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 44,
                shader_location: 12,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 52,
                shader_location: 13,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 60,
                shader_location: 14,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 68,
                shader_location: 8,
            },
        ];
        let instance_attributes = [
            wgpu::VertexAttribute {
                format: float4,
                offset: std::mem::offset_of!(InstanceGpu, rows) as u64,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: float4,
                offset: std::mem::offset_of!(InstanceGpu, rows) as u64 + 16,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: float4,
                offset: std::mem::offset_of!(InstanceGpu, rows) as u64 + 32,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: float4,
                offset: std::mem::offset_of!(InstanceGpu, rows) as u64 + 48,
                shader_location: 4,
            },
            wgpu::VertexAttribute {
                format: float4,
                offset: std::mem::offset_of!(InstanceGpu, color) as u64,
                shader_location: 5,
            },
            wgpu::VertexAttribute {
                format: float4,
                offset: std::mem::offset_of!(InstanceGpu, material) as u64,
                shader_location: 6,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32x4,
                offset: std::mem::offset_of!(InstanceGpu, batch) as u64,
                shader_location: 7,
            },
            wgpu::VertexAttribute {
                format: float4,
                offset: std::mem::offset_of!(InstanceGpu, imported_params) as u64,
                shader_location: 15,
            },
        ];
        let buffers = [
            wgpu::VertexBufferLayout {
                array_stride: (std::mem::size_of::<f32>() * 21) as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &vertex_attributes,
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<InstanceGpu>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &instance_attributes,
            },
        ];
        let make_pipeline = |label: &'static str, opaque: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: match settings.cull {
                        CullMode::None => None,
                        CullMode::Front => Some(wgpu::Face::Front),
                        CullMode::Back => Some(wgpu::Face::Back),
                    },
                    ..Default::default()
                },
                depth_stencil: (settings.depth != DepthMode::Disabled).then_some(
                    wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth24Plus,
                        depth_write_enabled: Some(settings.depth == DepthMode::TestWrite && opaque),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: Default::default(),
                        bias: Default::default(),
                    },
                ),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: (settings.blend == BlendMode::Alpha)
                            .then_some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let opaque_pipeline = make_pipeline("mogf-editor-opaque-pipeline", true);
        let transparent_pipeline = make_pipeline("mogf-editor-transparent-pipeline", false);
        let shadow_pipelines = create_shadow_pipeline(
            device,
            &shader,
            &render_bind_group_layout,
            &render_pipeline_layout,
        );
        let (effect_layout, effect_sampler, effect_pipelines) =
            if let Some(effect_source) = settings.effect_shader.as_deref() {
                let layout = create_effect_bind_group_layout(device);
                let sampler = create_effect_sampler(device);
                let pipeline = create_camera_effect_pipeline(
                    device,
                    target_format,
                    sample_count,
                    effect_source,
                    &layout,
                );
                let source: Arc<str> = Arc::from(effect_source);
                (
                    Some(layout),
                    Some(sampler),
                    HashMap::from([(source, pipeline)]),
                )
            } else {
                (None, None, HashMap::new())
            };
        Self {
            target_format,
            settings: settings.clone(),
            prepared_frame: None,
            cull_bind_group,
            shadow_cull_bind_group,
            cull_bind_group_layout,
            render_bind_group,
            render_bind_group_layout,
            render_pipeline_layout,
            scene_color_bind_group_layout,
            scene_color_sampler,
            default_scene_color_bind_group,
            default_scene_color_view,
            image_bind_group_layout,
            image_sampler,
            default_image_bind_group,
            image_cache: HashMap::new(),
            shadow_bind_group,
            shadow_bind_group_layout,
            shadow_sampler,
            shadow_filter_sampler,
            shadow_transmittance_sampler,
            shadow_alpha_bind_group,
            shadow_caster_bind_group,
            camera_buffer,
            shadow_camera_buffer,
            shadow_camera_bytes: Vec::new(),
            light_buffer,
            light_capacity,
            cull_pipeline,
            opaque_pipeline,
            transparent_pipeline,
            project_surface_pipelines: HashMap::new(),
            project_vertex_pipelines: None,
            shadow_pipeline: shadow_pipelines.opaque,
            shadow_alpha_pipeline: shadow_pipelines.alpha,
            shadow_mask_pipeline: shadow_pipelines.mask,
            shadow_depth_view,
            shadow_transmittance_view,
            shadow_resolution: SHADOW_MAP_SIZE,
            shadow_signature: None,
            effect_pipelines,
            effect_layout,
            effect_sampler,
            scene_target: None,
            mesh_cache: HashMap::new(),
            frame_meshes: HashMap::new(),
            mesh_cache_bytes: 0,
            frame_serial: 0,
            instance_buffer,
            instance_capacity,
            compact_buffer,
            compact_capacity,
            shadow_compact_buffer,
            shadow_compact_capacity: compact_capacity,
            indirect_buffer,
            indirect_capacity,
            shadow_indirect_buffer,
            shadow_indirect_capacity: indirect_capacity,
            instance_data: Vec::new(),
            instance_bytes: Vec::new(),
            camera_bytes: Vec::new(),
            stats_started: std::time::Instant::now(),
            stats_prepare_ms: 0.0,
            stats_prepares: 0,
            stats_mesh_hits: 0,
            stats_mesh_misses: 0,
            stats_batches: 0,
            stats_instances: 0,
            stats_shadow_renders: 0,
            stats_shadow_reuses: 0,
        }
    }

    fn prepare_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &SceneFrame,
    ) -> (usize, usize) {
        self.ensure_project_surface_pipelines(device, frame);
        let used_texture_sets = frame
            .draws
            .iter()
            .map(material_texture_key)
            .collect::<HashSet<_>>();
        self.image_cache
            .retain(|key, _| used_texture_sets.contains(key));
        for draw in &frame.draws {
            let key = material_texture_key(draw);
            self.image_cache.entry(key).or_insert_with(|| {
                let material = draw.imported_material.as_ref();
                let images = [
                    draw.image.as_deref(),
                    material
                        .and_then(|material| material.normal_texture.as_ref())
                        .map(|binding| binding.image.as_ref()),
                    material
                        .and_then(|material| material.metallic_roughness_texture.as_ref())
                        .map(|binding| binding.image.as_ref()),
                    material
                        .and_then(|material| material.occlusion_texture.as_ref())
                        .map(|binding| binding.image.as_ref()),
                    material
                        .and_then(|material| material.emissive_texture.as_ref())
                        .map(|binding| binding.image.as_ref()),
                ];
                let samplers = [
                    draw.image_sampler,
                    material
                        .and_then(|material| material.normal_texture.as_ref())
                        .map(|binding| binding.sampler),
                    material
                        .and_then(|material| material.metallic_roughness_texture.as_ref())
                        .map(|binding| binding.sampler),
                    material
                        .and_then(|material| material.occlusion_texture.as_ref())
                        .map(|binding| binding.sampler),
                    material
                        .and_then(|material| material.emissive_texture.as_ref())
                        .map(|binding| binding.sampler),
                ];
                let mut owned_textures = Vec::new();
                let mut views = Vec::with_capacity(5);
                let mut owned_samplers = Vec::with_capacity(5);
                for slot in 0..5 {
                    let image = images[slot].or(draw.image.as_deref());
                    let view = if let Some(image) = image {
                        let srgb = slot == 0 || slot == 4;
                        let mip_levels = imported_texture_mips(image, srgb, slot == 1);
                        let texture = device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("mogf-imported-material-slot"),
                            size: wgpu::Extent3d {
                                width: image.width,
                                height: image.height,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: mip_levels.len() as u32,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: if srgb {
                                wgpu::TextureFormat::Rgba8UnormSrgb
                            } else {
                                wgpu::TextureFormat::Rgba8Unorm
                            },
                            usage: wgpu::TextureUsages::TEXTURE_BINDING
                                | wgpu::TextureUsages::COPY_DST,
                            view_formats: &[],
                        });
                        for (mip_level, (width, height, bytes)) in mip_levels.iter().enumerate() {
                            queue.write_texture(
                                wgpu::TexelCopyTextureInfo {
                                    texture: &texture,
                                    mip_level: mip_level as u32,
                                    origin: wgpu::Origin3d::ZERO,
                                    aspect: wgpu::TextureAspect::All,
                                },
                                bytes,
                                wgpu::TexelCopyBufferLayout {
                                    offset: 0,
                                    bytes_per_row: Some(width * 4),
                                    rows_per_image: Some(*height),
                                },
                                wgpu::Extent3d {
                                    width: *width,
                                    height: *height,
                                    depth_or_array_layers: 1,
                                },
                            );
                        }
                        let view = texture.create_view(&Default::default());
                        owned_textures.push(texture);
                        view
                    } else {
                        // The default scene-color view is a valid neutral placeholder; shader
                        // flags prevent sampling slots that the source material does not bind.
                        self.default_scene_color_view.clone()
                    };
                    views.push(view);
                    owned_samplers.push(
                        samplers[slot].map(|sampler| create_imported_sampler(device, sampler)),
                    );
                }
                let entries = (0..5)
                    .flat_map(|slot| {
                        [
                            wgpu::BindGroupEntry {
                                binding: (slot * 2) as u32,
                                resource: wgpu::BindingResource::TextureView(&views[slot]),
                            },
                            wgpu::BindGroupEntry {
                                binding: (slot * 2 + 1) as u32,
                                resource: wgpu::BindingResource::Sampler(
                                    owned_samplers[slot].as_ref().unwrap_or(&self.image_sampler),
                                ),
                            },
                        ]
                    })
                    .collect::<Vec<_>>();
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mogf-imported-material-bindings"),
                    layout: &self.image_bind_group_layout,
                    entries: &entries,
                })
            });
        }
        self.frame_serial += 1;
        self.frame_meshes.clear();
        let mut cache_hits = 0;
        let mut cache_misses = 0;
        let used: HashSet<_> = frame.meshes.iter().map(|mesh| mesh.key.clone()).collect();
        for mesh in &frame.meshes {
            if !self.mesh_cache.contains_key(&mesh.key) {
                cache_misses += 1;
                let cache_bytes = mesh.key.cache_bytes();
                while self.mesh_cache_bytes + cache_bytes > MAX_CACHED_GEOMETRY_BYTES {
                    let Some(evict) = self
                        .mesh_cache
                        .iter()
                        .filter(|(key, _)| !used.contains(*key))
                        .min_by_key(|(_, mesh)| mesh.last_used)
                        .map(|(key, _)| key.clone())
                    else {
                        break;
                    };
                    if let Some(removed) = self.mesh_cache.remove(&evict) {
                        self.mesh_cache_bytes =
                            self.mesh_cache_bytes.saturating_sub(removed.cache_bytes);
                    }
                }
                let label = if self.mesh_cache_bytes + cache_bytes <= MAX_CACHED_GEOMETRY_BYTES {
                    "mogf-cached-local-mesh"
                } else {
                    "mogf-frame-local-mesh-over-cache-limit"
                };
                let buffer = mesh.key.create_buffer(device, label);
                let gpu_mesh = CachedMesh {
                    buffer,
                    cache_bytes,
                    last_used: self.frame_serial,
                };
                if self.mesh_cache_bytes + cache_bytes <= MAX_CACHED_GEOMETRY_BYTES {
                    self.mesh_cache_bytes += cache_bytes;
                    self.mesh_cache.insert(mesh.key.clone(), gpu_mesh);
                } else {
                    self.frame_meshes.insert(mesh.key.clone(), gpu_mesh);
                }
            } else if let Some(cached) = self.mesh_cache.get_mut(&mesh.key) {
                cache_hits += 1;
                cached.last_used = self.frame_serial;
            }
        }
        // Remove old revisions after 180 frames, while preserving meshes used now.
        let stale: Vec<_> = self
            .mesh_cache
            .iter()
            .filter(|(key, cached)| {
                !used.contains(*key) && self.frame_serial.saturating_sub(cached.last_used) > 180
            })
            .map(|(key, _)| key.clone())
            .collect();
        for key in stale {
            if let Some(removed) = self.mesh_cache.remove(&key) {
                self.mesh_cache_bytes = self.mesh_cache_bytes.saturating_sub(removed.cache_bytes);
            }
        }

        let camera_bytes = bytemuck::bytes_of(&frame.camera);
        if camera_bytes != self.camera_bytes {
            queue.write_buffer(&self.camera_buffer, 0, camera_bytes);
            self.camera_bytes.clear();
            self.camera_bytes.extend_from_slice(camera_bytes);
        }
        if let Some(shadow_camera) = &frame.shadow_camera {
            let shadow_camera_bytes = bytemuck::bytes_of(shadow_camera);
            if shadow_camera_bytes != self.shadow_camera_bytes {
                queue.write_buffer(&self.shadow_camera_buffer, 0, shadow_camera_bytes);
                self.shadow_camera_bytes.clear();
                self.shadow_camera_bytes
                    .extend_from_slice(shadow_camera_bytes);
            }
        }
        if !frame.lights.is_empty() {
            debug_assert!(frame.lights.len() <= self.light_capacity);
            queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&frame.lights));
        }
        self.instance_data.clear();
        self.instance_data
            .extend(frame.draws.iter().map(|draw| draw.gpu));
        let bytes = bytemuck::cast_slice(&self.instance_data);
        let mut recreate_bind_group = false;
        if bytes != self.instance_bytes {
            let required = (bytes.len() as u64).max(std::mem::size_of::<InstanceGpu>() as u64);
            if required > self.instance_capacity {
                self.instance_capacity = required.next_power_of_two();
                self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mogf-editor-instance-buffer-grown"),
                    size: self.instance_capacity,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                recreate_bind_group = true;
            }
            if !bytes.is_empty() {
                queue.write_buffer(&self.instance_buffer, 0, bytes);
            }
            self.instance_bytes.clear();
            self.instance_bytes.extend_from_slice(bytes);
        }
        let required_compact = (frame.draws.len() as u64)
            .max(1)
            .saturating_mul(std::mem::size_of::<InstanceGpu>() as u64);
        if required_compact > self.compact_capacity {
            self.compact_capacity = required_compact.next_power_of_two();
            self.compact_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mogf-editor-compacted-instances-grown"),
                size: self.compact_capacity,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            });
            recreate_bind_group = true;
        }
        if required_compact > self.shadow_compact_capacity {
            self.shadow_compact_capacity = required_compact.next_power_of_two();
            self.shadow_compact_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mogf-editor-shadow-compacted-instances-grown"),
                size: self.shadow_compact_capacity,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            });
            recreate_bind_group = true;
        }
        let required_indirect = (frame.batches.len() as u64)
            .max(1)
            .saturating_mul(std::mem::size_of::<DrawIndirectGpu>() as u64);
        if required_indirect > self.indirect_capacity {
            self.indirect_capacity = required_indirect.next_power_of_two();
            self.indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mogf-editor-indirect-draws-grown"),
                size: self.indirect_capacity,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            recreate_bind_group = true;
        }
        if required_indirect > self.shadow_indirect_capacity {
            self.shadow_indirect_capacity = required_indirect.next_power_of_two();
            self.shadow_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mogf-editor-shadow-indirect-draws-grown"),
                size: self.shadow_indirect_capacity,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            recreate_bind_group = true;
        }
        let indirect_args: Vec<_> = frame
            .batches
            .iter()
            .map(|batch| DrawIndirectGpu {
                vertex_count: u32::try_from(frame.meshes[batch.mesh].key.vertex_count())
                    .unwrap_or(u32::MAX),
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            })
            .collect();
        if !indirect_args.is_empty() {
            queue.write_buffer(
                &self.indirect_buffer,
                0,
                bytemuck::cast_slice(&indirect_args),
            );
            queue.write_buffer(
                &self.shadow_indirect_buffer,
                0,
                bytemuck::cast_slice(&indirect_args),
            );
        }
        if recreate_bind_group {
            self.cull_bind_group = Self::create_cull_bind_group(
                device,
                &self.cull_bind_group_layout,
                &self.camera_buffer,
                &self.indirect_buffer,
                &self.instance_buffer,
                &self.compact_buffer,
            );
            self.shadow_cull_bind_group = Self::create_cull_bind_group(
                device,
                &self.cull_bind_group_layout,
                &self.shadow_camera_buffer,
                &self.shadow_indirect_buffer,
                &self.instance_buffer,
                &self.shadow_compact_buffer,
            );
            self.render_bind_group = Self::create_render_bind_group(
                device,
                &self.render_bind_group_layout,
                &self.camera_buffer,
                &self.light_buffer,
                &self.compact_buffer,
            );
            self.shadow_caster_bind_group = Self::create_render_bind_group(
                device,
                &self.render_bind_group_layout,
                &self.shadow_camera_buffer,
                &self.light_buffer,
                &self.shadow_compact_buffer,
            );
        }
        (cache_hits, cache_misses)
    }

    fn ensure_project_surface_pipelines(&mut self, device: &wgpu::Device, frame: &SceneFrame) {
        let default_vertex = "fn mogf_vertex(input: VertexOut) -> VertexOut { return input; }";
        if let Some(source) = &frame.vertex_shader {
            let needs_rebuild = self
                .project_vertex_pipelines
                .as_ref()
                .map_or(true, |cached| cached.source.as_ref() != source.as_ref());
            if needs_rebuild {
                let surface = self.settings.surface_shader.as_deref().unwrap_or(
                    "fn mogf_surface(input: VertexOut) -> vec4<f32> { return mogf_default_lit_surface(input); }",
                );
                let shader_source = include_str!("editor_gpu.wgsl")
                    .replace(
                        "// MOGF_VERTEX_HOOK",
                        self.settings
                            .vertex_shader
                            .as_deref()
                            .unwrap_or(default_vertex),
                    )
                    .replace("// MOGF_GEOMETRY_HOOK", source.as_ref())
                    .replace("// MOGF_SURFACE_HOOK", surface)
                    .replace(
                        "// MOGF_SHADOW_OPACITY_HOOK",
                        shadow_opacity_hook(self.settings.surface_shader.as_deref()),
                    );
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("mogf-project-vertex-wgsl"),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                });
                let pipelines = create_surface_pipelines(
                    device,
                    self.target_format,
                    &self.settings,
                    &self.render_pipeline_layout,
                    &module,
                    "mogf-project-vertex",
                );
                let shadow_shader_source = include_str!("editor_gpu.wgsl")
                    .replace("// MOGF_VERTEX_HOOK", default_vertex)
                    .replace("// MOGF_GEOMETRY_HOOK", source.as_ref())
                    .replace("// MOGF_SURFACE_HOOK", surface)
                    .replace(
                        "// MOGF_SHADOW_OPACITY_HOOK",
                        shadow_opacity_hook(self.settings.surface_shader.as_deref()),
                    );
                let shadow_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("mogf-project-shadow-wgsl"),
                    source: wgpu::ShaderSource::Wgsl(shadow_shader_source.into()),
                });
                let shadow = create_shadow_pipeline(
                    device,
                    &shadow_module,
                    &self.render_bind_group_layout,
                    &self.render_pipeline_layout,
                );
                self.project_vertex_pipelines = Some(ProjectVertexPipelines {
                    source: Arc::clone(source),
                    pipelines,
                    shadow_opaque: shadow.opaque,
                    shadow_alpha: shadow.alpha,
                    shadow_mask: shadow.mask,
                });
            }
        }
        let used: HashSet<_> = frame
            .batches
            .iter()
            .filter_map(|batch| batch.shader_program.as_deref())
            .collect();
        for (id, source) in &frame.shader_sources {
            if !used.contains(id.as_str()) {
                continue;
            }
            if self
                .project_surface_pipelines
                .get(id)
                .is_some_and(|cached| {
                    cached.source.as_ref() == source.as_ref()
                        && cached.vertex_source.as_deref() == frame.vertex_shader.as_deref()
                })
            {
                continue;
            }
            let shader_source = include_str!("editor_gpu.wgsl")
                .replace(
                    "// MOGF_VERTEX_HOOK",
                    self.settings
                        .vertex_shader
                        .as_deref()
                        .unwrap_or(default_vertex),
                )
                .replace(
                    "// MOGF_GEOMETRY_HOOK",
                    frame
                        .vertex_shader
                        .as_deref()
                        .unwrap_or(DEFAULT_GEOMETRY_HOOK),
                )
                .replace("// MOGF_SURFACE_HOOK", source.as_ref())
                .replace(
                    "// MOGF_SHADOW_OPACITY_HOOK",
                    shadow_opacity_hook(Some(source.as_ref())),
                );
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mogf-project-material-wgsl"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
            let pipelines = create_surface_pipelines(
                device,
                self.target_format,
                &self.settings,
                &self.render_pipeline_layout,
                &module,
                &format!("mogf-project-material-{id}"),
            );
            let shadow_shader_source = include_str!("editor_gpu.wgsl")
                .replace("// MOGF_VERTEX_HOOK", default_vertex)
                .replace(
                    "// MOGF_GEOMETRY_HOOK",
                    frame
                        .vertex_shader
                        .as_deref()
                        .unwrap_or(DEFAULT_GEOMETRY_HOOK),
                )
                .replace("// MOGF_SURFACE_HOOK", source.as_ref())
                .replace(
                    "// MOGF_SHADOW_OPACITY_HOOK",
                    shadow_opacity_hook(Some(source.as_ref())),
                );
            let shadow_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mogf-project-material-shadow-wgsl"),
                source: wgpu::ShaderSource::Wgsl(shadow_shader_source.into()),
            });
            let shadow = create_shadow_pipeline(
                device,
                &shadow_module,
                &self.render_bind_group_layout,
                &self.render_pipeline_layout,
            );
            self.project_surface_pipelines.insert(
                id.clone(),
                ProjectSurfacePipelines {
                    source: Arc::clone(source),
                    vertex_source: frame.vertex_shader.clone(),
                    pipelines,
                    shadow_alpha: shadow.alpha,
                },
            );
        }
        self.project_surface_pipelines
            .retain(|id, _| used.contains(id.as_str()));
    }
}
