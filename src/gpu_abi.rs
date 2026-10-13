use std::mem::{align_of, offset_of, size_of};

use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct InstanceGpu {
    pub(super) rows: [[f32; 4]; 4],
    pub(super) color: [f32; 4],
    pub(super) bounds_min: [f32; 4],
    pub(super) bounds_max: [f32; 4],
    pub(super) batch: [u32; 4],
    pub(super) material: [f32; 4],
    pub(super) shader_params: [f32; 4],
    pub(super) imported_params: [f32; 4],
    pub(super) shadow_flags: [u32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct LightGpu {
    pub(super) position: [f32; 4],
    pub(super) direction: [f32; 4],
    pub(super) color_intensity: [f32; 4],
    pub(super) kind: [u32; 4],
    pub(super) cone_cosines: [f32; 4],
    pub(super) shadow_params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct DrawIndirectGpu {
    pub(super) vertex_count: u32,
    pub(super) instance_count: u32,
    pub(super) first_vertex: u32,
    pub(super) first_instance: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct CameraGpu {
    pub(super) view_rows: [[f32; 4]; 4],
    pub(super) projection: [f32; 4],
    pub(super) flags: [u32; 4],
    pub(super) shadow: [u32; 4],
}

#[derive(Deserialize)]
struct AbiSchema {
    abi: String,
    version: u32,
    records: Vec<RecordSchema>,
    resources: ResourceSets,
}

#[derive(Deserialize)]
struct ResourceSets {
    surface: Vec<ResourceSchema>,
}

#[derive(Deserialize)]
struct ResourceSchema {
    group: u32,
    binding: u32,
    name: String,
}

#[derive(Deserialize)]
struct RecordSchema {
    host: String,
    wgsl: String,
    size: usize,
    alignment: usize,
    fields: Vec<FieldSchema>,
}

#[derive(Deserialize)]
struct FieldSchema {
    host: String,
    wgsl: String,
    offset: usize,
}

const SCHEMA: &str = include_str!("../assets/gpu-shader-abi/abi-v1.json");

pub(super) fn validate_surface_abi(wgsl: &str) -> Result<(), String> {
    let schema: AbiSchema = serde_json::from_str(SCHEMA)
        .map_err(|error| format!("invalid embedded GPU ABI schema: {error}"))?;
    if schema.abi != "mogf.surface" || schema.version != 1 {
        return Err("unsupported embedded GPU ABI identity or version".to_owned());
    }

    let module = naga::front::wgsl::parse_str(wgsl).map_err(|error| {
        format!(
            "cannot inspect GPU ABI WGSL: {}",
            error.emit_to_string(wgsl)
        )
    })?;
    for record in schema.records {
        let (host_size, host_alignment, host_offset) = host_layout(&record.host)?;
        if host_size != record.size || host_alignment != record.alignment {
            return Err(format!(
                "GPU ABI {} host layout is size {host_size}, alignment {host_alignment}; schema requires size {}, alignment {}",
                record.host, record.size, record.alignment
            ));
        }
        for field in &record.fields {
            let offset = host_offset(&field.host)?;
            if offset != field.offset {
                return Err(format!(
                    "GPU ABI {}.{} host offset is {offset}, schema requires {}",
                    record.host, field.host, field.offset
                ));
            }
        }

        let (_, ty) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some(record.wgsl.as_str()))
            .ok_or_else(|| format!("GPU ABI WGSL struct {} is missing", record.wgsl))?;
        let naga::TypeInner::Struct { members, span } = &ty.inner else {
            return Err(format!("GPU ABI WGSL type {} is not a struct", record.wgsl));
        };
        if *span as usize != record.size || members.len() != record.fields.len() {
            return Err(format!(
                "GPU ABI WGSL {} has size {} and {} fields; schema requires {} and {}",
                record.wgsl,
                span,
                members.len(),
                record.size,
                record.fields.len()
            ));
        }
        for field in &record.fields {
            let member = members
                .iter()
                .find(|member| member.name.as_deref() == Some(field.wgsl.as_str()))
                .ok_or_else(|| {
                    format!(
                        "GPU ABI WGSL field {}.{} is missing",
                        record.wgsl, field.wgsl
                    )
                })?;
            if member.offset as usize != field.offset {
                return Err(format!(
                    "GPU ABI WGSL {}.{} offset is {}, schema requires {}",
                    record.wgsl, field.wgsl, member.offset, field.offset
                ));
            }
        }
    }
    for expected in schema.resources.surface {
        let found = module.global_variables.iter().any(|(_, variable)| {
            variable.name.as_deref() == Some(expected.name.as_str())
                && variable.binding.as_ref().is_some_and(|binding| {
                    binding.group == expected.group && binding.binding == expected.binding
                })
        });
        if !found {
            return Err(format!(
                "GPU ABI resource {} expected at group {} binding {} is missing or moved",
                expected.name, expected.group, expected.binding
            ));
        }
    }
    Ok(())
}

fn host_layout(
    name: &str,
) -> Result<(usize, usize, Box<dyn Fn(&str) -> Result<usize, String>>), String> {
    macro_rules! layout {
        ($ty:ty, $($field:ident),+ $(,)?) => {
            (size_of::<$ty>(), align_of::<$ty>(), Box::new(|field: &str| match field {
                $(stringify!($field) => Ok(offset_of!($ty, $field)),)+
                _ => Err(format!("unknown GPU ABI host field {field}")),
            }))
        };
    }
    let result: (usize, usize, Box<dyn Fn(&str) -> Result<usize, String>>) = match name {
        "CameraGpu" => layout!(CameraGpu, view_rows, projection, flags, shadow),
        "InstanceGpu" => layout!(
            InstanceGpu,
            rows,
            color,
            bounds_min,
            bounds_max,
            batch,
            material,
            shader_params,
            imported_params,
            shadow_flags
        ),
        "LightGpu" => layout!(
            LightGpu,
            position,
            direction,
            color_intensity,
            kind,
            cone_cosines,
            shadow_params
        ),
        "DrawIndirectGpu" => layout!(
            DrawIndirectGpu,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance
        ),
        _ => return Err(format!("unknown GPU ABI host record {name}")),
    };
    Ok(result)
}
