struct Camera {
    view_rows: array<vec4<f32>, 4>,
    projection: vec4<f32>, // fov_y or ortho_height, aspect, near, far
    flags: vec4<u32>,      // projection kind, target format is sRGB, instance count, light count
    shadow: vec4<u32>,     // light index, enabled, transmittance enabled, shadow-camera pass
};

@group(0) @binding(0) var<uniform> camera: Camera;
struct IndirectDraw {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(1) var<storage, read_write> indirect_draws: array<IndirectDraw>;

struct InstanceData {
    model_rows: array<vec4<f32>, 4>,
    color: vec4<f32>,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    batch: vec4<u32>,
    material: vec4<f32>, // ambient, diffuse, specular, shininess
    shader_params: vec4<f32>,
    imported_params: vec4<f32>,
    shadow_flags: vec4<u32>, // casts shadow, receives shadow
};

struct LightData {
    position: vec4<f32>,
    direction: vec4<f32>, // w: range, negative means unlimited
    color_intensity: vec4<f32>,
    kind: vec4<u32>, // 0 ambient, 1 directional, 2 point, 3 spot
    cone_cosines: vec4<f32>, // inner, outer
    shadow_params: vec4<f32>, // depth bias, PCF radius in texels
};

// Public surface-shader light ABI. Positions and directions use view space;
// directional direction points along the emitted rays.
struct MogfLight {
    position: vec3<f32>,
    direction: vec3<f32>,
    color: vec3<f32>,
    intensity: f32,
    range: f32,
    kind: u32,
    casts_shadow: u32,
    inner_cone_cos: f32,
    outer_cone_cos: f32,
    shadow_bias: f32,
    shadow_filter_radius: f32,
};

const MOGF_LIGHT_AMBIENT: u32 = 0u;
const MOGF_LIGHT_DIRECTIONAL: u32 = 1u;
const MOGF_LIGHT_POINT: u32 = 2u;
const MOGF_LIGHT_SPOT: u32 = 3u;

@group(0) @binding(1) var<storage, read> lights: array<LightData>;

@group(0) @binding(2) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(3) var<storage, read> compact_instances: array<InstanceData>;
@group(0) @binding(4) var<storage, read_write> compact_instances_write: array<InstanceData>;

@group(1) @binding(0) var<uniform> shadow_camera: Camera;
@group(1) @binding(1) var shadow_depth: texture_depth_2d;
@group(1) @binding(2) var shadow_sampler: sampler_comparison;
@group(1) @binding(3) var shadow_transmittance: texture_2d<f32>;
@group(1) @binding(4) var shadow_filter_sampler: sampler_comparison;
@group(1) @binding(5) var shadow_transmittance_sampler: sampler;

@group(2) @binding(0) var mogf_scene_color_buffer: texture_2d<f32>;
@group(2) @binding(1) var mogf_scene_color_sampler: sampler;
@group(3) @binding(0) var mogf_image: texture_2d<f32>;
@group(3) @binding(1) var mogf_image_sampler: sampler;
@group(3) @binding(2) var mogf_normal_image: texture_2d<f32>;
@group(3) @binding(3) var mogf_normal_sampler: sampler;
@group(3) @binding(4) var mogf_mr_image: texture_2d<f32>;
@group(3) @binding(5) var mogf_mr_sampler: sampler;
@group(3) @binding(6) var mogf_occlusion_image: texture_2d<f32>;
@group(3) @binding(7) var mogf_occlusion_sampler: sampler;
@group(3) @binding(8) var mogf_emissive_image: texture_2d<f32>;
@group(3) @binding(9) var mogf_emissive_sampler: sampler;

fn mogf_light_count() -> u32 {
    return camera.flags.w;
}

fn mogf_get_light(index: u32) -> MogfLight {
    var result: MogfLight;
    result.position = vec3<f32>(0.0);
    result.direction = vec3<f32>(0.0);
    result.color = vec3<f32>(0.0);
    result.intensity = 0.0;
    result.range = 0.0;
    result.kind = MOGF_LIGHT_AMBIENT;
    result.casts_shadow = 0u;
    result.inner_cone_cos = 1.0;
    result.outer_cone_cos = 1.0;
    result.shadow_bias = 0.0012;
    result.shadow_filter_radius = 0.5;
    if (index < camera.flags.w) {
        let light = lights[index];
        result.position = light.position.xyz;
        result.direction = light.direction.xyz;
        result.color = light.color_intensity.rgb;
        result.intensity = light.color_intensity.a;
        result.range = light.direction.w;
        result.kind = light.kind.x;
        result.casts_shadow = light.kind.y;
        result.inner_cone_cos = light.cone_cosines.x;
        result.outer_cone_cos = light.cone_cosines.y;
        result.shadow_bias = light.shadow_params.x;
        result.shadow_filter_radius = light.shadow_params.y;
    }
    return result;
}

fn camera_point(instance: InstanceData, local: vec3<f32>) -> vec3<f32> {
    let point = vec4<f32>(local, 1.0);
    let world = vec4<f32>(
        dot(instance.model_rows[0], point),
        dot(instance.model_rows[1], point),
        dot(instance.model_rows[2], point),
        dot(instance.model_rows[3], point),
    );
    return vec3<f32>(
        dot(camera.view_rows[0], world),
        dot(camera.view_rows[1], world),
        dot(camera.view_rows[2], world),
    );
}

@compute @workgroup_size(64)
fn cull_instances(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= camera.flags.z) {
        return;
    }
    let instance = instances[index];
    if (camera.shadow.w != 0u && instance.shadow_flags.x == 0u) {
        return;
    }
    var before_near = true;
    var beyond_far = true;
    var left = true;
    var right = true;
    var below = true;
    var above = true;
    let half_height = camera.projection.x * 0.5;
    let tan_y = tan(half_height);
    let tan_x = tan_y * camera.projection.y;
    for (var corner = 0u; corner < 8u; corner += 1u) {
        let local = vec3<f32>(
            select(instance.bounds_min.x, instance.bounds_max.x, (corner & 1u) != 0u),
            select(instance.bounds_min.y, instance.bounds_max.y, (corner & 2u) != 0u),
            select(instance.bounds_min.z, instance.bounds_max.z, (corner & 4u) != 0u),
        );
        let point = camera_point(instance, local);
        let depth = -point.z;
        let epsilon = max(max(abs(point.x), max(abs(point.y), abs(depth))), camera.projection.w)
            * 1.0e-6;
        before_near = before_near && depth < camera.projection.z - epsilon;
        beyond_far = beyond_far && depth > camera.projection.w + epsilon;
        if (camera.flags.x == 0u) {
            left = left && point.x < -depth * tan_x - epsilon;
            right = right && point.x > depth * tan_x + epsilon;
            below = below && point.y < -depth * tan_y - epsilon;
            above = above && point.y > depth * tan_y + epsilon;
        } else {
            let half_width = half_height * camera.projection.y;
            left = left && point.x < -half_width - epsilon;
            right = right && point.x > half_width + epsilon;
            below = below && point.y < -half_height - epsilon;
            above = above && point.y > half_height + epsilon;
        }
    }
    let outside = (instance.batch.w & 1u) == 0u
        && (before_near || beyond_far || left || right || below || above);
    if (!outside) {
        let batch_index = instance.batch.x;
        let slot = atomicAdd(&indirect_draws[batch_index].instance_count, 1u);
        compact_instances_write[instance.batch.y + slot] = instance;
    }
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) view_position: vec3<f32>,
    @location(2) material: vec4<f32>,
    @location(3) @interpolate(flat) shading_mode: u32,
    @location(4) @interpolate(flat) shader_params: vec4<f32>,
    @location(5) world_position: vec3<f32>,
    @location(6) @interpolate(flat) receives_shadow: u32,
    @location(7) image_uv: vec2<f32>,
    @location(8) @interpolate(flat) has_image: u32,
    @location(9) view_normal: vec3<f32>,
    @location(10) @interpolate(flat) imported_alpha_mode: u32,
    @location(11) normal_uv: vec2<f32>,
    @location(12) metallic_roughness_uv: vec2<f32>,
    @location(13) occlusion_uv: vec2<f32>,
    @location(14) emissive_uv: vec2<f32>,
    @location(15) @interpolate(flat) imported_params: vec4<f32>,
};

// Available only to project shaders declared with stage "surface_scene".
// The snapshot contains opaque scene color from this frame, before transparency.
fn mogf_scene_color(input: VertexOut, offset_uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(mogf_scene_color_buffer));
    let uv = input.position.xy / size + offset_uv;
    return textureSampleLevel(
        mogf_scene_color_buffer,
        mogf_scene_color_sampler,
        clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)),
        0.0,
    );
}

fn mogf_surface_normal(input: VertexOut) -> vec3<f32> {
    if (dot(input.view_normal, input.view_normal) > 1.0e-12) {
        return normalize(input.view_normal);
    }
    var normal = normalize(cross(dpdy(input.view_position), dpdx(input.view_position)));
    if (dot(normal, -input.view_position) < 0.0) {
        normal = -normal;
    }
    return normal;
}

fn mogf_imported_surface_normal(input: VertexOut, geometric_normal: vec3<f32>) -> vec3<f32> {
    let flags = bitcast<u32>(input.imported_params.w);
    if ((flags & 1u) == 0u) {
        return geometric_normal;
    }
    let uv_dx = dpdx(input.normal_uv);
    let uv_dy = dpdy(input.normal_uv);
    let determinant = uv_dx.x * uv_dy.y - uv_dx.y * uv_dy.x;
    if (abs(determinant) < 1.0e-8) {
        return geometric_normal;
    }
    let position_dx = dpdx(input.view_position);
    let position_dy = dpdy(input.view_position);
    let tangent = normalize((position_dx * uv_dy.y - position_dy * uv_dx.y) / determinant);
    let bitangent = normalize((-position_dx * uv_dy.x + position_dy * uv_dx.x) / determinant);
    let sampled = textureSample(mogf_normal_image, mogf_normal_sampler, input.normal_uv).xyz * 2.0 - vec3<f32>(1.0);
    let scaled = vec3<f32>(sampled.xy * input.imported_params.x, sampled.z);
    return normalize(tangent * scaled.x + bitangent * scaled.y + geometric_normal * scaled.z);
}

fn mogf_surface_base_color(input: VertexOut) -> vec4<f32> {
    if (input.has_image != 0u) {
        return input.color * textureSample(mogf_image, mogf_image_sampler, input.image_uv);
    }
    return input.color;
}

fn mogf_view_direction(input: VertexOut) -> vec3<f32> {
    return normalize(-input.view_position);
}

fn mogf_light_direction(input: VertexOut, light: MogfLight) -> vec3<f32> {
    if (light.kind == MOGF_LIGHT_POINT || light.kind == MOGF_LIGHT_SPOT) {
        return normalize(light.position - input.view_position);
    }
    if (light.kind == MOGF_LIGHT_DIRECTIONAL) {
        return normalize(-light.direction);
    }
    return vec3<f32>(0.0);
}

fn mogf_light_attenuation(input: VertexOut, light: MogfLight) -> f32 {
    if (light.kind != MOGF_LIGHT_POINT && light.kind != MOGF_LIGHT_SPOT) {
        return 1.0;
    }
    let distance = length(light.position - input.view_position);
    var attenuation = 1.0 / (1.0 + distance * distance * 0.02);
    if (light.range > 0.0) {
        if (distance >= light.range) {
            return 0.0;
        }
        let cutoff = 1.0 - distance / light.range;
        attenuation *= cutoff * cutoff;
    }
    if (light.kind == MOGF_LIGHT_SPOT) {
        let from_light = normalize(input.view_position - light.position);
        let cone_cos = dot(from_light, normalize(light.direction));
        attenuation *= smoothstep(light.outer_cone_cos, light.inner_cone_cos, cone_cos);
    }
    return attenuation;
}

// Project vertex assets provide a local-space position deformation.
// MOGF_GEOMETRY_HOOK

// Row-major affine matrices need the inverse transpose for authored normals.
// Cofactors avoid an inverse instruction and handle nonuniform/mirrored scale.
// Singular transforms and missing normals use the derivative surface fallback.
fn mogf_transform_normal(row0: vec3<f32>, row1: vec3<f32>, row2: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let cofactor0 = cross(row1, row2);
    let cofactor1 = cross(row2, row0);
    let cofactor2 = cross(row0, row1);
    let determinant = dot(row0, cofactor0);
    if (abs(determinant) < 1.0e-12 || dot(normal, normal) < 1.0e-12) {
        return vec3<f32>(0.0);
    }
    return normalize(vec3<f32>(dot(cofactor0, normal), dot(cofactor1, normal), dot(cofactor2, normal)) / determinant);
}

@vertex
fn vs_main(
    @location(0) position_and_shade: vec4<f32>,
    @location(1) model_row_0: vec4<f32>,
    @location(2) model_row_1: vec4<f32>,
    @location(3) model_row_2: vec4<f32>,
    @location(4) model_row_3: vec4<f32>,
    @location(5) color: vec4<f32>,
    @location(6) material: vec4<f32>,
    @location(7) batch: vec4<u32>,
    @location(8) source_color: vec4<f32>,
    @location(9) source_uv: vec2<f32>,
    @location(10) source_normal: vec3<f32>,
    @location(11) source_normal_uv: vec2<f32>,
    @location(12) source_metallic_roughness_uv: vec2<f32>,
    @location(13) source_occlusion_uv: vec2<f32>,
    @location(14) source_emissive_uv: vec2<f32>,
    @builtin(instance_index) instance_index: u32,
) -> VertexOut {
    // The instance vertex buffer is sliced per batch, so instance_index is
    // batch-local. The storage buffer is not sliced and needs the batch base.
    let instance_data = compact_instances[batch.y + instance_index];
    let shader_params = instance_data.shader_params;
    let imported_params = instance_data.imported_params;
    let local = vec4<f32>(mogf_deform(position_and_shade.xyz, shader_params), 1.0);
    var out: VertexOut;
    let world = vec4<f32>(
        dot(model_row_0, local), dot(model_row_1, local),
        dot(model_row_2, local), dot(model_row_3, local),
    );
    let view = vec4<f32>(
        dot(camera.view_rows[0], world), dot(camera.view_rows[1], world),
        dot(camera.view_rows[2], world), dot(camera.view_rows[3], world),
    );
    let depth = -view.z;
    var clip: vec4<f32>;
    if (camera.flags.x == 0u) {
        let tangent = tan(camera.projection.x * 0.5);
        let near = camera.projection.z;
        let far = camera.projection.w;
        clip = vec4<f32>(
            view.x / (tangent * camera.projection.y),
            view.y / tangent,
            (far * depth - near * far) / (far - near),
            depth,
        );
    } else {
        let half_height = camera.projection.x * 0.5;
        let near = camera.projection.z;
        let far = camera.projection.w;
        clip = vec4<f32>(
            view.x / (half_height * camera.projection.y),
            view.y / half_height,
            (depth - near) / (far - near),
            1.0,
        );
    }
    out.position = clip;
    out.color = vec4<f32>(color.rgb * source_color.rgb * position_and_shade.w, color.a * source_color.a);
    out.view_position = view.xyz;
    let world_normal = mogf_transform_normal(model_row_0.xyz, model_row_1.xyz, model_row_2.xyz, source_normal);
    out.view_normal = mogf_transform_normal(camera.view_rows[0].xyz, camera.view_rows[1].xyz, camera.view_rows[2].xyz, world_normal);
    // Project geometry shaders can deform positions without providing the
    // corresponding normal map. Retain geometric normals for those draws.
    if ((batch.w & 1u) != 0u) {
        out.view_normal = vec3<f32>(0.0);
    }
    out.material = material;
    out.shading_mode = batch.z;
    out.shader_params = shader_params;
    out.world_position = world.xyz;
    out.receives_shadow = (batch.w >> 2u) & 1u;
    out.imported_alpha_mode = (batch.w >> 5u) & 3u;
    out.normal_uv = source_normal_uv;
    out.metallic_roughness_uv = source_metallic_roughness_uv;
    out.occlusion_uv = source_occlusion_uv;
    out.emissive_uv = source_emissive_uv;
    out.imported_params = imported_params;
    out.image_uv = source_uv;
    out.has_image = (batch.w >> 3u) & 1u;
    if (out.has_image != 0u) {
        out.image_uv = mix(shader_params.xy, shader_params.zw, out.image_uv);
    }
    if (camera.shadow.w != 0u && (batch.w & 2u) == 0u) {
        out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    return mogf_vertex(out);
}

// Custom files provide: fn mogf_vertex(input: VertexOut) -> VertexOut.
// MOGF_VERTEX_HOOK

fn linear_to_srgb(channel: f32) -> f32 {
    if (channel <= 0.0031308) {
        return channel * 12.92;
    }
    return 1.055 * pow(max(channel, 0.0), 1.0 / 2.4) - 0.055;
}

fn mogf_fresnel_schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    let grazing = 1.0 - clamp(cos_theta, 0.0, 1.0);
    let grazing2 = grazing * grazing;
    let grazing5 = grazing2 * grazing2 * grazing;
    return f0 + (vec3<f32>(1.0) - f0) * grazing5;
}

fn mogf_ggx_distribution(alpha: f32, no_h: f32) -> f32 {
    let alpha2 = alpha * alpha;
    let denominator = no_h * no_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / max(3.14159265 * denominator * denominator, 1.0e-6);
}

fn mogf_smith_visibility(alpha: f32, no_v: f32, no_l: f32) -> f32 {
    let alpha2 = alpha * alpha;
    let gv = 2.0 * no_v / max(no_v + sqrt(alpha2 + (1.0 - alpha2) * no_v * no_v), 1.0e-5);
    let gl = 2.0 * no_l / max(no_l + sqrt(alpha2 + (1.0 - alpha2) * no_l * no_l), 1.0e-5);
    return gv * gl;
}

fn mogf_shadow_tap(uv: vec2<f32>, reference_depth: f32, filtered_sample: bool) -> f32 {
    var visibility: f32;
    if (filtered_sample) {
        visibility = textureSampleCompare(
            shadow_depth, shadow_filter_sampler, uv, reference_depth,
        );
    } else {
        visibility = textureSampleCompare(shadow_depth, shadow_sampler, uv, reference_depth);
    }
    if (camera.shadow.z != 0u) {
        if (filtered_sample) {
            visibility *= textureSampleLevel(
                shadow_transmittance, shadow_transmittance_sampler, uv, 0.0,
            ).r;
        } else {
            let dimensions = textureDimensions(shadow_transmittance);
            let pixel = clamp(
                vec2<i32>(uv * vec2<f32>(dimensions)),
                vec2<i32>(0),
                vec2<i32>(dimensions) - vec2<i32>(1),
            );
            visibility *= textureLoad(shadow_transmittance, pixel, 0).r;
        }
    }
    return visibility;
}

fn mogf_shadow_visibility(world_position: vec3<f32>) -> f32 {
    if (camera.shadow.y == 0u) {
        return 1.0;
    }
    let point = vec4<f32>(world_position, 1.0);
    let view = vec3<f32>(
        dot(shadow_camera.view_rows[0], point),
        dot(shadow_camera.view_rows[1], point),
        dot(shadow_camera.view_rows[2], point),
    );
    let depth = -view.z;
    if (depth < shadow_camera.projection.z || depth > shadow_camera.projection.w) {
        return 1.0;
    }
    var clip: vec3<f32>;
    if (shadow_camera.flags.x == 0u) {
        let tangent = tan(shadow_camera.projection.x * 0.5);
        clip = vec3<f32>(
            view.x / (depth * tangent * shadow_camera.projection.y),
            view.y / (depth * tangent),
            (depth - shadow_camera.projection.z)
                / (shadow_camera.projection.w - shadow_camera.projection.z),
        );
    } else {
        let half_height = shadow_camera.projection.x * 0.5;
        clip = vec3<f32>(
            view.x / (half_height * shadow_camera.projection.y),
            view.y / half_height,
            (depth - shadow_camera.projection.z)
                / (shadow_camera.projection.w - shadow_camera.projection.z),
        );
    }
    let uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
    if (clip.z < 0.0 || clip.z > 1.0 || any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return 1.0;
    }
    let texel = 1.0 / vec2<f32>(textureDimensions(shadow_depth));
    let light = mogf_get_light(camera.shadow.x);
    let reference_depth = clip.z - light.shadow_bias;
    var visibility: f32;
    if (light.shadow_filter_radius <= 0.0) {
        visibility = mogf_shadow_tap(uv, reference_depth, false);
    } else {
        let offset = texel * light.shadow_filter_radius;
        visibility = 0.0;
        visibility += mogf_shadow_tap(
            uv + offset * vec2<f32>(-1.0, -1.0), reference_depth, true,
        );
        visibility += mogf_shadow_tap(
            uv + offset * vec2<f32>(1.0, -1.0), reference_depth, true,
        );
        visibility += mogf_shadow_tap(
            uv + offset * vec2<f32>(-1.0, 1.0), reference_depth, true,
        );
        visibility += mogf_shadow_tap(
            uv + offset * vec2<f32>(1.0, 1.0), reference_depth, true,
        );
        visibility *= 0.25;
    }
    return visibility;
}

fn mogf_surface_shadow_visibility(input: VertexOut) -> f32 {
    if (input.receives_shadow == 0u) {
        return 1.0;
    }
    return mogf_shadow_visibility(input.world_position);
}

fn mogf_principled_surface(input: VertexOut, normal: vec3<f32>, view_direction: vec3<f32>) -> vec4<f32> {
    let base_sample = mogf_surface_base_color(input);
    let base_color = base_sample.rgb;
    let imported_flags = bitcast<u32>(input.imported_params.w);
    let mr_sample = textureSample(mogf_mr_image, mogf_mr_sampler, input.metallic_roughness_uv);
    let metallic = clamp(input.material.x * select(1.0, mr_sample.b, (imported_flags & 2u) != 0u), 0.0, 1.0);
    let roughness = clamp(input.material.y * select(1.0, mr_sample.g, (imported_flags & 2u) != 0u), 0.045, 1.0);
    let reflectance = clamp(input.material.z, 0.0, 1.0);
    let clear_coat = clamp(input.material.w, 0.0, 1.0);
    let alpha = roughness * roughness;
    let dielectric_f0 = vec3<f32>(0.16 * reflectance * reflectance);
    let f0 = mix(dielectric_f0, base_color, metallic);
    let no_v = max(dot(normal, view_direction), 1.0e-4);
    var ambient_light = vec3<f32>(0.0);
    var direct = vec3<f32>(0.0);
    for (var index = 0u; index < camera.flags.w; index += 1u) {
        let light = mogf_get_light(index);
        if (light.kind == MOGF_LIGHT_AMBIENT) {
            ambient_light += light.color * light.intensity;
            continue;
        }
        let to_light = mogf_light_direction(input, light);
        let no_l = max(dot(normal, to_light), 0.0);
        if (no_l <= 0.0) {
            continue;
        }
        let radiance = light.color * light.intensity;
        let attenuation = mogf_light_attenuation(input, light);
        let half_vector = normalize(to_light + view_direction);
        let no_h = max(dot(normal, half_vector), 0.0);
        let vo_h = max(dot(view_direction, half_vector), 0.0);
        let fresnel = mogf_fresnel_schlick(f0, vo_h);
        let distribution = mogf_ggx_distribution(alpha, no_h);
        let visibility = mogf_smith_visibility(alpha, no_v, no_l);
        let specular = distribution * visibility * fresnel / max(4.0 * no_v * no_l, 1.0e-5);
        let diffuse = (vec3<f32>(1.0) - fresnel) * (1.0 - metallic) * base_color / 3.14159265;

        var layered_reflection = diffuse + specular;
        if (clear_coat > 0.0) {
            let coat_roughness = 0.25;
            let coat_alpha = coat_roughness * coat_roughness;
            let grazing = 1.0 - vo_h;
            let grazing2 = grazing * grazing;
            let coat_fresnel = 0.04 + 0.96 * grazing2 * grazing2 * grazing;
            let coat_specular = clear_coat * mogf_ggx_distribution(coat_alpha, no_h)
                * mogf_smith_visibility(coat_alpha, no_v, no_l) * coat_fresnel
                / max(4.0 * no_v * no_l, 1.0e-5);
            let coat_attenuation = 1.0 - clear_coat * coat_fresnel;
            layered_reflection = layered_reflection * coat_attenuation + coat_specular;
        }
        var shadow = 1.0;
        if (index == camera.shadow.x && light.casts_shadow != 0u) {
            shadow = mogf_surface_shadow_visibility(input);
        }
        direct += radiance * attenuation * no_l * layered_reflection * shadow;
    }
    if (camera.flags.w == 0u) {
        ambient_light = vec3<f32>(0.16);
    }
    let occlusion_sample = textureSample(mogf_occlusion_image, mogf_occlusion_sampler, input.occlusion_uv).r;
    let occlusion = select(1.0, mix(1.0, occlusion_sample, input.imported_params.y), (imported_flags & 4u) != 0u);
    let emissive_sample = textureSample(mogf_emissive_image, mogf_emissive_sampler, input.emissive_uv).rgb;
    let emissive_texture = select(vec3<f32>(1.0), emissive_sample, (imported_flags & 8u) != 0u);
    let emissive_scale = select(1.0, input.imported_params.z, (imported_flags & 16u) != 0u);
    let emissive = input.shader_params.rgb * emissive_texture * emissive_scale;
    let ambient = ambient_light * base_color * (1.0 - metallic)
        * (1.0 - 0.04 * clear_coat) * occlusion;
    return vec4<f32>(ambient + direct + emissive, base_sample.a);
}

fn mogf_default_lit_surface(input: VertexOut) -> vec4<f32> {
    if (input.shading_mode == 0u) {
        return mogf_surface_base_color(input);
    }
    let view_direction = mogf_view_direction(input);
    let normal = mogf_imported_surface_normal(input, mogf_surface_normal(input));
    if (input.shading_mode == 2u) {
        return mogf_principled_surface(input, normal, view_direction);
    }
    var ambient_light = vec3<f32>(0.0);
    var direct_light = vec3<f32>(0.0);
    var specular_light = vec3<f32>(0.0);
    for (var index = 0u; index < camera.flags.w; index += 1u) {
        let light = mogf_get_light(index);
        if (light.kind == MOGF_LIGHT_AMBIENT) {
            ambient_light += light.color * light.intensity;
            continue;
        }
        let to_light = mogf_light_direction(input, light);
        let diffuse = max(dot(normal, to_light), 0.0);
        if (diffuse <= 0.0) {
            continue;
        }
        let radiance = light.color * light.intensity;
        let attenuation = mogf_light_attenuation(input, light);
        var shadow = 1.0;
        if (index == camera.shadow.x && light.casts_shadow != 0u) {
            shadow = mogf_surface_shadow_visibility(input);
        }
        direct_light += radiance * attenuation * diffuse * shadow;
        let halfway = normalize(to_light + view_direction);
        specular_light += radiance * attenuation
            * pow(max(dot(normal, halfway), 0.0), input.material.w) * shadow;
    }
    if (camera.flags.w == 0u) {
        ambient_light = vec3<f32>(0.16);
    }
    let base_sample = mogf_surface_base_color(input);
    let rgb = base_sample.rgb * (ambient_light * input.material.x + direct_light * input.material.y)
        + specular_light * input.material.z;
    return vec4<f32>(rgb, base_sample.a);
}

// Custom files provide: fn mogf_surface(input: VertexOut) -> vec4<f32>.
// MOGF_SURFACE_HOOK
// Optional custom files can provide: fn mogf_shadow_opacity(input: VertexOut) -> f32.
// MOGF_SHADOW_OPACITY_HOOK

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    var surface = mogf_surface(input);
    if (input.imported_alpha_mode == 2u) {
        if (surface.a < input.shader_params.w) { discard; }
        surface.a = 1.0;
    } else if (input.imported_alpha_mode == 1u) {
        // glTF OPAQUE ignores alpha from the base-color factor and texture.
        surface.a = 1.0;
    }
    if (input.has_image != 0u && surface.a < 0.001) { discard; }
    var rgb = surface.rgb;
    if (camera.flags.y == 0u) {
        rgb = vec3<f32>(linear_to_srgb(rgb.r), linear_to_srgb(rgb.g), linear_to_srgb(rgb.b));
    }
    return vec4<f32>(rgb, surface.a);
}

@fragment
fn fs_shadow_alpha(input: VertexOut) -> @location(0) vec4<f32> {
    var alpha = clamp(mogf_shadow_opacity(input), 0.0, 1.0);
    if (input.has_image != 0u) {
        alpha *= textureSample(mogf_image, mogf_image_sampler, input.image_uv).a;
    }
    return vec4<f32>(1.0, 1.0, 1.0, alpha);
}

@fragment
fn fs_shadow_mask(input: VertexOut) {
    if (mogf_surface_base_color(input).a < input.shader_params.w) { discard; }
}
