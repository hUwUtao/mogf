struct EffectOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var mogf_scene: texture_2d<f32>;
@group(0) @binding(1) var mogf_sampler: sampler;
@group(0) @binding(2) var<uniform> mogf_resolution: vec2<f32>;

@vertex
fn mogf_effect_vertex(@builtin(vertex_index) index: u32) -> EffectOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let position = positions[index];
    var output: EffectOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.uv = vec2<f32>(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
    return output;
}

// Custom files provide: fn mogf_effect(uv: vec2<f32>, color: vec4<f32>, resolution: vec2<f32>) -> vec4<f32>.
// MOGF_EFFECT_HOOK

@fragment
fn mogf_effect_fragment(input: EffectOutput) -> @location(0) vec4<f32> {
    let color = textureSample(mogf_scene, mogf_sampler, input.uv);
    return mogf_effect(input.uv, color, mogf_resolution);
}
