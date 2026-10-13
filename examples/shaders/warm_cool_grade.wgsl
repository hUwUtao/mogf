// A restrained vertical color grade for the ordered camera-effect example.
fn mogf_effect(uv: vec2<f32>, color: vec4<f32>, resolution: vec2<f32>) -> vec4<f32> {
    let vertical = smoothstep(0.0, 1.0, uv.y);
    let cool = vec3<f32>(0.92, 0.98, 1.08);
    let warm = vec3<f32>(1.08, 0.96, 0.86);
    let grade = mix(cool, warm, vertical);
    return vec4<f32>(color.rgb * grade, color.a);
}
