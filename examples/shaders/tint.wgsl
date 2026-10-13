fn mogf_surface(input: VertexOut) -> vec4<f32> {
    let color = input.color;
    return vec4<f32>(color.r * 0.85, color.g, min(color.b * 1.15, 1.0), color.a);
}
