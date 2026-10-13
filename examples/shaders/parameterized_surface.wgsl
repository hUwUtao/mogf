fn mogf_surface(input: VertexOut) -> vec4<f32> {
    // shader_params = [tint_mix, tint_r, tint_g, tint_b]
    let tint_mix = clamp(input.shader_params.x, 0.0, 1.0);
    let tint = input.shader_params.yzw;
    return vec4<f32>(mix(input.color.rgb, tint, tint_mix), input.color.a);
}
