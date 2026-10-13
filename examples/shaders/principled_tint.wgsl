fn mogf_surface(input: VertexOut) -> vec4<f32> {
    let lit = mogf_default_lit_surface(input);
    let tint_amount = clamp(input.shader_params.x, 0.0, 1.0);
    let tint = input.shader_params.yzw;
    return vec4<f32>(mix(lit.rgb, lit.rgb * tint, tint_amount), lit.a);
}
