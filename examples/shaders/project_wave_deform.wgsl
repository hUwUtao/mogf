fn mogf_deform(position: vec3<f32>, shader_params: vec4<f32>) -> vec3<f32> {
    var deformed = position;
    let amplitude = 0.018 + 0.03 * clamp(shader_params.x, 0.0, 1.0);
    deformed.z += sin(position.x * 7.0 + position.y * 4.0) * amplitude;
    return deformed;
}
