fn mogf_effect(uv: vec2<f32>, color: vec4<f32>, resolution: vec2<f32>) -> vec4<f32> {
    let block_size = 4.0;
    let pixel = vec2<i32>(floor(uv * resolution / block_size) * block_size);
    return textureLoad(mogf_scene, pixel, 0);
}
