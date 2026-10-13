// A compact custom surface response using the public MOGF light and shadow ABI.
fn mogf_surface(input: VertexOut) -> vec4<f32> {
    if (input.shading_mode == 0u) {
        return input.color;
    }

    let normal = mogf_surface_normal(input);
    var ambient = vec3<f32>(0.1);
    var direct = vec3<f32>(0.0);
    for (var index = 0u; index < mogf_light_count(); index += 1u) {
        let light = mogf_get_light(index);
        let radiance = light.color * light.intensity;
        if (light.kind == MOGF_LIGHT_AMBIENT) {
            ambient += radiance;
            continue;
        }

        let to_light = mogf_light_direction(input, light);
        let diffuse = max(dot(normal, to_light), 0.0);
        if (diffuse <= 0.0) {
            continue;
        }
        var shadow = 1.0;
        if (light.casts_shadow != 0u && input.receives_shadow != 0u) {
            shadow = mogf_surface_shadow_visibility(input);
        }
        direct += radiance * mogf_light_attenuation(input, light) * diffuse * shadow;
    }

    let color = input.color.rgb * (ambient + direct * (1.0 - input.material.x));
    return vec4<f32>(color, input.color.a);
}
