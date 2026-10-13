// Stylized, mobile-friendly liquid glass. This tints the surface and adds a
// Fresnel rim and a bounded scene-color refraction offset.
// shader_params = [ripple_frequency, ripple_strength, opacity, fresnel_mix]
fn mogf_surface(input: VertexOut) -> vec4<f32> {
    if (input.shading_mode == 0u) {
        return input.color;
    }

    let frequency = max(input.shader_params.x, 0.0);
    let ripple_strength = clamp(input.shader_params.y, 0.0, 0.5);
    let phase = (input.world_position.x + input.world_position.y * 0.63) * frequency;
    let base_normal = mogf_surface_normal(input);
    let normal = normalize(base_normal + vec3<f32>(
        cos(phase) * ripple_strength,
        sin(phase * 0.73) * ripple_strength,
        0.0,
    ));
    let view = mogf_view_direction(input);
    let fresnel = pow(1.0 - clamp(dot(normal, view), 0.0, 1.0), 5.0);
    let fresnel_mix = clamp(input.shader_params.w, 0.0, 1.0);
    let tint = input.color.rgb;
    let refraction_offset = normal.xy * (0.004 + ripple_strength * 0.025);
    let refracted_scene = mogf_scene_color(input, refraction_offset);
    var ambient = vec3<f32>(0.025);
    var direct = vec3<f32>(0.0);

    for (var index = 0u; index < mogf_light_count(); index += 1u) {
        let light = mogf_get_light(index);
        let radiance = light.color * light.intensity;
        if (light.kind == MOGF_LIGHT_AMBIENT) {
            ambient += radiance * 0.2;
            continue;
        }

        let to_light = mogf_light_direction(input, light);
        let attenuation = mogf_light_attenuation(input, light);
        let diffuse = max(dot(normal, to_light), 0.0);
        let halfway = normalize(to_light + view);
        let specular_power = mix(96.0, 20.0, clamp(input.material.y, 0.0, 1.0));
        let specular = pow(max(dot(normal, halfway), 0.0), specular_power);
        var shadow = 1.0;
        if (light.casts_shadow != 0u && input.receives_shadow != 0u) {
            shadow = mogf_surface_shadow_visibility(input);
        }
        direct += radiance * attenuation * shadow
            * (tint * diffuse * 0.22 + vec3<f32>(specular * (0.12 + 0.38 * fresnel)));
    }

    let reflected_tint = mix(tint, vec3<f32>(0.7, 0.96, 1.0), fresnel_mix * (0.25 + 0.75 * fresnel));
    let refraction_weight = (1.0 - fresnel) * (0.35 + 0.4 * fresnel_mix);
    let rgb = mix(refracted_scene.rgb * tint + ambient * tint * 0.35, reflected_tint * (0.2 + fresnel * 0.6) + direct, fresnel);
    let final_rgb = mix(rgb, refracted_scene.rgb * tint + direct + ambient * tint * 0.35, refraction_weight * 0.35);
    let opacity = clamp(input.shader_params.z, 0.0, 1.0);
    return vec4<f32>(final_rgb, input.color.a * opacity);
}

// The shadow pass only needs opacity, so avoid rerunning refraction and lighting.
fn mogf_shadow_opacity(input: VertexOut) -> f32 {
    if (input.shading_mode == 0u) {
        return input.color.a;
    }
    return input.color.a * clamp(input.shader_params.z, 0.0, 1.0);
}
