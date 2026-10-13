# WGSL renderer hooks

The interactive `edit` and `live` renderers accept user-owned WGSL programs from project assets and launch-time files. Each module is combined with the renderer ABI and parsed and validated before its pipeline is created.

```sh
mogf-mvp live examples/principled_plastic.json --composition-id main \
  --shader examples/shaders/parameterized_surface.wgsl \
  --vertex-shader examples/shaders/screen_wave_vertex.wgsl \
  --effect-shader examples/shaders/pixelate_camera.wgsl
```

`--shader` supplies `mogf_surface(input: VertexOut) -> vec4<f32>` as the session-wide fallback. A project can instead assign a WGSL surface program to an individual material. Declare a shader asset, bind it in the `shaders` table, then name that shader from the material's `params.shader`:

```json
"assets": {
  "surface.blue": { "uri": "shaders/blue.wgsl", "kind": "shader", "digest": null }
},
"shaders": {
  "blue.surface": { "asset": "surface.blue", "stage": "surface", "abi_version": 1 }
},
"materials": {
  "blue": {
    "operation": "mogf.material.principled", "version": 1, "inputs": {},
    "params": {
      "color": [0.08, 0.28, 0.9, 1], "roughness": 0.32,
      "reflectance": 0.5, "shader": "blue.surface",
      "shader_params": [0.25, 0.05, 1.0, 0.3]
    }
  }
}
```

The WGSL file provides `mogf_surface(input: VertexOut) -> vec4<f32>`. It can call `mogf_default_lit_surface(input)` to retain built-in lighting, then apply its own response. `VertexOut` exposes `color`, `view_position`, the built-in `material` tuple, flat `shading_mode`, material-owned `shader_params`, and flat `receives_shadow`. Materials independently set `casts_shadow` and `receives_shadow`, both defaulting to true. The renderer caches each material pipeline by shader ID and source, and opaque instances only batch together when both mesh and program match. A custom surface can optionally provide `mogf_shadow_opacity(input: VertexOut) -> f32`; the shadow pass calls this instead of evaluating the full surface response. Without it, custom shaders preserve compatibility by taking alpha from `mogf_surface(input)`, while built-in materials use vertex/material alpha directly.

For a fully custom light response, WGSL programs can use `mogf_light_count()`, `mogf_get_light(index)`, `mogf_surface_normal(input)`, `mogf_view_direction(input)`, `mogf_light_direction(input, light)`, `mogf_light_attenuation(input, light)`, and `mogf_surface_shadow_visibility(input)`. The lower-level `mogf_shadow_visibility(world_position)` bypasses the material receive switch. `MogfLight` provides view-space position and ray direction, linear RGB color, intensity, range, a `kind` (`MOGF_LIGHT_AMBIENT`, `MOGF_LIGHT_DIRECTIONAL`, `MOGF_LIGHT_POINT`, or `MOGF_LIGHT_SPOT`), shadow state and bias/filter values, and spot inner/outer cone cosines. Spot lights point along local -Z; their half-cone angles are authored in radians and attenuation fades smoothly between inner and outer angles. `shadow_filter_radius` is measured in shadow-map texels; zero uses one nearest depth comparison and alpha lookup, while a positive value takes four hardware-filtered comparison taps and linearly filters the matching alpha transmittance. Project inputs bound radius to 0–8 texels and normalized depth bias to 0–0.05. Out-of-range indices return a zero-intensity ambient light. The helpers keep coordinate transforms and point/spot falloff consistent with built-in surfaces while letting the surface program choose its own diffuse/specular/rim model. `mobile_lambert.wgsl` is a small diffuse example that receives the directional shadow; `principled_tint.wgsl` shows the alternate route of retaining built-in Principled lighting.

`--vertex-shader` supplies session-wide `mogf_vertex(input: VertexOut) -> VertexOut`; returning the input preserves the built-in projection. Project camera effects use the same imported shader assets, with each `effects` record mapping to a `stage: "camera_effect"` shader and its ID in the native layer's ordered `effects` array. `mogf_effect(uv, color, resolution) -> vec4<f32>` can sample `mogf_scene` for pixelation and other whole-view operations. `--effect-shader` remains a session-wide fallback when the active layer has no project effect.

```json
"shaders": {
  "pixelate": { "asset": "pixelate.source", "stage": "camera_effect" }
},
"effects": {
  "pixelate.final": { "shader": "pixelate" }
},
"compositions": {
  "main": {
    "layers": [{
      "kind": "native_3d",
      "data": { "effects": ["pixelate.final"] }
    }]
  }
}
```

The native layer runs up to eight project-owned camera effects in listed order after the scene pass. Each effect samples the immediately preceding full-view buffer; intermediate surfaces remain GPU-resident and ping-pong between render targets. `examples/principled_pixelated.json` first pixelates, then applies a user WGSL warm/cool vertical grade. These are camera-level passes; per-mesh intermediate effects and a general resource/compute render graph remain future work.

A native layer can select a project-owned local-space vertex deformation using `vertex_shader`, whose ID resolves to a shader record with `stage: "vertex"`. The WGSL file defines `mogf_deform(position: vec3<f32>, shader_params: vec4<f32>) -> vec3<f32>`. This runs before model, camera, and light transforms, so the deformed geometry is used by both the camera render and shadow-map pass. Surface normals are derived from the deformed view-position derivatives. Because the WGSL hook does not declare a displacement bound, GPU frustum culling is bypassed for its instances; large deformations can also exceed the fitted shadow-camera margin. `--vertex-shader` remains a separate launch-time post-projection `mogf_vertex(VertexOut) -> VertexOut` hook.

Built-in materials are `unlit`, legacy `phong`, and `principled`. Principled parameters are linear base color, metallic, roughness, reflectance, and clear coat. A common plastic look is just a nonmetal Principled material with moderate roughness and 0.5 reflectance, as in `../principled_plastic.json`.

`liquid_glass_surface.wgsl` is a project-authored `surface_scene` response bound to an ordinary Principled material in that example. It builds a low-cost Fresnel rim and position-based ripple normal from public helpers, samples and offsets the current opaque scene buffer with `mogf_scene_color(input, offset_uv)`, applies the project light/shadow data, and uses `shader_params` for frequency, strength, opacity and tint blend. Its `mogf_shadow_opacity` hook keeps the shadow transmittance pass from rerunning refraction and lighting. The snapshot is taken after opaque rendering and before transparent rendering, so scene-reading materials do not see transparent objects or each other. This is a stylized refraction example, not a full transmission model.

The forward renderer accepts up to eight ambient, directional, point, or spot lights per frame. One directional or spot light may cast a fitted shadow (orthographic for directional, perspective for spot); `shadow_resolution` selects a 128, 256, or 512 square map, and `shadow_filter_radius` selects hard comparison or four-tap PCF while `shadow_bias` adjusts normalized light-depth comparison. Positive-radius taps use hardware-filtered depth comparisons and linearly filtered R8 transmittance, smoothing both opaque edges and alpha/cutout edges; radius zero retains nearest hard comparisons. A second pass accumulates transmittance from transparent instances and custom WGSL surfaces. The alpha returned from `mogf_surface(input)` controls shadow opacity per pixel, including procedural cutouts. Ordinary opaque casters write the depth map; custom surfaces are evaluated in the transmittance pass so their alpha is preserved. Built-in surfaces receive both effects, and custom material programs can use `mogf_default_lit_surface` or call `mogf_shadow_visibility(input.world_position)`. Point-light shadows, transparent-to-transparent refraction, mesh-local intermediate effects, and a composed multi-pass effect graph remain future work.
