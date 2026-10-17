# Falling paragraph and scale markers

Open this self-contained harness in interactive preview to watch the paragraph descend under gravity and gently rain onto the ground plane. Each glyph has an independent rigid body built from its mesh convex hull, with a fallback bound derived from the actual shaped mesh, minimum collider thickness, and a small contact skin for thin extruded text. Stable IDs and small deterministic drift and spin let the text start readable and disperse naturally instead of bursting apart. Physics advances on preview cache misses, so no bake is required to play. Before playback, the paragraph remains a normal shaped-text batch; on playback, each glyph keeps its own render instance while its own rigid body moves it.

```sh
rtk cargo run -- live examples/physics_lorem_isometric.json main
rtk cargo run -- edit examples/physics_lorem_isometric.json main
rtk cargo run -- lint examples/physics_lorem_isometric.json --all-frames
```

Three condensed bold 18 m glyphs with 0.5 m extrusion lie face-up on the ground, with a separate bold ground caption. The suspended lowercase `a` is 1.8 m high as a human-scale reference. The scene uses only MOGF's shared font assets and scene definitions; it has no dependency on another example's content or assets.

This is a scene harness for interactive physics, not a performance benchmark; it does not claim the roadmap's throughput targets.
