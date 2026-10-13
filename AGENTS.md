# Repository Guidelines

@/home/stdpi/.codex/RTK.md

## Project

MOGF is a Rust motion-graphics application. Its serialized scene model is non-ECS; the runtime evaluates scenes and renders through one direct wgpu/Vulkan path. Hardware Vulkan and Mesa software Vulkan are adapters to that same path. CPU work prepares scene data and geometry; it does not rasterize frames. Keep WGSL assets within the GPU ABI.

Primary content is 2D shapes, sprites/images, and planar text placed in 3D space. Preserve XYZ transforms, groups, cameras, depth, shared geometry, explicit instances, and stable domain IDs. Billboarding is opt-in. Keep scene evaluation at explicit project time and distinguish pure caches, stateful steps, and effects.

## Repository layout

- `src/`: Rust runtime, editor, and shared wgpu renderer.
- `examples/`: project documents and shader examples.
- `assets/`: runtime assets.
- `agentic/`: local research, proposal history, corpus/search artifacts, and design records. This directory is ignored by Git and is not shipped.

The proposal CLI uses the local store at `agentic/agents/paperwork/proposals`; set `PAPER_PROPOSALS_ROOT=agentic/agents/paperwork/proposals` when invoking it. Review its messages before changing proposal records. Submitted revisions are immutable; only humans change review state.

## Working rules

- Prefix Linux terminal commands with `rtk`; prefer `rg` and `rg --files`.
- Keep changes scoped and preserve unrelated staged or unstaged work.
- Use `rtk cargo build` for binary/link verification. Do not claim a build from `cargo check` alone.
- Do not add tests unless requested; use focused runtime checks where suitable.
- Document only current behavior in shipped files. Keep research and historical proposal claims in `agentic/`.
