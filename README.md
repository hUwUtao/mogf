# MOGF

A motion graphics system.

> Think of it like Houdini but FFMPEG

It support up to three dimension of graphics!

However, 2D graphics should be possible, not limited.

## How would I even do stuff?

Simple question, ask the damn LLM agents!

Long question: You might do, but its not designed mostly for you. Its not limited to your aesthetics or your layoutting.

The issue here are the solutions:

- Its actually too complex so that one layer is easy to manage: If you ask your AI for a gradient, it will write a generic shader. Complex enough, but it won't hardcode colors.
- In fact, you can cowork with it. Its your arts, shaders, musics, sfxs. Do whatever you like. Production is yours, with original artifacts (im not sure about the default color/layout choices of most LLM today, so mind it) 

## Whats the solution does it solve?

As far as I observed "just do it" result. LLM performs badly and repetitively. Mind that it cannot do actual work from start to end. A long steering is generally wanted.

## Rigid-body preview

Project format v2 compositions may define a `physics` world with gravity, `step_hz`, and rigid bodies keyed by scene object ID. Current preview bodies support ball, cuboid, and Y-capsule colliders with dynamic, fixed, or kinematic modes. In `mogf edit` and `mogf live`, physics runs on a worker thread; an uncached frame advances the session and the preview shows the latest completed simulation time while it catches up.

Open `examples/physics_preview.json` with `mogf edit` to see a ball fall onto a fixed floor. The preview cache is currently in memory for that session; persistent baking and final-render simulation playback are not implemented yet. The project `sample` and `render` commands continue to evaluate authored scene animation without simulating physics.
