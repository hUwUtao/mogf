//! Export adapter for the same wgpu renderer used by live/edit.
use crate::{editor_gpu, model::Frame};

pub(super) fn render_ppm(frame: &Frame, size: [u32; 2]) -> Result<Vec<u8>, String> {
    let mut ppm = Vec::new();
    editor_gpu::offscreen::render_frame(frame, size, &mut ppm)?;
    Ok(ppm)
}
