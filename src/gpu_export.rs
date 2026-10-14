//! Export adapter for the same wgpu renderer used by live/edit.
use crate::{editor_gpu, model::Frame};

pub(super) fn render_ppm(frame: &Frame, size: [u32; 2], msaa: u16) -> Result<Vec<u8>, String> {
    let mut ppm = Vec::new();
    editor_gpu::offscreen::render_frame(frame, size, msaa, &mut ppm)?;
    Ok(ppm)
}

pub(super) fn render_rgba8(frame: &Frame, size: [u32; 2], msaa: u16) -> Result<Vec<u8>, String> {
    let mut rgba = Vec::new();
    editor_gpu::offscreen::render_rgba8(frame, size, msaa, &mut rgba)?;
    Ok(rgba)
}
