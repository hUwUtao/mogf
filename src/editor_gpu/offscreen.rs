//! Headless host for the interactive renderer. Mesa software Vulkan is an
//! ordinary adapter here; there is no independent CPU scene rasterizer.
use super::*;
use std::sync::{Mutex, OnceLock, mpsc};

struct Target {
    size: [u32; 2],
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    readback: wgpu::Buffer,
    row_pitch: u32,
}

impl Target {
    fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        let texture = |format, sample_count, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("mogf-export-target"),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = texture(
            wgpu::TextureFormat::Rgba8UnormSrgb,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let color_view = color.create_view(&Default::default());
        let msaa_view = texture(
            wgpu::TextureFormat::Rgba8UnormSrgb,
            4,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
        .create_view(&Default::default());
        let depth_view = texture(
            wgpu::TextureFormat::Depth24Plus,
            4,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
        .create_view(&Default::default());
        let row_pitch = (size[0] * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mogf-export-readback"),
            size: u64::from(row_pitch) * u64::from(size[1]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            size,
            color,
            color_view,
            msaa_view,
            depth_view,
            readback,
            row_pitch,
        }
    }
}

struct Host {
    device: wgpu::Device,
    queue: wgpu::Queue,
    settings: Arc<RenderSettings>,
    resources: CallbackResources,
    target: Option<Target>,
}

impl Host {
    fn new() -> Result<Self, String> {
        // Validate the shared shader even when the caller only prepares exports.
        validate_surface_shader(
            "fn mogf_surface(input: VertexOut) -> vec4<f32> { return mogf_default_lit_surface(input); }",
        )?;
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| format!("cannot find a Vulkan adapter (hardware or Mesa software): {e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mogf-shared-render-device"),
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        let settings = Arc::new(RenderSettings::new(
            4,
            BlendMode::Alpha,
            DepthMode::TestWrite,
            CullMode::None,
            None,
            None,
            None,
            crate::quality::RenderProfile::Final,
        )?);
        Ok(Self {
            device,
            queue,
            settings,
            resources: Default::default(),
            target: None,
        })
    }

    fn render(&mut self, frame: &Frame, size: [u32; 2], ppm: &mut Vec<u8>) -> Result<(), String> {
        if size.contains(&0) || size.iter().any(|n| *n > 4096) {
            return Err("render dimensions must be 1..=4096".into());
        }
        if self
            .target
            .as_ref()
            .is_none_or(|target| target.size != size)
        {
            self.target = Some(Target::new(&self.device, size));
        }
        let packet = Arc::new(build_frame(
            frame,
            size[0] as f32 / size[1] as f32,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )?);
        let mut effects = packet.camera_effect_shaders.clone();
        if packet.requires_scene_color && effects.is_empty() {
            effects.push(Arc::from("fn mogf_effect(uv: vec2<f32>, color: vec4<f32>, resolution: vec2<f32>) -> vec4<f32> { return color; }"));
        }
        let callback = GpuCallback {
            frame: packet.clone(),
            effect_sources: effects,
            rect: egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(size[0] as f32, size[1] as f32),
            ),
            target_format: wgpu::TextureFormat::Rgba8UnormSrgb,
            settings: self.settings.clone(),
            emit_stats: false,
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mogf-shared-export"),
            });
        let additional = callback.prepare(
            &self.device,
            &self.queue,
            &eframe::egui_wgpu::ScreenDescriptor {
                size_in_pixels: size,
                pixels_per_point: 1.0,
            },
            &mut encoder,
            &mut self.resources,
        );
        let resources = self
            .resources
            .get::<GpuResources>()
            .ok_or("export resources were not prepared")?;
        let target = self.target.as_mut().unwrap();
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mogf-export-final-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.msaa_view,
                    resolve_target: Some(&target.color_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(effect) = callback.effect_sources.last() {
                let scene = resources
                    .scene_target
                    .as_ref()
                    .ok_or("missing export scene target")?;
                pass.set_pipeline(&resources.effect_pipelines[effect]);
                pass.set_bind_group(
                    0,
                    &scene.effect_bind_groups[callback.effect_sources.len() - 1],
                    &[],
                );
                pass.draw(0..3, 0..1);
            } else {
                draw_scene(
                    &packet,
                    resources,
                    &mut pass,
                    SceneDrawFilter::All,
                    &resources.default_scene_color_bind_group,
                );
            }
        }
        let pixel_bytes = u64::from(target.row_pitch) * u64::from(size[1]);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &target.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(target.row_pitch),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        let submission = self
            .queue
            .submit(additional.into_iter().chain([encoder.finish()]));
        let (tx, rx) = mpsc::channel();
        target
            .readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result.map_err(|e| e.to_string()));
            });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|e| e.to_string())?;
        rx.recv().map_err(|e| e.to_string())??;
        let mapped = target.readback.slice(..).get_mapped_range();
        ppm.clear();
        ppm.extend_from_slice(format!("P6\n{} {}\n255\n", size[0], size[1]).as_bytes());
        for row in mapped[..pixel_bytes as usize].chunks_exact(target.row_pitch as usize) {
            for pixel in row[..size[0] as usize * 4].chunks_exact(4) {
                ppm.extend_from_slice(&pixel[..3]);
            }
        }
        drop(mapped);
        target.readback.unmap();
        Ok(())
    }
}

pub(crate) fn render_frame(frame: &Frame, size: [u32; 2], ppm: &mut Vec<u8>) -> Result<(), String> {
    static HOST: OnceLock<Mutex<Host>> = OnceLock::new();
    if HOST.get().is_none() {
        let host = Host::new()?;
        let _ = HOST.set(Mutex::new(host));
    }
    HOST.get()
        .unwrap()
        .lock()
        .map_err(|_| "export host lock was poisoned".to_string())?
        .render(frame, size, ppm)
}
