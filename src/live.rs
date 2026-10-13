use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use eframe::egui;

use crate::{
    composition_frame_count, editor_gpu, evaluation, imports, model::Project, project_root,
};

pub(super) fn run(
    project_path: String,
    requested_composition: Option<String>,
    settings: editor_gpu::RenderSettings,
    emit_stats: bool,
) -> Result<(), String> {
    let loaded = imports::load_project(PathBuf::from(&project_path).as_path())
        .map_err(|error| error.to_string())?;
    let composition_id =
        crate::select_composition(&loaded.project, requested_composition.as_deref())?;
    let title = format!(
        "MOGF Live · {}",
        PathBuf::from(&project_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&project_path)
    );
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_app_id("mogf-live")
            .with_inner_size([960.0, 540.0])
            .with_min_inner_size([320.0, 180.0]),
        multisampling: settings.sample_count,
        depth_buffer: if settings.depth == editor_gpu::DepthMode::Disabled {
            0
        } else {
            24
        },
        wgpu_options: editor_gpu::vulkan_wgpu_configuration()?,
        ..Default::default()
    };
    eframe::run_native(
        &title,
        options,
        Box::new(move |cc| {
            let Some(render_state) = cc.wgpu_render_state.as_ref() else {
                return Err(std::io::Error::other("live requires a Vulkan wgpu renderer").into());
            };
            let adapter = render_state.adapter.get_info();
            if adapter.backend != eframe::wgpu::Backend::Vulkan {
                return Err(std::io::Error::other(format!(
                    "live expected Vulkan, got {:?}",
                    adapter.backend
                ))
                .into());
            }
            eprintln!(
                "live Vulkan adapter: {} ({:?})",
                adapter.name, adapter.device_type
            );
            let mut app = LiveApp::new(
                project_path,
                composition_id,
                Some(render_state.target_format),
                Arc::new(settings),
                emit_stats,
            );
            app.reload(&cc.egui_ctx, true);
            Ok(Box::new(app))
        }),
    )
    .map_err(|error| format!("live window failed: {error}"))
}

struct LiveApp {
    project_path: PathBuf,
    composition_id: String,
    project: Option<Project>,
    project_json: Option<serde_json::Value>,
    source_files: std::collections::BTreeMap<PathBuf, Option<imports::SourceStamp>>,
    gpu_format: Option<eframe::wgpu::TextureFormat>,
    render_settings: Arc<editor_gpu::RenderSettings>,
    emit_stats: bool,
    preview_gpu: Option<Arc<editor_gpu::SceneFrame>>,
    last_reload: Instant,
    last_poll: Instant,
    last_tick: Instant,
    last_title: Instant,
    fps_window: Instant,
    fps_window_frames: u32,
    frame: u64,
    measured_fps: u32,
    frame_prep_ms: f32,
    sample_ms: f32,
    scene_build_ms: f32,
    playing: bool,
    audio: Option<crate::audio::CompositionAudio>,
    error: Option<String>,
    status: String,
}

impl LiveApp {
    fn new(
        project_path: String,
        composition_id: String,
        gpu_format: Option<eframe::wgpu::TextureFormat>,
        render_settings: Arc<editor_gpu::RenderSettings>,
        emit_stats: bool,
    ) -> Self {
        Self {
            project_path: PathBuf::from(project_path),
            composition_id,
            project: None,
            project_json: None,
            source_files: Default::default(),
            gpu_format,
            render_settings,
            emit_stats,
            preview_gpu: None,
            last_reload: Instant::now() - Duration::from_secs(1),
            last_poll: Instant::now() - Duration::from_millis(300),
            last_tick: Instant::now(),
            last_title: Instant::now() - Duration::from_secs(1),
            fps_window: Instant::now(),
            fps_window_frames: 0,
            frame: 0,
            measured_fps: 0,
            frame_prep_ms: 0.0,
            sample_ms: 0.0,
            scene_build_ms: 0.0,
            playing: true,
            audio: None,
            error: None,
            status: "waiting for a valid scene".to_owned(),
        }
    }

    fn reload(&mut self, ctx: &egui::Context, force: bool) -> bool {
        if !force {
            let changed =
                self.error.is_some() || imports::watched_sources_changed(&self.source_files);
            if !changed {
                return false;
            }
            if self.error.is_some()
                && Instant::now().saturating_duration_since(self.last_reload)
                    < Duration::from_millis(500)
            {
                return false;
            }
        }
        self.last_reload = Instant::now();
        let loaded = match imports::load_project(&self.project_path) {
            Ok(loaded) => loaded,
            Err(error) => {
                let location = match (error.line, error.column) {
                    (Some(line), Some(column)) => format!(" at line {line}, column {column}"),
                    _ => String::new(),
                };
                self.show_error(
                    ctx,
                    format!("{}{}: {}", error.code, location, error.message),
                );
                return true;
            }
        };
        let next = loaded.project;
        let next_json = loaded.json;
        let next_source_files = loaded.source_files;
        let next_frame = self.preserved_frame(&next);
        let candidate_preview = match self.build_preview(&next, next_frame) {
            Ok(preview) => preview,
            Err(error) => {
                self.show_error(ctx, format!("scene error at frame {next_frame}: {error}"));
                return true;
            }
        };
        let root = project_root(&self.project_path.to_string_lossy());
        let candidate_audio =
            match crate::audio::CompositionAudio::load(&next, &root, &self.composition_id) {
                Ok(mut audio) => {
                    if let Some(audio) = &mut audio {
                        let fps = next.compositions[&self.composition_id]
                            .fps
                            .as_positive()
                            .unwrap();
                        if let Err(error) = audio.seek(
                            next_frame as f64 * fps.d as f64 / fps.n as f64,
                            self.playing,
                        ) {
                            self.show_error(ctx, error);
                            return true;
                        }
                    }
                    audio
                }
                Err(error) => {
                    self.show_error(ctx, error);
                    return true;
                }
            };

        let changes = if let (Some(current), Some(previous_json)) =
            (self.project.as_mut(), self.project_json.as_ref())
        {
            current.rehydrate(next, previous_json, &next_json)
        } else {
            self.project = Some(next);
            Vec::new()
        };
        self.preview_gpu = Some(candidate_preview);
        self.audio = candidate_audio;
        self.project_json = Some(next_json);
        self.source_files = next_source_files;
        self.frame = next_frame;
        self.error = None;
        self.last_tick = Instant::now();
        self.status = if changes.is_empty() {
            format!("frame {} (no data changes)", self.frame)
        } else {
            let summary = changes
                .iter()
                .map(|change| {
                    format!(
                        "{} +{} ~{} -{}",
                        change.table, change.added, change.updated, change.removed
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("frame {}: {summary}", self.frame)
        };
        eprintln!("live reloaded: {}", self.status);
        self.update_title(ctx);
        true
    }

    fn build_preview(
        &self,
        project: &Project,
        frame_number: u64,
    ) -> Result<Arc<editor_gpu::SceneFrame>, String> {
        let target_format = self
            .gpu_format
            .ok_or("live renderer did not receive a wgpu surface")?;
        let root = project_root(&self.project_path.to_string_lossy());
        let frame = evaluation::sample_with_profile(
            project,
            &root,
            &self.composition_id,
            frame_number,
            self.render_settings.profile,
        )?;
        let composition = project
            .compositions
            .get(&self.composition_id)
            .ok_or_else(|| format!("missing composition {:?}", self.composition_id))?;
        let aspect = composition.size[0] as f32 / composition.size[1].max(1) as f32;
        Ok(Arc::new(editor_gpu::build_frame(
            &frame,
            aspect,
            target_format,
        )?))
    }

    fn preserved_frame(&self, next: &Project) -> u64 {
        let Some(previous) = &self.project else {
            return 0;
        };
        let (Some(old_comp), Some(new_comp)) = (
            previous.compositions.get(&self.composition_id),
            next.compositions.get(&self.composition_id),
        ) else {
            return 0;
        };
        let old_rate = old_comp.fps.as_positive();
        let new_rate = new_comp.fps.as_positive();
        let (Ok(old_rate), Ok(new_rate)) = (old_rate, new_rate) else {
            return self.frame;
        };
        let mut numerator = [
            u128::from(self.frame),
            u128::from(old_rate.d),
            u128::from(new_rate.n as u64),
        ];
        let mut denominator = [u128::from(old_rate.n as u64), u128::from(new_rate.d)];
        for top in &mut numerator {
            for bottom in &mut denominator {
                let divisor = gcd(*top, *bottom);
                *top /= divisor;
                *bottom /= divisor;
            }
        }
        let frame = numerator
            .into_iter()
            .try_fold(1_u128, u128::checked_mul)
            .zip(denominator.into_iter().try_fold(1_u128, u128::checked_mul))
            .map(|(top, bottom)| top / bottom)
            .and_then(|frame| u64::try_from(frame).ok())
            .unwrap_or(self.frame);
        composition_frame_count(new_comp)
            .map(|count| frame.min(count.saturating_sub(1)))
            .unwrap_or(frame)
    }

    fn frame_interval(&self) -> Duration {
        self.project
            .as_ref()
            .and_then(|project| project.compositions.get(&self.composition_id))
            .and_then(|composition| composition.fps.as_positive().ok())
            .map(|fps| {
                Duration::from_secs_f64((fps.d as f64 / fps.n as f64).clamp(1.0 / 60.0, 1.0))
            })
            .unwrap_or(Duration::from_millis(33))
    }

    fn repaint_delay(&self, now: Instant) -> Duration {
        let interval = self.frame_interval();
        if !self.playing {
            return interval;
        }
        let interval_nanos = interval.as_nanos();
        let elapsed_nanos = now.saturating_duration_since(self.last_tick).as_nanos();
        let phase = Duration::from_nanos((elapsed_nanos % interval_nanos) as u64);
        interval.saturating_sub(phase).max(Duration::from_millis(1))
    }

    fn advance(&mut self, ctx: &egui::Context, now: Instant) {
        if !self.playing {
            return;
        }
        let interval = self.frame_interval();
        let elapsed = now.saturating_duration_since(self.last_tick);
        if elapsed < interval {
            return;
        }
        let steps = (elapsed.as_secs_f64() / interval.as_secs_f64())
            .floor()
            .max(1.0) as u64;
        self.last_tick =
            now - Duration::from_secs_f64(elapsed.as_secs_f64() % interval.as_secs_f64());
        let Some(project) = &self.project else { return };
        let Some(composition) = project.compositions.get(&self.composition_id) else {
            return;
        };
        let Ok(count) = composition_frame_count(composition) else {
            return;
        };
        if count == 0 {
            return;
        }
        if let Some(audio) = &mut self.audio {
            let fps = composition.fps.as_positive().unwrap();
            let rate = fps.n as f64 / fps.d as f64;
            let position = (audio.position() * rate).floor() as u64;
            if position >= count || audio.ended() {
                if let Err(error) = audio.seek(0.0, true) {
                    self.show_error(ctx, error);
                    return;
                }
                self.frame = 0;
            } else {
                self.frame = position;
            }
        } else {
            self.frame = self.frame.wrapping_add(steps) % count;
        }
        let prep_started = Instant::now();
        if let Err(error) = self.render_current() {
            self.show_error(ctx, format!("frame {}: {error}", self.frame));
        }
        self.frame_prep_ms = prep_started.elapsed().as_secs_f32() * 1000.0;
        if now.saturating_duration_since(self.last_title) >= Duration::from_secs(1) {
            self.update_title(ctx);
        }
    }

    fn render_current(&mut self) -> Result<(), String> {
        let project = self.project.as_ref().ok_or("no project loaded")?;
        let target_format = self
            .gpu_format
            .ok_or("live renderer did not receive a wgpu surface")?;
        let root = project_root(&self.project_path.to_string_lossy());
        let sample_started = Instant::now();
        let frame = evaluation::sample_with_profile(
            project,
            &root,
            &self.composition_id,
            self.frame,
            self.render_settings.profile,
        )?;
        self.sample_ms = sample_started.elapsed().as_secs_f32() * 1000.0;
        let composition = project
            .compositions
            .get(&self.composition_id)
            .ok_or_else(|| format!("missing composition {:?}", self.composition_id))?;
        let aspect = composition.size[0] as f32 / composition.size[1].max(1) as f32;
        let scene_started = Instant::now();
        self.preview_gpu = Some(Arc::new(editor_gpu::build_frame(
            &frame,
            aspect,
            target_format,
        )?));
        self.scene_build_ms = scene_started.elapsed().as_secs_f32() * 1000.0;
        self.error = None;
        Ok(())
    }

    fn show_error(&mut self, ctx: &egui::Context, error: String) {
        if self.error.as_deref() != Some(&error) {
            eprintln!("live: {error}");
        }
        self.error = Some(error);
        self.update_title(ctx);
    }

    fn update_title(&mut self, ctx: &egui::Context) {
        self.last_title = Instant::now();
        let mode = if self.playing { "playing" } else { "paused" };
        let state = self.error.as_deref().unwrap_or(mode);
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!(
            "MOGF Live · {} · frame {} · {state}",
            self.composition_id, self.frame
        )));
    }

    fn handle_input(&mut self, ctx: &egui::Context) {
        let (toggle, back, forward, reload) = ctx.input(|input| {
            (
                input.key_pressed(egui::Key::Space),
                input.key_pressed(egui::Key::ArrowLeft),
                input.key_pressed(egui::Key::ArrowRight),
                input.key_pressed(egui::Key::R),
            )
        });
        if toggle {
            self.playing = !self.playing;
            if let Some(audio) = &self.audio {
                audio.set_playing(self.playing);
            }
            self.last_tick = Instant::now();
            self.update_title(ctx);
        }
        if back || forward {
            self.playing = false;
            if let Some(project) = &self.project {
                if let Some(composition) = project.compositions.get(&self.composition_id) {
                    if let Ok(count) = composition_frame_count(composition) {
                        if count > 0 {
                            self.frame = if forward {
                                self.frame.wrapping_add(1) % count
                            } else if self.frame == 0 {
                                count - 1
                            } else {
                                self.frame - 1
                            };
                            if let Err(error) = self.render_current() {
                                self.show_error(ctx, format!("frame {}: {error}", self.frame));
                            }
                        }
                    }
                }
            }
            self.update_title(ctx);
        }
        if back || forward {
            let seconds = self.frame as f64 * self.frame_interval().as_secs_f64();
            if let Some(audio) = &mut self.audio {
                if let Err(error) = audio.seek(seconds, false) {
                    self.show_error(ctx, error);
                }
            }
        }
        if reload {
            self.reload(ctx, true);
        }
    }
}

impl eframe::App for LiveApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.fps_window_frames += 1;
        let fps_elapsed = self.fps_window.elapsed();
        if fps_elapsed >= Duration::from_secs(1) {
            self.measured_fps =
                (self.fps_window_frames as f64 / fps_elapsed.as_secs_f64()).round() as u32;
            if self.emit_stats {
                eprintln!(
                    "live metrics: ui_repaint_fps={} scene_prepare_ms={:.2} sample_ms={:.2} packet_ms={:.2} frame={}",
                    self.measured_fps,
                    self.frame_prep_ms,
                    self.sample_ms,
                    self.scene_build_ms,
                    self.frame
                );
            }
            self.fps_window_frames = 0;
            self.fps_window = Instant::now();
        }
        self.handle_input(&ctx);
        let now = Instant::now();
        if now.saturating_duration_since(self.last_poll) >= Duration::from_millis(200) {
            self.last_poll = now;
            self.reload(&ctx, false);
        }
        self.advance(&ctx, now);
        ctx.request_repaint_after(self.repaint_delay(Instant::now()));

        let rect = ui.available_rect_before_wrap();
        ui.painter()
            .rect_filled(rect, 0, egui::Color32::from_rgb(12, 20, 30));
        if let Some(frame) = &self.preview_gpu {
            let composition_size = self
                .project
                .as_ref()
                .and_then(|project| project.compositions.get(&self.composition_id))
                .map(|composition| composition.size)
                .unwrap_or([16, 9]);
            let image_size = egui::vec2(
                composition_size[0].max(1) as f32,
                composition_size[1].max(1) as f32,
            );
            let scale = (rect.width() / image_size.x).min(rect.height() / image_size.y);
            let image_rect = egui::Rect::from_center_size(rect.center(), image_size * scale);
            if let Some(target_format) = self.gpu_format {
                ui.painter().add(editor_gpu::callback_shape(
                    Arc::clone(frame),
                    image_rect,
                    target_format,
                    Arc::clone(&self.render_settings),
                    self.emit_stats,
                ));
            }
        } else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Waiting for a valid scene",
                egui::FontId::proportional(16.0),
                egui::Color32::LIGHT_GRAY,
            );
        }
        if let Some(error) = &self.error {
            ui.painter().text(
                rect.left_top() + egui::vec2(12.0, 12.0),
                egui::Align2::LEFT_TOP,
                error,
                egui::FontId::monospace(12.0),
                egui::Color32::from_rgb(241, 110, 107),
            );
        }
        ui.painter().text(
            rect.left_bottom() + egui::vec2(12.0, -10.0),
            egui::Align2::LEFT_BOTTOM,
            format!(
                "{} · frame {} · Space pause · ←/→ step · R reload",
                self.status, self.frame
            ),
            egui::FontId::monospace(11.0),
            egui::Color32::from_rgb(133, 157, 169),
        );
        ui.painter().text(
            rect.right_bottom() + egui::vec2(-12.0, -10.0),
            egui::Align2::RIGHT_BOTTOM,
            format!(
                "{} UI fps · frame prep {:.1} ms",
                self.measured_fps, self.frame_prep_ms
            ),
            egui::FontId::monospace(11.0),
            egui::Color32::from_rgb(133, 157, 169),
        );
        ui.allocate_rect(rect, egui::Sense::hover());
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}
