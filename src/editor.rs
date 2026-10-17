use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use eframe::egui::{self, Color32, RichText, Stroke, Vec2};

use crate::{
    composition_frame_count, editor_gpu, evaluation, imports,
    model::{Project, Rational},
    project_root,
};

const BG: Color32 = Color32::from_gray(32);
const PANEL: Color32 = Color32::from_gray(40);
const PANEL_LIGHT: Color32 = Color32::from_gray(52);
const INK: Color32 = Color32::from_gray(220);
const MUTED: Color32 = Color32::from_gray(145);
const STATUS: Color32 = Color32::from_gray(190);

pub(super) fn run(
    project_path: String,
    composition_id: Option<String>,
    settings: editor_gpu::RenderSettings,
) -> Result<(), String> {
    let title = format!(
        "MOGF Editor · {}",
        Path::new(&project_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(&project_path)
    );
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_app_id("mogf-editor")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([980.0, 680.0]),
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
                return Err(std::io::Error::other("editor requires a Vulkan wgpu renderer").into());
            };
            editor_gpu::install_pipeline_cache(
                &render_state.adapter,
                &render_state.device,
                &mut render_state.renderer.write().callback_resources,
            );
            let adapter = render_state.adapter.get_info();
            if adapter.backend != eframe::wgpu::Backend::Vulkan {
                return Err(std::io::Error::other(format!(
                    "editor expected Vulkan, got {:?}",
                    adapter.backend
                ))
                .into());
            }
            eprintln!(
                "editor Vulkan adapter: {} ({:?})",
                adapter.name, adapter.device_type
            );
            let gpu_format = Some(render_state.target_format);
            let mut app =
                EditorApp::new(project_path, composition_id, gpu_format, Arc::new(settings));
            app.apply_theme(&cc.egui_ctx);
            app.reload();
            Ok(Box::new(app))
        }),
    )
    .map_err(|error| format!("editor window failed: {error}"))
}

struct EditorApp {
    project_path: PathBuf,
    project: Option<Project>,
    project_json: Option<serde_json::Value>,
    source_files: std::collections::BTreeMap<PathBuf, Option<imports::SourceStamp>>,
    composition_id: Option<String>,
    selected_object: Option<String>,
    frame: u64,
    playing: bool,
    audio: Option<crate::audio::CompositionAudio>,
    audio_composition: Option<String>,
    last_tick: Instant,
    last_poll: Instant,
    last_reload: Instant,
    preview_gpu: Option<Arc<editor_gpu::SceneFrame>>,
    physics: Option<crate::physics::PhysicsWorker>,
    gpu_format: Option<eframe::wgpu::TextureFormat>,
    render_settings: Arc<editor_gpu::RenderSettings>,
    preview_size: [u32; 2],
    snapshot: Option<serde_json::Value>,
    show_snapshot: bool,
    inspector_open: bool,
    lint_messages: Vec<String>,
    error: Option<String>,
    status: String,
}

impl EditorApp {
    fn new(
        project_path: String,
        composition_id: Option<String>,
        gpu_format: Option<eframe::wgpu::TextureFormat>,
        render_settings: Arc<editor_gpu::RenderSettings>,
    ) -> Self {
        Self {
            project_path: PathBuf::from(project_path),
            project: None,
            project_json: None,
            source_files: Default::default(),
            composition_id,
            selected_object: None,
            frame: 0,
            playing: false,
            audio: None,
            audio_composition: None,
            last_tick: Instant::now(),
            last_poll: Instant::now() - Duration::from_secs(1),
            last_reload: Instant::now() - Duration::from_secs(1),
            preview_gpu: None,
            physics: None,
            gpu_format,
            render_settings,
            preview_size: [960, 540],
            snapshot: None,
            show_snapshot: false,
            inspector_open: false,
            lint_messages: Vec::new(),
            error: None,
            status: "Opening project".into(),
        }
    }

    fn apply_theme(&self, ctx: &egui::Context) {
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = BG;
        visuals.window_fill = PANEL;
        visuals.extreme_bg_color = Color32::from_gray(24);
        visuals.faint_bg_color = PANEL;
        visuals.widgets.noninteractive.bg_fill = PANEL;
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, INK);
        visuals.widgets.inactive.bg_fill = PANEL_LIGHT;
        visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, INK);
        visuals.widgets.hovered.bg_fill = Color32::from_gray(64);
        visuals.widgets.active.bg_fill = Color32::from_gray(72);
        visuals.widgets.open.bg_fill = Color32::from_gray(64);
        visuals.widgets.open.fg_stroke = Stroke::new(1.0, INK);
        visuals.selection.bg_fill = Color32::from_gray(72);
        visuals.selection.stroke = Stroke::new(1.0, INK);
        visuals.hyperlink_color = INK;
        visuals.warn_fg_color = INK;
        visuals.error_fg_color = INK;
        ctx.set_visuals(visuals);
        let mut style = (*ctx.style_of(egui::Theme::Dark)).clone();
        style.spacing.item_spacing = Vec2::new(8.0, 7.0);
        style.visuals.window_corner_radius = egui::CornerRadius::same(4);
        ctx.set_style_of(egui::Theme::Dark, style);
    }

    fn reload(&mut self) {
        self.last_reload = Instant::now();
        let loaded = match imports::load_project(&self.project_path) {
            Ok(project) => project,
            Err(error) => {
                let location = match (error.line, error.column) {
                    (Some(line), Some(column)) => format!(" · line {line}, column {column}"),
                    _ => String::new(),
                };
                self.error = Some(format!("{}{}: {}", error.code, location, error.message));
                self.status = "Project data needs attention".into();
                return;
            }
        };
        let next_json = loaded.json;
        let next_sources = loaded.source_files;
        let next = loaded.project;
        let next_composition = self
            .composition_id
            .as_ref()
            .filter(|id| next.compositions.contains_key(*id))
            .cloned()
            .or_else(|| {
                next.compositions
                    .contains_key("main")
                    .then(|| "main".to_owned())
            })
            .or_else(|| {
                (next.compositions.len() == 1)
                    .then(|| next.compositions.keys().next().cloned())
                    .flatten()
            })
            .or_else(|| next.compositions.keys().next().cloned());
        let same_composition = self.composition_id == next_composition;
        let next_frame = next_composition
            .as_ref()
            .and_then(|id| next.compositions.get(id))
            .and_then(|composition| composition_frame_count(composition).ok())
            .map(|count| {
                if same_composition {
                    self.frame.min(count.saturating_sub(1))
                } else {
                    0
                }
            })
            .unwrap_or(0);
        let next_root = project_root(&self.project_path.to_string_lossy());
        let next_physics = next_composition
            .as_ref()
            .and_then(|id| {
                next.compositions
                    .get(id)
                    .and_then(|composition| composition.physics.as_ref())
                    .map(|world| (id, world))
            })
            .map(|(id, world)| {
                evaluation::sample_with_profile(
                    &next,
                    &next_root,
                    id,
                    0,
                    self.render_settings.profile,
                )
                .and_then(|frame| crate::physics::PhysicsWorker::new(world, &frame))
            })
            .transpose();
        let next_physics = match next_physics {
            Ok(worker) => worker,
            Err(error) => {
                self.error = Some(error);
                self.status = "Simulation setup failed; previous scene retained".into();
                return;
            }
        };
        let (next_snapshot, next_preview, next_preview_size) = if let Some(composition_id) =
            &next_composition
        {
            let frame = match evaluation::sample_with_profile(
                &next,
                &next_root,
                composition_id,
                next_frame,
                self.render_settings.profile,
            ) {
                Ok(frame) => frame,
                Err(error) => {
                    self.error = Some(format!("Reload candidate frame {next_frame}: {error}"));
                    self.status = "Reload candidate rejected; previous scene retained".into();
                    return;
                }
            };
            let preview_size = next.compositions[composition_id].size;
            let preview = if let Some(target_format) = self.gpu_format {
                let aspect = preview_size[0] as f32 / preview_size[1].max(1) as f32;
                match editor_gpu::build_frame(&frame, aspect, target_format) {
                    Ok(packet) => Some(Arc::new(packet)),
                    Err(error) => {
                        self.error = Some(format!("Reload candidate GPU frame: {error}"));
                        self.status = "Reload candidate rejected; previous scene retained".into();
                        return;
                    }
                }
            } else {
                None
            };
            (
                serde_json::to_value(&frame).ok(),
                preview,
                Some(preview_size),
            )
        } else {
            (None, None, None)
        };
        let mut next_audio = match &next_composition {
            Some(composition_id) => {
                match crate::audio::CompositionAudio::load(&next, &next_root, composition_id) {
                    Ok(mut audio) => {
                        if let Some(audio) = &mut audio {
                            if let Some(composition) = next.compositions.get(composition_id) {
                                if let Ok(fps) = composition.fps.as_positive() {
                                    if let Err(error) = audio.seek(
                                        next_frame as f64 * fps.d as f64 / fps.n as f64,
                                        self.playing,
                                    ) {
                                        self.error = Some(error);
                                        self.status =
                                            "Reload candidate rejected; previous scene retained"
                                                .into();
                                        return;
                                    }
                                }
                            }
                        }
                        audio
                    }
                    Err(error) => {
                        self.error = Some(error);
                        self.status = "Reload candidate rejected; previous scene retained".into();
                        return;
                    }
                }
            }
            None => None,
        };
        if let (Some(current), Some(previous_json)) =
            (&mut self.project, self.project_json.as_ref())
        {
            let changes = current.rehydrate(next, previous_json, &next_json);
            self.status = if changes.is_empty() {
                "Rehydrated · records unchanged".into()
            } else {
                let summary = changes
                    .iter()
                    .map(|c| format!("{} +{} ~{} −{}", c.table, c.added, c.updated, c.removed))
                    .collect::<Vec<_>>()
                    .join("   ");
                format!("Rehydrated · {summary}")
            };
        } else {
            self.project = Some(next);
            self.status = "Project ready".into();
        }
        self.project_json = Some(next_json);
        self.physics = next_physics;
        self.source_files = next_sources;
        self.composition_id = next_composition;
        self.frame = next_frame;
        self.snapshot = next_snapshot;
        self.preview_gpu = next_preview;
        if let Some(size) = next_preview_size {
            self.preview_size = size;
        }
        self.audio = next_audio.take();
        self.audio_composition = self.composition_id.clone();
        self.error = None;
        self.lint_messages = self
            .project
            .as_ref()
            .map(|project| lint_project(project, &self.project_path))
            .unwrap_or_default();
        if !self.lint_messages.is_empty() {
            self.status = format!("Lint · {} issue(s)", self.lint_messages.len());
            self.inspector_open = true;
        }
        if let Some(project) = &self.project
            && self
                .selected_object
                .as_ref()
                .is_none_or(|id| !project.objects.contains_key(id))
        {
            self.selected_object = project.objects.keys().next().cloned();
        }
        self.last_tick = Instant::now();
    }

    fn poll_reload(&mut self) {
        if Instant::now().saturating_duration_since(self.last_poll) < Duration::from_millis(300) {
            return;
        }
        self.last_poll = Instant::now();
        let changed =
            self.project.is_none() || imports::watched_sources_changed(&self.source_files);
        if changed
            && Instant::now().saturating_duration_since(self.last_reload)
                >= Duration::from_millis(500)
        {
            self.reload();
        }
    }

    fn frame_count(&self) -> u64 {
        self.current_composition()
            .and_then(|c| composition_frame_count(c).ok())
            .unwrap_or(1)
            .max(1)
    }

    fn current_composition(&self) -> Option<&crate::model::Composition> {
        self.project
            .as_ref()?
            .compositions
            .get(self.composition_id.as_deref()?)
    }

    fn render_frame(&mut self, _ctx: &egui::Context) {
        let (Some(project), Some(composition_id)) = (&self.project, &self.composition_id) else {
            return;
        };
        let root = project_root(self.project_path.to_string_lossy().as_ref());
        let composition_size = self
            .current_composition()
            .map(|composition| composition.size)
            .unwrap_or([16, 9]);
        let mut frame = match evaluation::sample_with_profile(
            project,
            &root,
            composition_id,
            self.frame,
            self.render_settings.profile,
        ) {
            Ok(frame) => frame,
            Err(error) => {
                self.error = Some(format!("Frame {}: {error}", self.frame));
                self.inspector_open = true;
                return;
            }
        };
        let composition = self.project.as_ref().and_then(|project| {
            self.composition_id
                .as_ref()
                .and_then(|id| project.compositions.get(id))
        });
        if let (Some(composition), Some(worker)) = (composition, self.physics.as_mut()) {
            if let Some(world) = &composition.physics {
                match crate::physics::tick_for_frame(self.frame, &composition.fps, world.step_hz) {
                    Ok(tick) => {
                        worker.request_tick(tick);
                        match worker.poll() {
                            Ok(Some(sample)) => {
                                crate::physics::apply_sample(&mut frame, sample, world);
                                self.status = if sample.tick < tick {
                                    format!(
                                        "Physics catching up: showing {:.3}s",
                                        sample.time_seconds
                                    )
                                } else {
                                    format!("Physics ready: {:.3}s", sample.time_seconds)
                                };
                            }
                            Ok(None) => {}
                            Err(error) => self.error = Some(error),
                        }
                    }
                    Err(error) => self.error = Some(error),
                }
            }
        }
        if let Some(target_format) = self.gpu_format {
            let aspect = composition_size[0] as f32 / composition_size[1].max(1) as f32;
            match editor_gpu::build_frame(&frame, aspect, target_format) {
                Ok(packet) => {
                    self.preview_gpu = Some(Arc::new(packet));
                    self.preview_size = composition_size;
                    self.snapshot = serde_json::to_value(&frame).ok();
                    self.error = None;
                }
                Err(error) => {
                    self.error = Some(format!("GPU frame: {error}"));
                    self.inspector_open = true;
                }
            }
            return;
        }
        self.error = Some("Preview requires a wgpu adapter".into());
    }

    fn set_frame(&mut self, frame: u64, ctx: &egui::Context) {
        let last = self.frame_count().saturating_sub(1);
        self.frame = frame.min(last);
        self.last_tick = Instant::now();
        self.sync_audio(ctx, true);
        self.render_frame(ctx);
    }

    fn sync_audio(&mut self, _ctx: &egui::Context, seek: bool) {
        if self.audio_composition != self.composition_id {
            self.audio = None;
            if let (Some(project), Some(id)) = (&self.project, &self.composition_id) {
                let root = project_root(&self.project_path.to_string_lossy());
                match crate::audio::CompositionAudio::load(project, &root, id) {
                    Ok(audio) => {
                        self.audio = audio;
                        self.audio_composition = self.composition_id.clone();
                    }
                    Err(error) => {
                        self.error = Some(error);
                        self.playing = false;
                        return;
                    }
                }
            }
        }
        let rate = self
            .current_composition()
            .and_then(|c| c.fps.as_positive().ok())
            .map(|r| r.n as f64 / r.d as f64)
            .unwrap_or(24.0);
        if let Some(audio) = &mut self.audio {
            if seek {
                if let Err(error) = audio.seek(self.frame as f64 / rate, self.playing) {
                    self.error = Some(error);
                    self.playing = false;
                }
            } else {
                audio.set_playing(self.playing);
            }
        }
    }

    fn tick(&mut self, ctx: &egui::Context) {
        if !self.playing {
            return;
        }
        let fps = self
            .current_composition()
            .and_then(|c| c.fps.as_positive().ok())
            .map(|r| r.n as f64 / r.d as f64)
            .unwrap_or(24.0)
            .max(1.0);
        let step = Duration::from_secs_f64(1.0 / fps);
        if self.last_tick.elapsed() >= step {
            let count = self.frame_count();
            let next = self
                .audio
                .as_ref()
                .map(|audio| (audio.position() * fps).floor() as u64)
                .unwrap_or_else(|| {
                    self.frame + (self.last_tick.elapsed().as_secs_f64() * fps).floor() as u64
                });
            if next >= count || self.audio.as_ref().is_some_and(|audio| audio.ended()) {
                self.set_frame(0, ctx);
            } else {
                self.frame = next;
                self.last_tick = Instant::now();
                self.render_frame(ctx);
            }
        }
        ctx.request_repaint_after(Duration::from_millis(8));
    }

    fn playback_shortcuts(&mut self, ctx: &egui::Context) {
        let (toggle, previous, next) = ctx.input(|input| {
            (
                input.key_pressed(egui::Key::Space),
                input.key_pressed(egui::Key::ArrowLeft),
                input.key_pressed(egui::Key::ArrowRight),
            )
        });
        if toggle {
            self.playing = !self.playing;
            self.last_tick = Instant::now();
        }
        if !self.playing {
            if previous {
                self.set_frame(self.frame.saturating_sub(1), ctx);
            } else if next {
                self.set_frame((self.frame + 1) % self.frame_count(), ctx);
            }
        }
    }

    fn header(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::Frame::new()
            .fill(PANEL)
            .inner_margin(egui::Margin::symmetric(10, 5))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("MOGF").strong());
                    ui.label(RichText::new("GPU").small().strong());
                    let compositions = self
                        .project
                        .as_ref()
                        .map(|p| p.compositions.keys().cloned().collect::<Vec<_>>())
                        .unwrap_or_default();
                    let selected = self.composition_id.clone().unwrap_or_else(|| "—".into());
                    egui::ComboBox::from_id_salt("composition-picker")
                        .selected_text(selected)
                        .width(155.0)
                        .show_ui(ui, |ui| {
                            for id in compositions {
                                if ui
                                    .selectable_label(
                                        self.composition_id.as_deref() == Some(&id),
                                        &id,
                                    )
                                    .clicked()
                                {
                                    self.composition_id = Some(id);
                                    self.set_frame(0, ctx);
                                }
                            }
                        });
                    let objects = self
                        .project
                        .as_ref()
                        .map(|p| {
                            p.objects
                                .iter()
                                .map(|(id, object)| (id.clone(), object.name.clone()))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    let object_label = self
                        .selected_object
                        .as_ref()
                        .and_then(|id| {
                            objects
                                .iter()
                                .find(|(key, _)| key == id)
                                .map(|(_, name)| name.clone())
                        })
                        .unwrap_or_else(|| "Object".into());
                    egui::ComboBox::from_id_salt("object-picker")
                        .selected_text(object_label)
                        .width(155.0)
                        .show_ui(ui, |ui| {
                            for (id, name) in objects {
                                if ui
                                    .selectable_label(
                                        self.selected_object.as_deref() == Some(&id),
                                        name,
                                    )
                                    .clicked()
                                {
                                    self.selected_object = Some(id);
                                    self.show_snapshot = false;
                                }
                            }
                        });
                    ui.separator();
                    if ui.button("|◀").on_hover_text("First frame").clicked() {
                        self.set_frame(0, ctx);
                    }
                    if ui.button("◀").on_hover_text("Previous frame").clicked() {
                        self.set_frame(self.frame.saturating_sub(1), ctx);
                    }
                    if ui
                        .button(if self.playing { "Ⅱ" } else { "▶" })
                        .on_hover_text("Play / pause")
                        .clicked()
                    {
                        self.playing = !self.playing;
                        self.last_tick = Instant::now();
                    }
                    if ui.button("▶|").on_hover_text("Next frame").clicked() {
                        self.set_frame((self.frame + 1) % self.frame_count(), ctx);
                    }
                    let mut frame = self.frame as f64;
                    let max = self.frame_count().saturating_sub(1) as f64;
                    if ui
                        .add(
                            egui::Slider::new(&mut frame, 0.0..=max)
                                .show_value(false)
                                .trailing_fill(true),
                        )
                        .changed()
                    {
                        self.set_frame(frame.round() as u64, ctx);
                    }
                    ui.label(
                        RichText::new(format!(
                            "{:04}/{:04}",
                            self.frame,
                            self.frame_count().saturating_sub(1)
                        ))
                        .monospace()
                        .color(INK),
                    );
                    if ui
                        .button("Data")
                        .on_hover_text("Object and frame data, plus diagnostics")
                        .clicked()
                    {
                        self.inspector_open = !self.inspector_open;
                    }
                    if ui
                        .button("↻")
                        .on_hover_text(format!(
                            "Reload · {} watched files",
                            self.source_files.len()
                        ))
                        .clicked()
                    {
                        self.reload();
                    }
                    let status_color = STATUS;
                    if ui
                        .add(
                            egui::Label::new(RichText::new("●").color(status_color))
                                .sense(egui::Sense::click()),
                        )
                        .on_hover_text(self.status.clone())
                        .clicked()
                    {
                        self.inspector_open = true;
                    }
                });
            });
    }

    fn data_window(&mut self, ctx: &egui::Context) {
        if !self.inspector_open {
            return;
        }
        let mut open = self.inspector_open;
        egui::Window::new("Data")
            .open(&mut open)
            .default_width(390.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.show_snapshot, false, "Object");
                    ui.selectable_value(&mut self.show_snapshot, true, "Frame");
                });
                ui.separator();
                if self.show_snapshot {
                    ui.label(
                        RichText::new(format!("Frame {}", self.frame))
                            .small()
                            .color(MUTED),
                    );
                    self.json_view(ui, self.snapshot.as_ref());
                } else if let Some(id) = &self.selected_object {
                    let value = self
                        .project_json
                        .as_ref()
                        .and_then(|json| json.get("objects"))
                        .and_then(|table| table.get(id));
                    self.json_view(ui, value);
                } else {
                    ui.label(RichText::new("Choose an object").color(MUTED));
                }
                if let Some(error) = &self.error {
                    ui.separator();
                    ui.label(RichText::new(error).monospace().color(INK));
                }
                if !self.lint_messages.is_empty() {
                    ui.collapsing(format!("Lint · {}", self.lint_messages.len()), |ui| {
                        for message in &self.lint_messages {
                            ui.label(RichText::new(message).monospace().size(10.0).color(INK));
                        }
                    });
                }
            });
        self.inspector_open = open;
    }

    fn json_view(&self, ui: &mut egui::Ui, value: Option<&serde_json::Value>) {
        let Some(value) = value else {
            ui.label(RichText::new("No data").color(MUTED));
            return;
        };
        let text =
            serde_json::to_string_pretty(value).unwrap_or_else(|_| "<cannot format JSON>".into());
        egui::ScrollArea::vertical()
            .id_salt("json-inspector")
            .show(ui, |ui| {
                ui.add(
                    egui::Label::new(RichText::new(text).monospace().size(11.0).color(INK))
                        .selectable(true),
                );
            });
    }

    fn preview(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        ui.painter().rect_filled(rect, 0, Color32::from_gray(20));
        if self.project.is_none() {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No preview",
                egui::FontId::proportional(14.0),
                MUTED,
            );
        } else {
            let image_size = Vec2::new(self.preview_size[0] as f32, self.preview_size[1] as f32);
            let scale = (rect.width() / image_size.x).min(rect.height() / image_size.y);
            let fitted = image_size * scale;
            let image_rect = egui::Rect::from_center_size(rect.center(), fitted);
            if let (Some(frame), Some(target_format)) = (&self.preview_gpu, self.gpu_format) {
                ui.painter().add(editor_gpu::callback_shape(
                    Arc::clone(frame),
                    image_rect,
                    target_format,
                    Arc::clone(&self.render_settings),
                    false,
                ));
                ui.allocate_rect(rect, egui::Sense::hover());
                return;
            }
        }
        ui.allocate_rect(rect, egui::Sense::hover());
    }
}

impl eframe::App for EditorApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let _ = frame;
        let ctx = ui.ctx().clone();
        self.playback_shortcuts(&ctx);
        self.poll_reload();
        self.tick(&ctx);
        self.header(ui, &ctx);
        self.sync_audio(&ctx, false);
        ui.separator();
        let available = ui.available_size();
        let timeline_height = (available.y * 0.40).clamp(190.0, 360.0);
        let preview_height = (available.y - timeline_height - 8.0).max(100.0);
        ui.allocate_ui_with_layout(
            Vec2::new(available.x, preview_height),
            egui::Layout::top_down(egui::Align::Min),
            |ui| self.preview(ui),
        );
        ui.separator();
        ui.allocate_ui_with_layout(
            Vec2::new(available.x, timeline_height),
            egui::Layout::top_down(egui::Align::Min),
            |ui| timeline_panel(ui, self, &ctx),
        );
        self.data_window(&ctx);
    }
}

fn timeline_panel(ui: &mut egui::Ui, app: &mut EditorApp, ctx: &egui::Context) {
    let total = app.frame_count().saturating_sub(1).max(1) as f32;
    let tracks = app.track_list();
    let width = ui.available_width();
    let label_width = 190.0_f32.min(width * 0.25);
    let timeline_width = (width - label_width).max(1.0);
    let (ruler, ruler_response) =
        ui.allocate_exact_size(Vec2::new(width, 22.0), egui::Sense::click());
    let ruler_painter = ui.painter_at(ruler);
    let x0 = ruler.left() + label_width;
    ruler_painter.rect_filled(ruler, 0, PANEL);
    for tick in 0..=10 {
        let x = x0 + timeline_width * tick as f32 / 10.0;
        let frame = (total * tick as f32 / 10.0).round() as u64;
        ruler_painter.line_segment(
            [
                egui::pos2(x, ruler.top() + 12.0),
                egui::pos2(x, ruler.bottom()),
            ],
            Stroke::new(1.0, Color32::from_gray(70)),
        );
        if tick < 10 {
            ruler_painter.text(
                egui::pos2(x + 3.0, ruler.top()),
                egui::Align2::LEFT_TOP,
                frame.to_string(),
                egui::FontId::monospace(9.0),
                MUTED,
            );
        }
    }
    if let Some(pointer) = ruler_response.interact_pointer_pos() {
        if ruler_response.clicked() && pointer.x >= x0 {
            app.set_frame(
                (((pointer.x - x0) / timeline_width) * total).round() as u64,
                ctx,
            );
        }
    }
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for (label, record, at, duration) in tracks {
                let (row, response) =
                    ui.allocate_exact_size(Vec2::new(width, 24.0), egui::Sense::click());
                let painter = ui.painter_at(row);
                let selected = app.selected_object.as_deref() == Some(record.as_str());
                painter.rect_filled(row, 0, if selected { Color32::from_gray(58) } else { BG });
                painter
                    .with_clip_rect(egui::Rect::from_min_max(
                        row.left_top(),
                        egui::pos2(x0 - 6.0, row.bottom()),
                    ))
                    .text(
                        egui::pos2(row.left() + 8.0, row.center().y),
                        egui::Align2::LEFT_CENTER,
                        label,
                        egui::FontId::proportional(11.0),
                        INK,
                    );
                let normalized_start = at.clamp(0.0, 1.0);
                let start = x0 + normalized_start * timeline_width;
                let remaining_width = (1.0 - normalized_start) * timeline_width;
                let bar_width = (duration.max(0.025) * timeline_width)
                    .max(3.0)
                    .min(remaining_width.max(0.0));
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(start, row.top() + 5.0),
                        Vec2::new(bar_width, 14.0),
                    ),
                    2,
                    Color32::from_gray(if selected { 150 } else { 105 }),
                );
                for tick in 0..=10 {
                    let x = x0 + timeline_width * tick as f32 / 10.0;
                    painter.line_segment(
                        [egui::pos2(x, row.top()), egui::pos2(x, row.bottom())],
                        Stroke::new(1.0, Color32::from_gray(48)),
                    );
                }
                let play_x = x0 + (app.frame as f32 / total) * timeline_width;
                painter.line_segment(
                    [
                        egui::pos2(play_x, row.top()),
                        egui::pos2(play_x, row.bottom()),
                    ],
                    Stroke::new(1.5, Color32::from_gray(220)),
                );
                if response.clicked() {
                    if let Some(pointer) = response.interact_pointer_pos() {
                        if pointer.x < x0 {
                            if app
                                .project
                                .as_ref()
                                .is_some_and(|project| project.objects.contains_key(&record))
                            {
                                app.selected_object = Some(record);
                                app.show_snapshot = false;
                            }
                        } else {
                            app.set_frame(
                                (((pointer.x - x0) / timeline_width) * total).round() as u64,
                                ctx,
                            );
                        }
                    }
                }
            }
        });
}

impl EditorApp {
    fn track_list(&self) -> Vec<(String, String, f32, f32)> {
        let (Some(project), Some(composition)) = (&self.project, self.current_composition()) else {
            return Vec::new();
        };
        let mut output = Vec::new();
        let duration = rational_f32(&composition.duration).max(f32::EPSILON);
        for timeline_id in &composition.timelines {
            if let Some(timeline) = project.timelines.get(timeline_id) {
                for track in &timeline.tracks {
                    let object_name = project
                        .objects
                        .get(&track.target.record)
                        .map(|object| object.name.as_str())
                        .unwrap_or(track.target.record.as_str());
                    output.push((
                        format!("{object_name} · {}", track.target.slot),
                        track.target.record.clone(),
                        rational_f32(&track.at) / duration,
                        rational_f32(&track.duration) / duration,
                    ));
                }
            }
        }
        output
    }
}

fn rational_f32(value: &Rational) -> f32 {
    value
        .as_positive_or_zero()
        .map(|r| r.n as f32 / r.d as f32)
        .unwrap_or(0.0)
}

fn lint_project(project: &Project, project_path: &Path) -> Vec<String> {
    let root = project_root(project_path.to_string_lossy().as_ref());
    let mut messages = Vec::new();
    if project.compositions.is_empty() {
        messages.push("E_COMPOSITIONS_EMPTY: project has no compositions".into());
        return messages;
    }
    for (composition_id, composition) in &project.compositions {
        let count = match composition_frame_count(composition) {
            Ok(count) => count,
            Err(error) => {
                messages.push(format!("E_COMPOSITION_TIMING [{composition_id}]: {error}"));
                continue;
            }
        };
        let frames = if count <= 1 {
            vec![0]
        } else {
            vec![0, count - 1]
        };
        for frame in frames {
            if let Err(error) = evaluation::sample(project, &root, composition_id, frame) {
                messages.push(format!(
                    "E_SCENE_UNSUPPORTED [{composition_id}, frame {frame}]: {error}"
                ));
            }
        }
    }
    messages
}
