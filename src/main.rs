#![recursion_limit = "256"]

use std::{
    collections::BTreeMap,
    fs,
    io::{self, IsTerminal, Write},
    path::{Component, Path, PathBuf},
    process::{Child, ChildStdin, Command, ExitCode, Stdio},
    time::Instant,
};

use clap::{Args, Parser, Subcommand};

mod animation;
mod audio;
mod camera;
mod deformation;
mod editor;
mod editor_gpu;
mod evaluation;
mod gpu_abi;
mod gpu_export;
mod graph;
mod imports;
mod live;
mod model;
mod quality;
mod renderer;
mod scene_asset;
mod typography;
mod validation;

use model::*;

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum MsaaSetting {
    #[value(name = "off", alias = "1")]
    Off,
    #[value(name = "2x", alias = "2")]
    X2,
    #[value(name = "4x", alias = "4")]
    X4,
}

impl MsaaSetting {
    fn sample_count(self) -> u16 {
        match self {
            Self::Off => 1,
            Self::X2 => 2,
            Self::X4 => 4,
        }
    }
}

#[derive(Debug, Args)]
struct RendererArgs {
    /// Color blending behavior.
    #[arg(long, value_enum, default_value = "alpha")]
    blend: editor_gpu::BlendMode,
    /// Depth testing and writing behavior.
    #[arg(long, value_enum, default_value = "test-write")]
    depth: editor_gpu::DepthMode,
    /// Triangle face culling.
    #[arg(long, value_enum, default_value = "none")]
    cull: editor_gpu::CullMode,
    /// WGSL file defining `fn mogf_surface(input: VertexOut) -> vec4<f32>`.
    #[arg(long)]
    shader: Option<PathBuf>,
    /// WGSL file defining `fn mogf_vertex(input: VertexOut) -> VertexOut`.
    #[arg(long)]
    vertex_shader: Option<PathBuf>,
    /// WGSL file defining `fn mogf_effect(uv, color, resolution) -> vec4<f32>` for a camera postprocess pass.
    #[arg(long)]
    effect_shader: Option<PathBuf>,
}

impl RendererArgs {
    fn load(
        self,
        profile: quality::RenderProfile,
        msaa: u16,
    ) -> Result<editor_gpu::RenderSettings, String> {
        let source = self
            .shader
            .map(|path| {
                fs::read_to_string(&path)
                    .map_err(|error| format!("cannot read WGSL shader {}: {error}", path.display()))
            })
            .transpose()?;
        let vertex_source = self
            .vertex_shader
            .map(|path| {
                fs::read_to_string(&path).map_err(|error| {
                    format!("cannot read WGSL vertex shader {}: {error}", path.display())
                })
            })
            .transpose()?;
        let effect_source = self
            .effect_shader
            .map(|path| {
                fs::read_to_string(&path).map_err(|error| {
                    format!("cannot read WGSL effect shader {}: {error}", path.display())
                })
            })
            .transpose()?;
        editor_gpu::RenderSettings::new(
            msaa,
            self.blend,
            self.depth,
            self.cull,
            source,
            vertex_source,
            effect_source,
            profile,
        )
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "mogf-mvp",
    version,
    about = "Evaluate and render bounded MOGF project documents",
    after_help = "When COMMAND is omitted, PROJECT renders the default composition to ./renders/. Use --composition-id ID to select another composition.",
    arg_required_else_help = true,
    propagate_version = true
)]
struct Cli {
    /// Print one JSON document to stdout for command results.
    #[arg(long, global = true)]
    json: bool,
    /// Select the composition. Defaults to `main`, or the only composition when `main` is absent.
    #[arg(long, global = true)]
    composition_id: Option<String>,
    /// Engine rendering profile (default: final).
    #[arg(long, value_enum, global = true, conflicts_with = "preview")]
    profile: Option<quality::RenderProfile>,
    /// Shorthand for `--profile preview`.
    #[arg(long, global = true)]
    preview: bool,
    /// Multisample antialiasing: off, 2x, or 4x (numeric aliases: 1, 2, 4).
    #[arg(long, global = true, value_enum, default_value = "4x")]
    msaa: MsaaSetting,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Check Rust and WGSL declarations against the embedded GPU ABI schema.
    AbiCheck,
    /// Evaluate one composition frame and print its JSON snapshot.
    Sample {
        /// Path to a mogf.project JSON file.
        project: String,
        /// Zero-based frame index.
        frame: u64,
    },
    /// Measure the camera-aligned view plane in world units at a camera-space distance.
    MeasurePlane {
        /// Path to a mogf.project JSON file.
        project: String,
        /// Zero-based frame index, including animated camera state.
        frame: u64,
        /// Positive distance in camera-space units.
        distance: f64,
    },
    /// Render one or more frames to PPM images.
    Frame {
        /// Path to a mogf.project JSON file.
        project: String,
        /// Zero-based frame indices to render.
        #[arg(required = true, num_args = 1..)]
        frames: Vec<u64>,
        /// Directory for frame_XXXX.ppm output files.
        #[arg(short, long, default_value = "frames")]
        output: String,
    },
    /// Render evenly spaced frames as one labelled PPM contact sheet.
    Strip {
        /// Path to a mogf.project JSON file.
        project: String,
        /// Number of frames to sample across the composition.
        #[arg(short = 'n', long, default_value_t = 12)]
        count: usize,
        /// Number of frame columns in the sheet.
        #[arg(long, default_value_t = 4)]
        columns: usize,
        /// Output PPM file.
        #[arg(short, long, default_value = "strip.ppm")]
        output: String,
    },
    /// Render an end-exclusive frame range to PPM files or an FFmpeg MP4.
    #[command(name = "render", visible_alias = "render-range")]
    RenderRange {
        /// Path to a mogf.project JSON file.
        project: String,
        /// First frame to render, inclusive (defaults to 0).
        #[arg(requires = "end_frame_exclusive")]
        start_frame: Option<u64>,
        /// Frame after the last frame to render (defaults to composition end).
        #[arg(requires = "start_frame")]
        end_frame_exclusive: Option<u64>,
        /// Output directory, accepted positionally for compatibility.
        output_dir: Option<String>,
        /// Directory for frame_XXXX.ppm output files.
        #[arg(short, long)]
        output: Option<String>,
        /// Stream rendered frames to FFmpeg and encode an H.264 MP4 without storing PPM files.
        #[arg(long, value_name = "FILE.mp4", conflicts_with_all = ["output", "output_dir"])]
        video: Option<String>,
    },
    /// Mix the selected composition's audio timeline to a stereo PCM WAV file.
    RenderAudio {
        /// Path to a mogf.project JSON file.
        project: String,
        /// Output WAV file.
        #[arg(short, long, default_value = "renders/audio.wav")]
        output: String,
    },
    /// List composition dimensions, frame rate, duration, and frame count.
    Timeline {
        /// Path to a mogf.project JSON file.
        project: String,
    },
    /// Evaluate selected frames to check project and renderer support without writing images.
    Inspect {
        /// Path to a mogf.project JSON file.
        project: String,
        /// Frames to inspect; defaults to frame 0 and the last frame.
        frames: Vec<u64>,
        /// Check every frame in the composition (up to the MVP range limit).
        #[arg(long, conflicts_with = "frames")]
        all_frames: bool,
    },
    /// Check JSON decoding and supported scene data, reporting useful locations.
    Lint {
        /// Path to a mogf.project JSON file.
        project: String,
        /// Evaluate every composition frame (limited to 3000 frames per composition).
        #[arg(long)]
        all_frames: bool,
    },
    /// Open a continuously playing GPU-rendered scene and reload edits in place.
    Live {
        /// Path to a mogf.project JSON file.
        project: String,
        /// Print measured UI repaint rate and CPU scene-preparation time once per second.
        #[arg(long)]
        stats: bool,
        #[command(flatten)]
        renderer: RendererArgs,
    },
    /// Open the interactive composition, timeline, and data editor.
    Edit {
        /// Path to a mogf.project JSON file.
        project: String,
        #[command(flatten)]
        renderer: RendererArgs,
    },
}

fn fail(message: impl std::fmt::Display) -> ExitCode {
    eprintln!("mogf-mvp: {message}");
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let mut args: Vec<_> = std::env::args_os().collect();
    let known_commands = [
        "sample",
        "measure-plane",
        "frame",
        "strip",
        "render",
        "render-range",
        "render-audio",
        "timeline",
        "inspect",
        "lint",
        "live",
        "edit",
        "help",
        "abi-check",
    ];
    let mut index = 1;
    let mut command_position = None;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        if arg == "--profile" || arg == "--composition-id" || arg == "--msaa" {
            index += 2;
            continue;
        }
        if arg.starts_with('-') {
            index += 1;
            continue;
        }
        command_position = Some((index, arg.into_owned()));
        break;
    }
    if let Some((index, command_or_project)) = command_position {
        if !known_commands.contains(&command_or_project.as_str()) {
            args.insert(index, "render".into());
        }
    }
    let cli = Cli::parse_from(args);
    let json = cli.json;
    let composition_id = cli.composition_id.as_deref();
    let profile = if cli.preview {
        quality::RenderProfile::Preview
    } else {
        cli.profile.unwrap_or_default()
    };
    let msaa = cli.msaa.sample_count();
    match cli.command {
        Commands::AbiCheck => run_abi_check(json),
        Commands::Sample { project, frame } => run_sample(&project, composition_id, frame, profile),
        Commands::MeasurePlane {
            project,
            frame,
            distance,
        } => run_measure_plane(&project, composition_id, frame, distance, json, profile),
        Commands::Frame {
            project,
            frames,
            output,
        } => run_frames(
            &project,
            composition_id,
            &frames,
            &output,
            json,
            profile,
            msaa,
        ),
        Commands::Strip {
            project,
            count,
            columns,
            output,
        } => run_strip(
            &project,
            composition_id,
            count,
            columns,
            &output,
            json,
            profile,
            msaa,
        ),
        Commands::RenderRange {
            project,
            start_frame,
            end_frame_exclusive,
            output_dir,
            output,
            video,
        } => run_render_range(
            &project,
            composition_id,
            start_frame,
            end_frame_exclusive,
            output.as_deref().or(output_dir.as_deref()),
            video.as_deref(),
            json,
            profile,
            msaa,
        ),
        Commands::RenderAudio { project, output } => {
            run_render_audio(&project, composition_id, &output, json)
        }
        Commands::Timeline { project } => run_timeline(&project, composition_id, json),
        Commands::Inspect {
            project,
            frames,
            all_frames,
        } => run_inspect(&project, composition_id, &frames, all_frames, json, profile),
        Commands::Lint {
            project,
            all_frames,
        } => run_lint(&project, composition_id, all_frames, json),
        Commands::Live {
            project,
            stats,
            renderer,
        } => match renderer.load(profile, msaa).and_then(|settings| {
            live::run(project, composition_id.map(str::to_owned), settings, stats)
        }) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => fail(error),
        },
        Commands::Edit { project, renderer } => match renderer
            .load(profile, msaa)
            .and_then(|settings| editor::run(project, composition_id.map(str::to_owned), settings))
        {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => fail(error),
        },
    }
}

fn run_abi_check(json: bool) -> ExitCode {
    let source = "fn mogf_surface(input: VertexOut) -> vec4<f32> { return input.color; }";
    match editor_gpu::validate_surface_shader(source) {
        Ok(()) if json => {
            println!("{{\"abi\":\"mogf.surface\",\"version\":1,\"valid\":true}}");
            ExitCode::SUCCESS
        }
        Ok(()) => {
            println!("mogf.surface ABI v1: valid");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("mogf.surface ABI v1: {error}");
            ExitCode::FAILURE
        }
    }
}

fn read_project(path: &str) -> Result<Project, String> {
    imports::load_project(Path::new(path))
        .map(|loaded| loaded.project)
        .map_err(|error| error.to_string())
}

fn select_composition(project: &Project, requested: Option<&str>) -> Result<String, String> {
    if let Some(id) = requested {
        return project
            .compositions
            .contains_key(id)
            .then(|| id.to_owned())
            .ok_or_else(|| format!("unknown composition {id:?}"));
    }
    if project.compositions.contains_key("main") {
        return Ok("main".to_owned());
    }
    if let [id] = project.compositions.keys().collect::<Vec<_>>().as_slice() {
        return Ok((*id).clone());
    }
    Err("composition is ambiguous; pass --composition-id ID".to_owned())
}

fn run_sample(
    project_path: &str,
    requested_composition: Option<&str>,
    frame_index: u64,
    profile: quality::RenderProfile,
) -> ExitCode {
    let project = match read_project(project_path) {
        Ok(project) => project,
        Err(error) => return fail(error),
    };
    let composition_id = match select_composition(&project, requested_composition) {
        Ok(id) => id,
        Err(error) => return fail(error),
    };
    let root = project_root(project_path);
    match evaluation::sample_with_profile(&project, &root, &composition_id, frame_index, profile) {
        Ok(frame) => match serde_json::to_string_pretty(&frame) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => fail(format!("cannot encode frame: {error}")),
        },
        Err(error) => fail(error),
    }
}

fn run_measure_plane(
    project_path: &str,
    requested_composition: Option<&str>,
    frame_index: u64,
    distance: f64,
    json: bool,
    profile: quality::RenderProfile,
) -> ExitCode {
    let project = match read_project(project_path) {
        Ok(project) => project,
        Err(error) => return fail(error),
    };
    let composition_id = match select_composition(&project, requested_composition) {
        Ok(id) => id,
        Err(error) => return fail(error),
    };
    let Some(composition) = project.compositions.get(&composition_id) else {
        return fail(format!("unknown composition {composition_id:?}"));
    };
    let root = project_root(project_path);
    let frame = match evaluation::sample_with_profile(
        &project,
        &root,
        &composition_id,
        frame_index,
        profile,
    ) {
        Ok(frame) => frame,
        Err(error) => return fail(error),
    };
    let aspect = f64::from(composition.size[0]) / f64::from(composition.size[1].max(1));
    let plane = match camera::measurement_plane(&frame.camera, aspect, composition.size, distance) {
        Ok(plane) => plane,
        Err(error) => return fail(error),
    };
    if json {
        println!(
            "{}",
            serde_json::json!({
                "composition": composition_id,
                "frame": frame_index,
                "projection": frame.camera.projection,
                "measurement_plane": plane,
            })
        );
    } else {
        println!(
            "camera plane at distance {}: {} × {} camera-space units; {} × {} units per pixel",
            plane.distance,
            plane.size[0],
            plane.size[1],
            plane.units_per_pixel[0],
            plane.units_per_pixel[1]
        );
    }
    ExitCode::SUCCESS
}

struct RenderProgress {
    enabled: bool,
    total: u64,
    started: Instant,
    finished: bool,
}

impl RenderProgress {
    fn new(total: u64, json: bool) -> Self {
        Self {
            enabled: !json && io::stderr().is_terminal(),
            total,
            started: Instant::now(),
            finished: false,
        }
    }

    fn update(&self, completed: u64, frame: u64) {
        if !self.enabled {
            return;
        }
        let ratio = completed as f64 / self.total.max(1) as f64;
        let width = 24usize;
        let filled = (ratio * width as f64).round() as usize;
        let bar = format!("{}{}", "█".repeat(filled), "░".repeat(width - filled));
        let elapsed = self.started.elapsed().as_secs();
        let eta = if completed > 0 {
            let seconds_per_frame = elapsed as f64 / completed as f64;
            format_duration((seconds_per_frame * (self.total - completed) as f64) as u64)
        } else {
            "--:--".to_owned()
        };
        eprint!(
            "\rrender [{bar}] {:>4}/{:<4} {:>3}% frame {frame} elapsed {} ETA {eta}",
            completed,
            self.total,
            (ratio * 100.0).round() as u32,
            format_duration(elapsed),
        );
        let _ = io::stderr().flush();
    }

    fn finish(mut self) {
        if self.enabled {
            eprintln!();
        }
        self.finished = true;
    }
}

impl Drop for RenderProgress {
    fn drop(&mut self) {
        if self.enabled && !self.finished {
            eprintln!();
        }
    }
}

fn format_duration(seconds: u64) -> String {
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

struct VideoEncoder {
    child: Child,
    input: Option<ChildStdin>,
    output: PathBuf,
    finished: bool,
}

impl VideoEncoder {
    fn spawn(
        path: &str,
        fps: &Rational,
        size: [u32; 2],
        audio_path: Option<&Path>,
    ) -> Result<Self, String> {
        if Path::new(path).exists() {
            return Err(format!("video output already exists: {path}"));
        }
        if Path::new(path)
            .extension()
            .and_then(|extension| extension.to_str())
            .is_none_or(|extension| !extension.eq_ignore_ascii_case("mp4"))
        {
            return Err("--video output must use the .mp4 extension".to_owned());
        }
        let fps = fps.as_positive()?;
        let mut command = Command::new("ffmpeg");
        command
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgba",
                "-video_size",
            ])
            .arg(format!("{}x{}", size[0], size[1]))
            .arg("-framerate")
            .arg(format!("{}/{}", fps.n, fps.d))
            .args(["-i", "pipe:0"]);
        if let Some(audio_path) = audio_path {
            command
                .arg("-i")
                .arg(audio_path)
                .args(["-map", "0:v:0", "-map", "1:a:0"]);
        } else {
            command.arg("-an");
        }
        command
            .args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
            .args([
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            ]);
        if audio_path.is_some() {
            command.args(["-c:a", "aac", "-b:a", "192k", "-shortest"]);
        }
        let mut child = command
            .args(["-movflags", "+faststart"])
            .arg(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                format!(
                    "cannot start FFmpeg: {error}; install ffmpeg with libx264 and put it on PATH"
                )
            })?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| "FFmpeg did not open its input pipe".to_owned())?;
        Ok(Self {
            child,
            input: Some(input),
            output: PathBuf::from(path),
            finished: false,
        })
    }

    fn write_frame(&mut self, ppm: &[u8]) -> Result<(), String> {
        self.input
            .as_mut()
            .ok_or_else(|| "FFmpeg input pipe is closed".to_owned())?
            .write_all(ppm)
            .map_err(|error| format!("cannot stream frame to FFmpeg: {error}"))
    }

    fn finish(mut self) -> Result<(), String> {
        self.input.take();
        let status = self
            .child
            .wait()
            .map_err(|error| format!("cannot wait for FFmpeg: {error}"))?;
        if status.success() {
            self.finished = true;
            Ok(())
        } else {
            Err(format!("FFmpeg exited with {status}"))
        }
    }
}

struct TemporaryAudio(PathBuf);

impl Drop for TemporaryAudio {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn write_temporary_audio(samples: &[f32]) -> Result<TemporaryAudio, String> {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| format!("system clock is before the Unix epoch: {error}"))?
        .as_nanos();
    let path = std::env::temp_dir().join(format!("mogf-audio-{}-{nonce}.wav", std::process::id()));
    audio::write_wav(&path, samples)?;
    Ok(TemporaryAudio(path))
}

impl Drop for VideoEncoder {
    fn drop(&mut self) {
        if !self.finished {
            self.input.take();
            let _ = self.child.kill();
            let _ = self.child.wait();
            let _ = fs::remove_file(&self.output);
        }
    }
}

fn run_render_range(
    project_path: &str,
    requested_composition: Option<&str>,
    start: Option<u64>,
    end: Option<u64>,
    output: Option<&str>,
    video: Option<&str>,
    json: bool,
    profile: quality::RenderProfile,
    msaa: u16,
) -> ExitCode {
    let project = match read_project(project_path) {
        Ok(project) => project,
        Err(error) => return fail(error),
    };
    let composition_id = match select_composition(&project, requested_composition) {
        Ok(id) => id,
        Err(error) => return fail(error),
    };
    let composition = match project.compositions.get(&composition_id) {
        Some(value) => value,
        None => return fail(format!("unknown composition {composition_id:?}")),
    };
    let start = start.unwrap_or(0);
    let end = match end {
        Some(end) => end,
        None => match composition_frame_count(composition) {
            Ok(count) => count,
            Err(error) => return fail(error),
        },
    };
    let output_dir = output.unwrap_or("renders");
    if end <= start {
        return fail("frame range must be nonempty and end-exclusive");
    }
    if end - start > 3000 {
        return fail("MVP render range is limited to 3000 frames");
    }
    let root = project_root(project_path);
    if composition.size[0] > 4096
        || composition.size[1] > 4096
        || u64::from(composition.size[0]) * u64::from(composition.size[1]) > 16_777_216
    {
        return fail("MVP render dimensions exceed 4096 per side or 16,777,216 pixels");
    }
    let fps = match composition.fps.as_positive() {
        Ok(fps) => fps,
        Err(error) => return fail(error),
    };
    let audio_temp = if video.is_some() && !composition.audio.is_empty() {
        let range_start = start as f64 * fps.d as f64 / fps.n as f64;
        let range_duration = (end - start) as f64 * fps.d as f64 / fps.n as f64;
        let samples = match audio::mix_composition_range(
            &project,
            &root,
            composition,
            range_start,
            range_duration,
        ) {
            Ok(samples) => samples,
            Err(error) => return fail(error),
        };
        match write_temporary_audio(&samples) {
            Ok(file) => Some(file),
            Err(error) => return fail(error),
        }
    } else {
        None
    };
    let mut encoder = if let Some(path) = video {
        match VideoEncoder::spawn(
            path,
            &composition.fps,
            composition.size,
            audio_temp.as_ref().map(|file| file.0.as_path()),
        ) {
            Ok(encoder) => Some(encoder),
            Err(error) => return fail(error),
        }
    } else {
        if let Err(error) = fs::create_dir_all(output_dir) {
            return fail(format!("cannot create output directory: {error}"));
        }
        None
    };
    let progress = RenderProgress::new(end - start, json);
    for frame_number in start..end {
        let frame = match evaluation::sample_with_profile(
            &project,
            &root,
            &composition_id,
            frame_number,
            profile,
        ) {
            Ok(frame) => frame,
            Err(error) => return fail(error),
        };
        if let Some(encoder) = &mut encoder {
            let rgba = match gpu_export::render_rgba8(&frame, composition.size, msaa) {
                Ok(rgba) => rgba,
                Err(error) => return fail(error),
            };
            if let Err(error) = encoder.write_frame(&rgba) {
                return fail(error);
            }
        } else {
            let ppm = match gpu_export::render_ppm(&frame, composition.size, msaa) {
                Ok(ppm) => ppm,
                Err(error) => return fail(error),
            };
            let path = Path::new(output_dir).join(format!("frame_{frame_number:04}.ppm"));
            if let Err(error) = fs::write(&path, ppm) {
                return fail(format!("cannot write {}: {error}", path.display()));
            }
        }
        progress.update(frame_number - start + 1, frame_number);
    }
    if let Some(encoder) = encoder {
        if let Err(error) = encoder.finish() {
            return fail(error);
        }
    }
    progress.finish();
    if json {
        let output = video.unwrap_or(output_dir);
        let format = if video.is_some() { "mp4" } else { "ppm" };
        println!(
            "{}",
            serde_json::json!({"rendered": end - start, "start": start, "end_exclusive": end, "output": output, "format": format})
        );
    } else {
        if let Some(video) = video {
            println!("rendered frames {start}..{end} to {video}");
        } else {
            println!("rendered frames {start}..{end} to {output_dir}");
        }
    }
    ExitCode::SUCCESS
}

fn run_render_audio(
    project_path: &str,
    requested_composition: Option<&str>,
    output: &str,
    json: bool,
) -> ExitCode {
    if Path::new(output)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_none_or(|extension| !extension.eq_ignore_ascii_case("wav"))
    {
        return fail("audio output must use the .wav extension");
    }
    let project = match read_project(project_path) {
        Ok(project) => project,
        Err(error) => return fail(error),
    };
    let composition_id = match select_composition(&project, requested_composition) {
        Ok(id) => id,
        Err(error) => return fail(error),
    };
    let Some(composition) = project.compositions.get(&composition_id) else {
        return fail(format!("unknown composition {composition_id:?}"));
    };
    let root = project_root(project_path);
    let samples = match audio::mix_composition(&project, &root, composition) {
        Ok(samples) => samples,
        Err(error) => return fail(error),
    };
    let output_path = Path::new(output);
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        && let Err(error) = fs::create_dir_all(parent)
    {
        return fail(format!("cannot create audio output directory: {error}"));
    }
    if let Err(error) = audio::write_wav(output_path, &samples) {
        return fail(error);
    }
    let frames = samples.len() / 2;
    if json {
        println!(
            "{}",
            serde_json::json!({
                "composition": composition_id,
                "output": output,
                "format": "wav",
                "sample_rate": audio::OUTPUT_SAMPLE_RATE,
                "channels": 2,
                "frames": frames,
            })
        );
    } else {
        println!(
            "rendered {} stereo frames at {} Hz to {output}",
            frames,
            audio::OUTPUT_SAMPLE_RATE
        );
    }
    ExitCode::SUCCESS
}

fn run_frames(
    project_path: &str,
    requested_composition: Option<&str>,
    frames: &[u64],
    output_dir: &str,
    json: bool,
    profile: quality::RenderProfile,
    msaa: u16,
) -> ExitCode {
    let project = match read_project(project_path) {
        Ok(project) => project,
        Err(error) => return fail(error),
    };
    let composition_id = match select_composition(&project, requested_composition) {
        Ok(id) => id,
        Err(error) => return fail(error),
    };
    let composition = match project.compositions.get(&composition_id) {
        Some(composition) => composition,
        None => return fail(format!("unknown composition {composition_id:?}")),
    };
    if composition.size[0] > 4096
        || composition.size[1] > 4096
        || u64::from(composition.size[0]) * u64::from(composition.size[1]) > 16_777_216
    {
        return fail("MVP render dimensions exceed 4096 per side or 16,777,216 pixels");
    }
    if let Err(error) = fs::create_dir_all(output_dir) {
        return fail(format!("cannot create output directory: {error}"));
    }
    let root = project_root(project_path);
    let progress = RenderProgress::new(frames.len() as u64, json);
    for (index, &frame_number) in frames.iter().enumerate() {
        let frame = match evaluation::sample_with_profile(
            &project,
            &root,
            &composition_id,
            frame_number,
            profile,
        ) {
            Ok(frame) => frame,
            Err(error) => return fail(format!("frame {frame_number}: {error}")),
        };
        let ppm = match gpu_export::render_ppm(&frame, composition.size, msaa) {
            Ok(ppm) => ppm,
            Err(error) => return fail(error),
        };
        let path = Path::new(output_dir).join(format!("frame_{frame_number:04}.ppm"));
        if let Err(error) = fs::write(&path, ppm) {
            return fail(format!("cannot write {}: {error}", path.display()));
        }
        progress.update(index as u64 + 1, frame_number);
    }
    progress.finish();
    if json {
        let files: Vec<_> = frames
            .iter()
            .map(|frame| format!("{output_dir}/frame_{frame:04}.ppm"))
            .collect();
        println!(
            "{}",
            serde_json::json!({"rendered": files.len(), "files": files})
        );
    } else {
        println!("rendered {} frame(s) to {output_dir}", frames.len());
    }
    ExitCode::SUCCESS
}

fn run_strip(
    project_path: &str,
    requested_composition: Option<&str>,
    count: usize,
    columns: usize,
    output: &str,
    json: bool,
    profile: quality::RenderProfile,
    msaa: u16,
) -> ExitCode {
    if count == 0 || count > 3000 || columns == 0 || columns > count {
        return fail("strip count must be 1..=3000 and columns must be 1..=count");
    }
    let project = match read_project(project_path) {
        Ok(project) => project,
        Err(error) => return fail(error),
    };
    let composition_id = match select_composition(&project, requested_composition) {
        Ok(id) => id,
        Err(error) => return fail(error),
    };
    let composition = match project.compositions.get(&composition_id) {
        Some(composition) => composition,
        None => return fail(format!("unknown composition {composition_id:?}")),
    };
    let last = match composition_frame_count(composition) {
        Ok(count) => count - 1,
        Err(error) => return fail(error),
    };
    let frames: Vec<u64> = if count == 1 {
        vec![0]
    } else {
        (0..count)
            .map(|index| ((u128::from(last) * index as u128) / (count - 1) as u128) as u64)
            .collect()
    };
    let root = project_root(project_path);
    let mut rendered = Vec::with_capacity(count);
    let progress = RenderProgress::new(count as u64, json);
    for (index, frame_number) in frames.iter().enumerate() {
        let frame = match evaluation::sample_with_profile(
            &project,
            &root,
            &composition_id,
            *frame_number,
            profile,
        ) {
            Ok(frame) => frame,
            Err(error) => return fail(format!("frame {frame_number}: {error}")),
        };
        match gpu_export::render_ppm(&frame, composition.size, msaa) {
            Ok(ppm) => rendered.push(ppm),
            Err(error) => return fail(error),
        }
        progress.update(index as u64 + 1, *frame_number);
    }
    progress.finish();
    let sheet = match renderer::contact_sheet_ppm(&rendered, composition.size, columns) {
        Ok(sheet) => sheet,
        Err(error) => return fail(error),
    };
    if let Err(error) = fs::write(output, sheet) {
        return fail(format!("cannot write {output}: {error}"));
    }
    if json {
        println!(
            "{}",
            serde_json::json!({"rendered": frames.len(), "frames": frames, "output": output})
        );
    } else {
        println!("wrote {} labelled frames to {output}", frames.len());
    }
    ExitCode::SUCCESS
}

fn run_timeline(project_path: &str, selected: Option<&str>, json: bool) -> ExitCode {
    let project = match read_project(project_path) {
        Ok(project) => project,
        Err(error) => return fail(error),
    };
    let rows = project
        .compositions
        .iter()
        .filter(|(id, _)| selected.is_none_or(|value| value == id.as_str()))
        .map(|(id, composition)| {
            let fps = composition.fps.as_positive()?;
            let duration = composition.duration.as_positive()?;
            let count = composition_frame_count(composition)?;
            Ok(serde_json::json!({
                "composition": id,
                "size": composition.size,
                "fps": format!("{}/{}", fps.n, fps.d),
                "duration": format!("{}/{}", duration.n, duration.d),
                "frames": count.to_string(),
                "output_profile": composition.output_profile,
            }))
        })
        .collect::<Result<Vec<_>, String>>();
    let rows = match rows {
        Ok(rows) if !rows.is_empty() => rows,
        Ok(_) => return fail("no matching compositions"),
        Err(error) => return fail(error),
    };
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".into())
        );
    } else {
        for row in rows {
            println!(
                "{}  {}x{}  {} fps  {} frames  {}",
                row["composition"].as_str().unwrap_or("?"),
                row["size"][0],
                row["size"][1],
                row["fps"].as_str().unwrap_or("?"),
                row["frames"].as_str().unwrap_or("?"),
                row["output_profile"].as_str().unwrap_or("?")
            );
        }
    }
    ExitCode::SUCCESS
}

fn run_inspect(
    project_path: &str,
    requested_composition: Option<&str>,
    frames: &[u64],
    all_frames: bool,
    json: bool,
    profile: quality::RenderProfile,
) -> ExitCode {
    let project = match read_project(project_path) {
        Ok(project) => project,
        Err(error) => return fail(error),
    };
    let composition_id = match select_composition(&project, requested_composition) {
        Ok(id) => id,
        Err(error) => return fail(error),
    };
    let composition = match project.compositions.get(&composition_id) {
        Some(composition) => composition,
        None => return fail(format!("unknown composition {composition_id:?}")),
    };
    let last = match composition_frame_count(composition) {
        Ok(count) => count - 1,
        Err(error) => return fail(error),
    };
    let checks: Vec<u64> = if all_frames {
        if last >= 3000 {
            return fail("--all-frames is limited to 3000 frames");
        }
        (0..=last).collect()
    } else if frames.is_empty() {
        if last == 0 { vec![0] } else { vec![0, last] }
    } else {
        frames.to_vec()
    };
    let root = project_root(project_path);
    let mut issues = Vec::new();
    for frame_number in &checks {
        if let Err(error) = evaluation::sample_with_profile(
            &project,
            &root,
            &composition_id,
            *frame_number,
            profile,
        ) {
            issues.push(serde_json::json!({"frame": frame_number, "error": error}));
        }
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({"composition": composition_id, "checked": checks.len(), "issues": issues})).unwrap_or_else(|_| "{}".into()));
    } else if issues.is_empty() {
        println!(
            "inspected {} frame(s) in {composition_id}: no issues",
            checks.len()
        );
    } else {
        for issue in &issues {
            eprintln!("frame {}: {}", issue["frame"], issue["error"]);
        }
        return ExitCode::from(2);
    }
    if issues.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    }
}

#[derive(serde::Serialize)]
struct LintDiagnostic {
    severity: &'static str,
    code: &'static str,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    column: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    composition: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frame: Option<u64>,
}

fn run_lint(project_path: &str, selected: Option<&str>, all_frames: bool, json: bool) -> ExitCode {
    let (project, scene_warnings): (Project, Vec<(String, String)>) =
        match imports::load_project(Path::new(project_path)) {
            Ok(loaded) => {
                let warnings = loaded
                    .project
                    .assets
                    .iter()
                    .flat_map(|(id, asset)| {
                        asset
                            .scene_warnings
                            .iter()
                            .map(move |warning| (id.clone(), warning.clone()))
                    })
                    .collect();
                (loaded.project, warnings)
            }
            Err(error) => {
                let diagnostic = LintDiagnostic {
                    severity: "error",
                    code: error.code,
                    message: error.message,
                    line: error.line,
                    column: error.column,
                    composition: None,
                    frame: None,
                };
                return print_lint(project_path, vec![diagnostic], json);
            }
        };
    let root = project_root(project_path);
    let composition_ids: Vec<_> = match selected {
        Some(id) if project.compositions.contains_key(id) => vec![id.to_owned()],
        Some(id) => {
            return print_lint(
                project_path,
                vec![LintDiagnostic {
                    severity: "error",
                    code: "E_COMPOSITION_MISSING",
                    message: format!("unknown composition {id:?}"),
                    line: None,
                    column: None,
                    composition: Some(id.to_owned()),
                    frame: None,
                }],
                json,
            );
        }
        None => project.compositions.keys().cloned().collect(),
    };
    let mut diagnostics = scene_warnings
        .into_iter()
        .map(|(asset, message)| LintDiagnostic {
            severity: "warning",
            code: "W_SCENE_CAPABILITY",
            message: format!("scene asset {asset:?}: {message}"),
            line: None,
            column: None,
            composition: None,
            frame: None,
        })
        .collect::<Vec<_>>();
    if composition_ids.is_empty() {
        diagnostics.push(LintDiagnostic {
            severity: "error",
            code: "E_COMPOSITIONS_EMPTY",
            message: "project has no compositions".to_owned(),
            line: None,
            column: None,
            composition: None,
            frame: None,
        });
    }
    for composition_id in composition_ids {
        let composition = &project.compositions[&composition_id];
        let count = match composition_frame_count(composition) {
            Ok(count) => count,
            Err(error) => {
                diagnostics.push(LintDiagnostic {
                    severity: "error",
                    code: "E_COMPOSITION_TIMING",
                    message: error,
                    line: None,
                    column: None,
                    composition: Some(composition_id),
                    frame: None,
                });
                continue;
            }
        };
        let frames: Vec<u64> = if all_frames {
            if count > 3000 {
                diagnostics.push(LintDiagnostic {
                    severity: "error",
                    code: "E_LINT_FRAME_LIMIT",
                    message: format!(
                        "composition has {count} frames; --all-frames is limited to 3000"
                    ),
                    line: None,
                    column: None,
                    composition: Some(composition_id),
                    frame: None,
                });
                continue;
            }
            (0..count).collect()
        } else {
            match count {
                0 => Vec::new(),
                1 => vec![0],
                _ => vec![0, count - 1],
            }
        };
        for frame in frames {
            if let Err(error) = evaluation::sample(&project, &root, &composition_id, frame) {
                diagnostics.push(LintDiagnostic {
                    severity: "error",
                    code: "E_SCENE_UNSUPPORTED",
                    message: error,
                    line: None,
                    column: None,
                    composition: Some(composition_id.clone()),
                    frame: Some(frame),
                });
            }
        }
    }
    print_lint(project_path, diagnostics, json)
}

fn print_lint(project_path: &str, diagnostics: Vec<LintDiagnostic>, json: bool) -> ExitCode {
    let has_errors = diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == "error");
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "project": project_path,
                "ok": !has_errors,
                "diagnostics": diagnostics,
            }))
            .unwrap_or_else(|_| "{}".to_owned())
        );
    } else if diagnostics.is_empty() {
        println!("{}: no issues", project_path);
    } else {
        for diagnostic in &diagnostics {
            let location = match (diagnostic.line, diagnostic.column) {
                (Some(line), Some(column)) if line > 0 => {
                    format!("{}:{line}:{column}: ", project_path)
                }
                _ => format!("{}: ", project_path),
            };
            let target = match (&diagnostic.composition, diagnostic.frame) {
                (Some(composition), Some(frame)) => format!(" [{composition} frame {frame}]"),
                (Some(composition), None) => format!(" [{composition}]"),
                _ => String::new(),
            };
            eprintln!(
                "{}{}{}{}",
                location,
                diagnostic.severity,
                target,
                format!(" [{}] {}", diagnostic.code, diagnostic.message)
            );
        }
    }
    if has_errors {
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    }
}

pub(crate) fn project_root(path: &str) -> std::path::PathBuf {
    Path::new(path)
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

pub(crate) fn composition_frame_count(composition: &Composition) -> Result<u64, String> {
    let fps = composition.fps.as_positive()?;
    let duration = composition.duration.as_positive()?;
    let numerator = i128::from(duration.n)
        .checked_mul(i128::from(fps.n))
        .ok_or("composition frame count overflow")?;
    let denominator = i128::from(duration.d)
        .checked_mul(i128::from(fps.d))
        .ok_or("composition frame rate denominator overflow")?;
    let count = (numerator - 1) / denominator + 1;
    u64::try_from(count).map_err(|_| "composition frame count exceeds u64".to_owned())
}

fn validate_assets(assets: &BTreeMap<String, AssetRecord>) -> Result<(), String> {
    for (id, asset) in assets {
        require(
            matches!(
                asset.kind.as_str(),
                "font" | "shader" | "image" | "audio" | "scene"
            ),
            format!("asset {id:?} has unsupported kind {:?}", asset.kind),
        )?;
        let uri = Path::new(&asset.uri);
        require(
            !uri.is_absolute()
                && uri
                    .components()
                    .all(|part| matches!(part, Component::Normal(_))),
            format!("asset {id:?} URI must be a relative path without traversal"),
        )?;
    }
    Ok(())
}

fn resolve_asset(root: &Path, asset: &AssetRecord) -> Result<PathBuf, String> {
    let relative = Path::new(&asset.uri);
    require(
        !relative.is_absolute()
            && relative
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "font asset URI must be a relative path without traversal",
    )?;
    let source_root = asset.source_root.as_deref().unwrap_or(root);
    let path = source_root.join(relative);
    require(
        path.is_file(),
        format!("font asset does not exist: {}", path.display()),
    )?;
    Ok(path)
}

fn normalized(q: [f64; 4]) -> Result<[f64; 4], String> {
    let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    require(
        norm.is_finite() && norm > 1e-12,
        "rotation quaternion must be finite and nonzero",
    )?;
    Ok([q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm])
}

type Mat4 = [[f64; 4]; 4];

fn identity() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn matrix(t: &Transform) -> Mat4 {
    let [x, y, z, w] = t.rotation;
    let [sx, sy, sz] = t.scale;
    let [tx, ty, tz] = t.translation;
    [
        [
            (1.0 - 2.0 * (y * y + z * z)) * sx,
            (2.0 * (x * y - z * w)) * sy,
            (2.0 * (x * z + y * w)) * sz,
            tx,
        ],
        [
            (2.0 * (x * y + z * w)) * sx,
            (1.0 - 2.0 * (x * x + z * z)) * sy,
            (2.0 * (y * z - x * w)) * sz,
            ty,
        ],
        [
            (2.0 * (x * z - y * w)) * sx,
            (2.0 * (y * z + x * w)) * sy,
            (1.0 - 2.0 * (x * x + y * y)) * sz,
            tz,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn multiply(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            for k in 0..4 {
                out[row][col] += a[row][k] * b[k][col];
            }
        }
    }
    out
}

fn require(ok: bool, message: impl std::fmt::Display) -> Result<(), String> {
    if ok { Ok(()) } else { Err(message.to_string()) }
}
