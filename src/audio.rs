//! Composition-aligned local WAV playback for interactive hosts only.
//! Frame evaluation never opens an audio device or advances playback.
use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Duration,
};

use crate::{model::Project, resolve_asset};
use rodio::{Decoder, OutputStream, OutputStreamBuilder, Sink};

pub(super) struct CompositionAudio {
    stream: OutputStream,
    sink: Sink,
    path: PathBuf,
}

impl CompositionAudio {
    pub(super) fn load(
        project: &Project,
        root: &Path,
        composition: &str,
    ) -> Result<Option<Self>, String> {
        let comp = project
            .compositions
            .get(composition)
            .ok_or("missing audio composition")?;
        if comp.audio.is_empty() {
            return Ok(None);
        }
        if comp.audio.len() != 1 {
            return Err("interactive audio supports one composition-aligned WAV asset".into());
        }
        let asset = project
            .assets
            .get(&comp.audio[0])
            .ok_or("missing audio asset")?;
        if asset.kind != "audio" {
            return Err("audio reference must name an audio asset".into());
        }
        let path = resolve_asset(root, asset)?;
        let mut stream = OutputStreamBuilder::open_default_stream()
            .map_err(|e| format!("cannot open audio output: {e}"))?;
        stream.log_on_drop(false);
        let sink = Sink::connect_new(stream.mixer());
        sink.pause();
        let source = Decoder::try_from(File::open(&path).map_err(|e| e.to_string())?)
            .map_err(|e| format!("cannot decode audio {}: {e}", path.display()))?;
        sink.append(source);
        Ok(Some(Self { stream, sink, path }))
    }

    pub(super) fn seek(&mut self, seconds: f64, playing: bool) -> Result<(), String> {
        if self.sink.empty() {
            self.sink = Sink::connect_new(self.stream.mixer());
            self.sink.pause();
            self.sink.append(
                Decoder::try_from(File::open(&self.path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?,
            );
        }
        self.sink
            .try_seek(Duration::from_secs_f64(seconds.max(0.0)))
            .map_err(|e| format!("audio seek failed: {e}"))?;
        self.set_playing(playing);
        Ok(())
    }

    pub(super) fn set_playing(&self, playing: bool) {
        if playing {
            self.sink.play();
        } else {
            self.sink.pause();
        }
    }

    pub(super) fn position(&self) -> f64 {
        self.sink.get_pos().as_secs_f64()
    }
    pub(super) fn ended(&self) -> bool {
        self.sink.empty()
    }
}
