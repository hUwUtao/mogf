//! Composition-clock WAV decoding, mixing, and interactive playback.
use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    path::Path,
    time::Duration,
};

use crate::{
    model::{AudioBus, AudioClip, Composition, Project, ScheduledAudioClip},
    resolve_asset,
};
use rodio::{Decoder, OutputStream, OutputStreamBuilder, Sink, Source, buffer::SamplesBuffer};

pub(super) const OUTPUT_SAMPLE_RATE: u32 = 48_000;
const MAX_AUDIO_SECONDS: f64 = 600.0;

struct DecodedClip {
    samples: Vec<f32>,
    channels: u16,
    sample_rate: u32,
}

impl DecodedClip {
    fn decode(path: &Path) -> Result<Self, String> {
        let decoder = Decoder::try_from(
            File::open(path)
                .map_err(|error| format!("cannot open audio {}: {error}", path.display()))?,
        )
        .map_err(|error| format!("cannot decode WAV {}: {error}", path.display()))?;
        let channels = decoder.channels();
        let sample_rate = decoder.sample_rate();
        if !(channels == 1 || channels == 2) || sample_rate == 0 {
            return Err(format!(
                "audio {} must be mono or stereo with a valid sample rate",
                path.display()
            ));
        }
        let samples: Vec<f32> = decoder.collect();
        if samples.is_empty() || samples.len() % usize::from(channels) != 0 {
            return Err(format!(
                "audio {} has no complete sample frames",
                path.display()
            ));
        }
        Ok(Self {
            samples,
            channels,
            sample_rate,
        })
    }

    fn frame_count(&self) -> usize {
        self.samples.len() / usize::from(self.channels)
    }

    fn stereo_at(&self, position: f64) -> Option<[f32; 2]> {
        if !position.is_finite() || position < 0.0 || position >= self.frame_count() as f64 {
            return None;
        }
        let first = position.floor() as usize;
        let second = (first + 1).min(self.frame_count() - 1);
        let blend = (position - first as f64) as f32;
        let channel = usize::from(self.channels);
        let sample = |frame: usize, index: usize| self.samples[frame * channel + index];
        let left = sample(first, 0) + (sample(second, 0) - sample(first, 0)) * blend;
        let right = if self.channels == 1 {
            left
        } else {
            sample(first, 1) + (sample(second, 1) - sample(first, 1)) * blend
        };
        Some([left, right])
    }
}

fn schedule(clip: &AudioClip) -> (&str, &str, f64, f64, Option<f64>, f64) {
    match clip {
        AudioClip::Asset(asset) => (asset, "main", 0.0, 0.0, None, 1.0),
        AudioClip::Scheduled(ScheduledAudioClip {
            asset,
            channel,
            start,
            source_start,
            duration,
            gain,
        }) => (asset, channel, *start, *source_start, *duration, *gain),
    }
}

/// Offline processor contract used by bus and master stages. Plugin adapters can implement this
/// later without changing timeline scheduling or the WAV/FFmpeg sinks.
trait AudioProcessor {
    fn process(&self, samples: &mut [f32]) -> Result<(), String>;
}

struct AudioPipeline {
    stages: Vec<Box<dyn AudioProcessor>>,
}

impl AudioPipeline {
    fn process(&self, samples: &mut [f32]) -> Result<(), String> {
        for stage in &self.stages {
            stage.process(samples)?;
        }
        Ok(())
    }
}

struct BusGainPan(AudioBus);

impl AudioProcessor for BusGainPan {
    fn process(&self, samples: &mut [f32]) -> Result<(), String> {
        if samples.len() % 2 != 0 {
            return Err("stereo bus has an incomplete sample frame".into());
        }
        let pan = self.0.pan;
        let left_gain = self.0.gain * (1.0 - pan.max(0.0));
        let right_gain = self.0.gain * (1.0 + pan.min(0.0));
        for frame in samples.chunks_exact_mut(2) {
            frame[0] *= left_gain as f32;
            frame[1] *= right_gain as f32;
        }
        Ok(())
    }
}

struct OutputClamp;

impl AudioProcessor for OutputClamp {
    fn process(&self, samples: &mut [f32]) -> Result<(), String> {
        for sample in samples {
            *sample = if sample.is_finite() {
                sample.clamp(-1.0, 1.0)
            } else {
                0.0
            };
        }
        Ok(())
    }
}

pub(super) fn mix_composition(
    project: &Project,
    root: &Path,
    composition: &Composition,
) -> Result<Vec<f32>, String> {
    let duration = composition.duration.as_positive()?;
    let seconds = duration.n as f64 / duration.d as f64;
    mix_composition_range(project, root, composition, 0.0, seconds)
}

pub(super) fn mix_composition_range(
    project: &Project,
    root: &Path,
    composition: &Composition,
    range_start: f64,
    range_duration: f64,
) -> Result<Vec<f32>, String> {
    if !range_start.is_finite()
        || range_start < 0.0
        || !range_duration.is_finite()
        || range_duration <= 0.0
        || range_duration > MAX_AUDIO_SECONDS
    {
        return Err(format!(
            "audio composition duration must be at most {MAX_AUDIO_SECONDS:.0} seconds"
        ));
    }
    let frame_count = (range_duration * f64::from(OUTPUT_SAMPLE_RATE)).ceil() as usize;
    let sample_count = frame_count
        .checked_mul(2)
        .ok_or("audio sample count overflow")?;
    let mut mix = vec![0.0_f32; sample_count];
    if composition.audio_buses.len() > 32 {
        return Err("composition supports at most 32 audio buses".into());
    }
    for (id, bus) in &composition.audio_buses {
        if id.is_empty()
            || !bus.gain.is_finite()
            || !(bus.gain as f32).is_finite()
            || !bus.pan.is_finite()
            || !(-1.0..=1.0).contains(&bus.pan)
        {
            return Err(format!("audio bus {id:?} has invalid gain or pan"));
        }
    }
    let mut buses: std::collections::BTreeMap<&str, Vec<f32>> = std::collections::BTreeMap::new();

    for clip in &composition.audio {
        let (asset_id, channel, start, source_start, clip_duration, gain) = schedule(clip);
        if channel.is_empty()
            || (channel != "main" && !composition.audio_buses.contains_key(channel))
        {
            return Err(format!(
                "audio clip {asset_id:?} references missing bus {channel:?}"
            ));
        }
        let asset = project
            .assets
            .get(asset_id)
            .ok_or_else(|| format!("missing audio asset {asset_id:?}"))?;
        if asset.kind != "audio" {
            return Err(format!(
                "audio reference {asset_id:?} must name an audio asset"
            ));
        }
        if !start.is_finite()
            || start < 0.0
            || !source_start.is_finite()
            || source_start < 0.0
            || !gain.is_finite()
            || clip_duration.is_some_and(|duration| !duration.is_finite() || duration <= 0.0)
        {
            return Err(format!(
                "audio clip {asset_id:?} has invalid timing or gain"
            ));
        }
        let path = resolve_asset(root, asset)?;
        let source = DecodedClip::decode(&path)?;
        let source_start_frame = source_start * f64::from(source.sample_rate);
        let available_seconds =
            (source.frame_count() as f64 / f64::from(source.sample_rate) - source_start).max(0.0);
        let clip_seconds = clip_duration
            .unwrap_or(available_seconds)
            .min(available_seconds);
        let clip_end = start + clip_seconds;
        let range_end = range_start + range_duration;
        let first_output_frame = ((start - range_start) * f64::from(OUTPUT_SAMPLE_RATE))
            .ceil()
            .clamp(0.0, frame_count as f64) as usize;
        let end_frame = ((clip_end.min(range_end) - range_start) * f64::from(OUTPUT_SAMPLE_RATE))
            .ceil()
            .clamp(0.0, frame_count as f64) as usize;
        let bus_samples = buses
            .entry(channel)
            .or_insert_with(|| vec![0.0; sample_count]);
        for output_frame in first_output_frame..end_frame {
            let timeline_time = range_start + output_frame as f64 / f64::from(OUTPUT_SAMPLE_RATE);
            if timeline_time < start {
                continue;
            }
            let source_position =
                source_start_frame + (timeline_time - start) * f64::from(source.sample_rate);
            if let Some([left, right]) = source.stereo_at(source_position) {
                bus_samples[output_frame * 2] += left * gain as f32;
                bus_samples[output_frame * 2 + 1] += right * gain as f32;
            }
        }
    }
    for (id, mut samples) in buses {
        let bus = composition.audio_buses.get(id).cloned().unwrap_or_default();
        let pipeline = AudioPipeline {
            stages: vec![Box::new(BusGainPan(bus))],
        };
        pipeline.process(&mut samples)?;
        for (master, sample) in mix.iter_mut().zip(samples) {
            *master += sample;
        }
    }
    AudioPipeline {
        stages: vec![Box::new(OutputClamp)],
    }
    .process(&mut mix)?;
    Ok(mix)
}

pub(super) fn write_wav(path: &Path, samples: &[f32]) -> Result<(), String> {
    if samples.len() % 2 != 0 {
        return Err("stereo audio buffer has an incomplete frame".into());
    }
    let data_len = u32::try_from(samples.len().checked_mul(2).ok_or("WAV size overflow")?)
        .map_err(|_| "WAV exceeds the RIFF size limit")?;
    let riff_len = 36_u32
        .checked_add(data_len)
        .ok_or("WAV exceeds the RIFF size limit")?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    let write_result = (|| -> io::Result<()> {
        file.write_all(b"RIFF")?;
        file.write_all(&riff_len.to_le_bytes())?;
        file.write_all(b"WAVEfmt ")?;
        file.write_all(&16_u32.to_le_bytes())?;
        file.write_all(&1_u16.to_le_bytes())?;
        file.write_all(&2_u16.to_le_bytes())?;
        file.write_all(&OUTPUT_SAMPLE_RATE.to_le_bytes())?;
        file.write_all(&(OUTPUT_SAMPLE_RATE * 4).to_le_bytes())?;
        file.write_all(&4_u16.to_le_bytes())?;
        file.write_all(&16_u16.to_le_bytes())?;
        file.write_all(b"data")?;
        file.write_all(&data_len.to_le_bytes())?;
        for sample in samples {
            let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
            file.write_all(&pcm.to_le_bytes())?;
        }
        Ok(())
    })();
    if let Err(error) = write_result {
        drop(file);
        let _ = std::fs::remove_file(path);
        return Err(format!("cannot write WAV {}: {error}", path.display()));
    }
    Ok(())
}

pub(super) struct CompositionAudio {
    stream: OutputStream,
    sink: Sink,
    samples: Vec<f32>,
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
        let samples = mix_composition(project, root, comp)?;
        let mut stream = OutputStreamBuilder::open_default_stream()
            .map_err(|error| format!("cannot open audio output: {error}"))?;
        stream.log_on_drop(false);
        let sink = Sink::connect_new(stream.mixer());
        sink.pause();
        sink.append(SamplesBuffer::new(2, OUTPUT_SAMPLE_RATE, samples.clone()));
        Ok(Some(Self {
            stream,
            sink,
            samples,
        }))
    }

    pub(super) fn seek(&mut self, seconds: f64, playing: bool) -> Result<(), String> {
        if self.sink.empty() {
            self.sink = Sink::connect_new(self.stream.mixer());
            self.sink.pause();
            self.sink.append(SamplesBuffer::new(
                2,
                OUTPUT_SAMPLE_RATE,
                self.samples.clone(),
            ));
        }
        self.sink
            .try_seek(Duration::from_secs_f64(seconds.max(0.0)))
            .map_err(|error| format!("audio seek failed: {error}"))?;
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
