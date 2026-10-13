use super::{
    Clock, CurveParams, FractionValue, InstanceTrackValue, Project, Track, Transform, normalized,
    require,
};

pub(super) fn apply_instance_tracks(
    project: &Project,
    composition_id: &str,
    object_id: &str,
    selectors: &[String],
    frame: u64,
    fps: FractionValue,
    timeline_ids: &[String],
    placement: &mut Transform,
    opacity: &mut f64,
) -> Result<(), String> {
    let mut tracks = Vec::new();
    let mut has_opacity_track = false;
    let mut opacity_track_active = false;
    for timeline_id in timeline_ids {
        let timeline = project
            .timelines
            .get(timeline_id)
            .ok_or_else(|| format!("missing timeline {timeline_id:?}"))?;
        for track in &timeline.tracks {
            if track.target.record != object_id
                || track
                    .target
                    .instances
                    .as_ref()
                    .is_none_or(|ids| !ids.iter().any(|id| selectors.contains(id)))
            {
                continue;
            }
            let Clock::Composition(clock_id) = &track.clock;
            require(
                clock_id == composition_id,
                "track uses another composition clock",
            )?;
            let axis = match track.target.slot.as_str() {
                "transform.translation.x" => Some(0),
                "transform.translation.y" => Some(1),
                "transform.translation.z" => Some(2),
                "transform.rotation" => Some(3),
                _ => None,
            };
            require(
                axis.is_some() || track.target.slot == "material.opacity",
                "instance track target must be translation, yaw, or material.opacity",
            )?;
            has_opacity_track |= track.target.slot == "material.opacity";
            let Some(mut t) = track_progress(track, frame, fps)? else {
                continue;
            };
            if track.target.slot == "material.opacity" {
                opacity_track_active = true;
            }
            t = ease(track, t)?;
            let value = if track.target.slot == "transform.rotation" {
                InstanceTrackValue::Quaternion(quaternion_curve_value(track, t)?)
            } else {
                let (from, to) = scalar_curve_endpoints(&track.curve.params)?;
                let value = from + (to - from) * t;
                require(value.is_finite(), "curve produced a non-finite value")?;
                InstanceTrackValue::Scalar(value)
            };
            tracks.push((
                track.priority,
                track
                    .ordinal
                    .parse::<u64>()
                    .map_err(|_| "track ordinal must be a u64 string")?,
                axis,
                track.target.slot.as_str(),
                value,
            ));
        }
    }
    tracks.sort_by_key(|(priority, ordinal, _, _, _)| (*priority, *ordinal));
    for (_, _, axis, slot, value) in tracks {
        match (axis, slot, value) {
            (Some(axis @ 0..=2), _, InstanceTrackValue::Scalar(value)) => {
                placement.translation[axis] += value;
            }
            (Some(3), _, InstanceTrackValue::Quaternion(value)) => {
                placement.rotation = value;
            }
            (None, "material.opacity", InstanceTrackValue::Scalar(value)) => {
                *opacity = value.clamp(0.0, 1.0);
            }
            _ => return Err("instance curve value type does not match its target".to_owned()),
        }
    }
    if has_opacity_track && !opacity_track_active {
        *opacity = 0.0;
    }
    Ok(())
}

pub(super) fn track_progress(
    track: &Track,
    frame: u64,
    fps: FractionValue,
) -> Result<Option<f64>, String> {
    let at = track.at.as_positive_or_zero()?;
    let duration = track.duration.as_positive()?;
    let sample_n = i128::from(frame) * i128::from(fps.d);
    let sample_d = i128::from(fps.n);
    let sample_at = sample_n
        .checked_mul(i128::from(at.d))
        .ok_or("track-time comparison overflow")?;
    let start = i128::from(at.n)
        .checked_mul(sample_d)
        .ok_or("track-time comparison overflow")?;
    let elapsed = sample_at - start;
    if elapsed < 0 {
        return Ok(None);
    }
    let elapsed_scaled = elapsed
        .checked_mul(i128::from(duration.d))
        .ok_or("track-time comparison overflow")?;
    let duration_scaled = i128::from(duration.n)
        .checked_mul(sample_d)
        .and_then(|v| v.checked_mul(i128::from(at.d)))
        .ok_or("track-time comparison overflow")?;
    Ok(Some(if elapsed_scaled >= duration_scaled {
        1.0
    } else {
        (elapsed_scaled as f64 / duration_scaled as f64).clamp(0.0, 1.0)
    }))
}

pub(super) fn evaluated_text_weight(
    project: &Project,
    composition_id: &str,
    object_id: &str,
    frame: u64,
    fps: FractionValue,
    timeline_ids: &[String],
    initial: f64,
) -> Result<f64, String> {
    let mut tracks = Vec::new();
    for timeline_id in timeline_ids {
        let timeline = project
            .timelines
            .get(timeline_id)
            .ok_or_else(|| format!("missing timeline {timeline_id:?}"))?;
        for track in &timeline.tracks {
            if track.target.record != object_id
                || track.target.instances.is_some()
                || track.target.slot != "text.weight"
            {
                continue;
            }
            let Clock::Composition(clock_id) = &track.clock;
            require(
                clock_id == composition_id,
                "text-weight track uses another composition clock",
            )?;
            let Some(progress) = track_progress(track, frame, fps)? else {
                continue;
            };
            let progress = ease(track, progress)?;
            let (from, to) = scalar_curve_endpoints(&track.curve.params)?;
            let value = from + (to - from) * progress;
            require(
                value.is_finite() && (100.0..=900.0).contains(&value),
                "animated font weight must remain in [100, 900]",
            )?;
            let ordinal = track
                .ordinal
                .parse::<u64>()
                .map_err(|_| "track ordinal must be a u64 string")?;
            tracks.push((track.priority, ordinal, value));
        }
    }
    tracks.sort_by_key(|(priority, ordinal, _)| (*priority, *ordinal));
    Ok(tracks.last().map_or(initial, |(_, _, value)| *value))
}

pub(super) fn ease(track: &Track, mut t: f64) -> Result<f64, String> {
    if track.target.slot == "transform.rotation"
        && !matches!(track.curve.params.ease.as_str(), "linear" | "cubic_out")
    {
        return Err(
            "quaternion spin must use monotone easing; configure settle_bounce for the stop"
                .to_owned(),
        );
    }
    match track.curve.params.ease.as_str() {
        "linear" => {}
        "cubic_out" => t = 1.0 - (1.0 - t).powi(3),
        "quadratic_bounce_out" => t = quadratic_bounce_out(t),
        "quadratic_bounce_in" => t = 1.0 - quadratic_bounce_out(1.0 - t),
        other => return Err(format!("unsupported ease {other:?}")),
    }
    Ok(t)
}

pub(super) fn scalar_curve_endpoints(params: &CurveParams) -> Result<(f64, f64), String> {
    match (params.from, params.to) {
        (Some(from), Some(to))
            if params.from_quaternion.is_none()
                && params.to_quaternion.is_none()
                && params.rotation_unit.is_none()
                && params.rotation_axis.is_none()
                && params.spin_turns.is_none()
                && params.settle_bounce.is_none()
                && params.settle_bounce_radians.is_none()
                && params.dolly_zoom_focus_z.is_none()
                && params.dolly_zoom_start_distance.is_none() =>
        {
            Ok((from, to))
        }
        _ => Err("scalar track needs scalar from/to endpoints".to_owned()),
    }
}

pub(super) fn quaternion_curve_value(track: &Track, t: f64) -> Result<[f64; 4], String> {
    let params = &track.curve.params;
    require(
        params.rotation_unit.as_deref() == Some("quaternion_xyzw"),
        "rotation tracks must declare quaternion_xyzw units",
    )?;
    let from = normalized(
        params
            .from_quaternion
            .ok_or("rotation needs from_quaternion")?,
    )?;
    let to = normalized(params.to_quaternion.ok_or("rotation needs to_quaternion")?)?;
    let axis = params.rotation_axis.ok_or("spin needs a rotation_axis")?;
    let axis_length = axis.iter().map(|value| value * value).sum::<f64>().sqrt();
    require(
        axis_length.is_finite() && axis_length > 1e-12,
        "rotation_axis must be finite and nonzero",
    )?;
    let axis = axis.map(|value| value / axis_length);
    require(
        params.from.is_none() && params.to.is_none(),
        "quaternion rotation cannot also have scalar endpoints",
    )?;
    require(
        matches!(params.settle_bounce.as_deref(), Some("at_start" | "at_end")),
        "quaternion spin needs an at_start or at_end settle bounce",
    )?;
    let amplitude = params
        .settle_bounce_radians
        .ok_or("spin settle needs settle_bounce_radians")?;
    require(
        amplitude.is_finite() && (0.0..=0.5).contains(&amplitude),
        "settle_bounce_radians must be finite and in [0, 0.5]",
    )?;
    let mut base = quaternion_slerp(from, to, t)?;
    let turns = f64::from(params.spin_turns.unwrap_or(0));
    if turns != 0.0 {
        let half_angle = std::f64::consts::PI * turns * t;
        let spin = [
            axis[0] * half_angle.sin(),
            axis[1] * half_angle.sin(),
            axis[2] * half_angle.sin(),
            half_angle.cos(),
        ];
        base = quaternion_multiply(base, spin);
    }
    let settle = match params.settle_bounce.as_deref() {
        Some("at_end") => damped_bounce_at_end(t),
        Some("at_start") => damped_bounce_at_end(1.0 - t),
        _ => 0.0,
    };
    if amplitude != 0.0 {
        let angle = amplitude * settle;
        let bounce = [
            axis[0] * (angle * 0.5).sin(),
            axis[1] * (angle * 0.5).sin(),
            axis[2] * (angle * 0.5).sin(),
            (angle * 0.5).cos(),
        ];
        base = quaternion_multiply(base, bounce);
    }
    normalized(base)
}

pub(super) fn damped_bounce_at_end(t: f64) -> f64 {
    let start = 0.82;
    if t <= start || t >= 1.0 {
        return 0.0;
    }
    let phase = (t - start) / (1.0 - start);
    let ramp = phase * phase * (3.0 - 2.0 * phase);
    ramp * (1.0 - phase).powi(2) * (4.0 * std::f64::consts::PI * phase).sin()
}

pub(super) fn quaternion_slerp(
    from: [f64; 4],
    mut to: [f64; 4],
    t: f64,
) -> Result<[f64; 4], String> {
    let mut dot = from.iter().zip(to).map(|(a, b)| a * b).sum::<f64>();
    if dot < 0.0 {
        to = to.map(|value| -value);
        dot = -dot;
    }
    if dot > 0.9995 {
        return normalized(std::array::from_fn(|index| {
            from[index] + (to[index] - from[index]) * t
        }));
    }
    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_theta = theta.sin();
    let a = ((1.0 - t) * theta).sin() / sin_theta;
    let b = (t * theta).sin() / sin_theta;
    normalized(std::array::from_fn(|index| a * from[index] + b * to[index]))
}

pub(super) fn quaternion_multiply(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

pub(super) fn quadratic_bounce_out(t: f64) -> f64 {
    let n = 7.5625;
    let d = 2.75;
    if t < 1.0 / d {
        n * t * t
    } else if t < 2.0 / d {
        let t = t - 1.5 / d;
        n * t * t + 0.75
    } else if t < 2.5 / d {
        let t = t - 2.25 / d;
        n * t * t + 0.9375
    } else {
        let t = t - 2.625 / d;
        n * t * t + 0.984375
    }
}
