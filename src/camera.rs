use serde::{Deserialize, Serialize};

use super::{Clock, CurveParams, FractionValue, Project, animation, require};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CameraParams {
    pub(super) fov_y: Option<f64>,
    pub(super) ortho_height: Option<f64>,
    pub(super) near: f64,
    pub(super) far: f64,
}

#[derive(Debug, Serialize)]
pub(super) struct CameraFrame {
    pub(super) object: String,
    pub(super) projection: CameraProjection,
    pub(super) near: f64,
    pub(super) far: f64,
    pub(super) world_matrix: [[f64; 4]; 4],
}

#[derive(Serialize)]
pub(super) struct MeasurementPlane {
    pub(super) distance: f64,
    /// Visible camera-space width and height at this distance.
    pub(super) size: [f64; 2],
    /// Visible camera-space units per composition pixel.
    pub(super) units_per_pixel: [f64; 2],
    /// Camera-aligned plane transform translated to the requested depth.
    pub(super) world_matrix: [[f64; 4]; 4],
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum CameraProjection {
    Perspective { fov_y: f64 },
    Orthographic { height: f64 },
}

pub(super) fn measurement_plane(
    camera: &CameraFrame,
    aspect: f64,
    composition_size: [u32; 2],
    distance: f64,
) -> Result<MeasurementPlane, String> {
    require(
        distance.is_finite() && distance > 0.0,
        "measurement-plane distance must be finite and positive",
    )?;
    require(
        aspect.is_finite() && aspect > 0.0,
        "measurement-plane aspect must be finite and positive",
    )?;
    require(
        composition_size[0] > 0 && composition_size[1] > 0,
        "measurement-plane composition dimensions must be nonzero",
    )?;
    let height = match camera.projection {
        CameraProjection::Perspective { fov_y } => 2.0 * distance * (fov_y * 0.5).tan(),
        CameraProjection::Orthographic { height } => height,
    };
    let size = [height * aspect, height];
    let local_center = [0.0, 0.0, -distance, 1.0];
    let center: [f64; 4] = std::array::from_fn(|row| {
        (0..4)
            .map(|column| camera.world_matrix[row][column] * local_center[column])
            .sum()
    });
    let world_matrix = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            if column == 3 {
                center[row] / center[3]
            } else {
                camera.world_matrix[row][column]
            }
        })
    });
    Ok(MeasurementPlane {
        distance,
        size,
        units_per_pixel: [
            size[0] / f64::from(composition_size[0]),
            size[1] / f64::from(composition_size[1]),
        ],
        world_matrix,
    })
}

pub(super) fn evaluated_fov_y(
    project: &Project,
    composition_id: &str,
    camera_id: &str,
    frame: u64,
    fps: FractionValue,
    timeline_ids: &[String],
    initial: f64,
    camera_world_z: f64,
) -> Result<f64, String> {
    let mut tracks = Vec::new();
    for timeline_id in timeline_ids {
        let timeline = project
            .timelines
            .get(timeline_id)
            .ok_or_else(|| format!("missing timeline {timeline_id:?}"))?;
        for track in &timeline.tracks {
            if track.target.record != camera_id
                || track.target.instances.is_some()
                || track.target.slot != "camera.fov_y"
            {
                continue;
            }
            let Clock::Composition(clock_id) = &track.clock;
            require(
                clock_id == composition_id,
                "camera FOV track uses another composition clock",
            )?;
            let Some(progress) = animation::track_progress(track, frame, fps)? else {
                continue;
            };
            let (from, to) = fov_curve_endpoints(&track.curve.params)?;
            let value = if let Some((focus_z, start_distance)) =
                dolly_zoom_params(&track.curve.params)?
            {
                let current_distance = camera_world_z - focus_z;
                require(
                    current_distance > 0.0,
                    "dolly_zoom focus distance must remain positive",
                )?;
                2.0 * (((start_distance - focus_z) * (from * 0.5).tan()) / current_distance).atan()
            } else {
                let progress = animation::ease(track, progress)?;
                from + (to - from) * progress
            };
            require(
                value.is_finite() && value > 0.0 && value < std::f64::consts::PI,
                "animated camera fov_y must remain in (0, pi) radians",
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

pub(super) fn fov_curve_endpoints(params: &CurveParams) -> Result<(f64, f64), String> {
    let (Some(from), Some(to)) = (params.from, params.to) else {
        return Err("camera FOV track needs scalar from/to endpoints".to_owned());
    };
    require(
        params.from_quaternion.is_none()
            && params.to_quaternion.is_none()
            && params.rotation_unit.is_none()
            && params.rotation_axis.is_none()
            && params.spin_turns.is_none()
            && params.settle_bounce.is_none()
            && params.settle_bounce_radians.is_none(),
        "camera FOV track needs scalar endpoints",
    )?;
    Ok((from, to))
}

pub(super) fn dolly_zoom_params(params: &CurveParams) -> Result<Option<(f64, f64)>, String> {
    if params.ease != "dolly_zoom" {
        require(
            params.dolly_zoom_focus_z.is_none() && params.dolly_zoom_start_distance.is_none(),
            "dolly_zoom parameters require the dolly_zoom ease",
        )?;
        return Ok(None);
    }

    let focus_z = params
        .dolly_zoom_focus_z
        .ok_or("dolly_zoom track needs dolly_zoom_focus_z")?;
    let start_distance = params
        .dolly_zoom_start_distance
        .ok_or("dolly_zoom track needs dolly_zoom_start_distance")?;
    require(
        focus_z.is_finite() && start_distance.is_finite() && start_distance > focus_z,
        "dolly_zoom focus distance must remain positive",
    )?;
    Ok(Some((focus_z, start_distance)))
}
