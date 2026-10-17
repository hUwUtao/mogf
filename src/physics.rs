//! Fixed-step rigid-body simulation used by interactive preview and baking.
//!
//! The caller owns a session and decides when to advance it. Rendering never
//! implicitly advances physics, which keeps a cached sample and a live sample
//! interchangeable at the interface.

use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::thread::{self, JoinHandle};

use rapier3d::{
    math::{Vector, glamx::Quat},
    prelude::*,
};
use serde::{Deserialize, Serialize};

const MAX_CACHED_SAMPLES: usize = 4096;
const TEXT_COLLIDER_CONTACT_SKIN: f32 = 0.025;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Pose {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
}

impl Default for Pose {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ColliderShape {
    Ball {
        radius: f32,
    },
    Cuboid {
        half_extents: [f32; 3],
    },
    CapsuleY {
        half_height: f32,
        radius: f32,
    },
    ConvexHull {
        points: Vec<[f32; 3]>,
        fallback_half_extents: [f32; 3],
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RigidBodySpec {
    pub id: String,
    #[serde(default)]
    pub per_instance: bool,
    #[serde(default)]
    pub mode: BodyMode,
    #[serde(default)]
    pub pose: Pose,
    #[serde(default)]
    pub shape: Option<ColliderShape>,
    #[serde(default = "one_f32")]
    pub mass: f32,
    #[serde(default = "default_friction")]
    pub friction: f32,
    #[serde(default = "default_restitution")]
    pub restitution: f32,
    #[serde(default)]
    pub linear_velocity: [f32; 3],
    #[serde(default)]
    pub angular_velocity: [f32; 3],
    #[serde(default)]
    pub linear_velocity_spread: [f32; 3],
    #[serde(default)]
    pub angular_velocity_spread: [f32; 3],
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub ccd: bool,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BodyMode {
    #[default]
    Dynamic,
    Fixed,
    Kinematic,
}

fn one_f32() -> f32 {
    1.0
}
fn default_friction() -> f32 {
    0.7
}
fn default_restitution() -> f32 {
    0.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PhysicsSample {
    pub tick: u64,
    pub time_seconds: f64,
    pub bodies: BTreeMap<String, Pose>,
}

/// A stateful fixed-step Rapier simulation. The cache is deliberately bounded;
/// callers can persist samples through the bake layer without retaining the
/// entire composition in RAM.
pub(crate) struct PhysicsSession {
    pipeline: PhysicsPipeline,
    gravity: Vector,
    integration: IntegrationParameters,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    soft_bodies: SoftBodySet,
    handles: BTreeMap<String, RigidBodyHandle>,
    tick: u64,
    step_seconds: f64,
    cache: BTreeMap<u64, PhysicsSample>,
}

impl PhysicsSession {
    pub(crate) fn with_rate(
        specs: &[RigidBodySpec],
        gravity: [f32; 3],
        step_hz: u32,
    ) -> Result<Self, String> {
        if step_hz == 0 || step_hz > 1000 {
            return Err("physics step_hz must be in 1..=1000".into());
        }
        if gravity.iter().any(|v| !v.is_finite()) {
            return Err("physics gravity must be finite".into());
        }
        let mut session = Self {
            pipeline: PhysicsPipeline::new(),
            gravity: Vector::new(gravity[0], gravity[1], gravity[2]),
            integration: IntegrationParameters {
                dt: 1.0 / step_hz as f32,
                ..Default::default()
            },
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            soft_bodies: SoftBodySet::new(),
            handles: BTreeMap::new(),
            tick: 0,
            step_seconds: 1.0 / step_hz as f64,
            cache: BTreeMap::new(),
        };
        let mut ids = std::collections::BTreeSet::new();
        for spec in specs {
            if spec.id.is_empty() || !ids.insert(spec.id.as_str()) {
                return Err(format!(
                    "rigid body ID {:?} is empty or duplicated",
                    spec.id
                ));
            }
            if !spec.mass.is_finite()
                || spec.mass <= 0.0
                || !spec.friction.is_finite()
                || spec.friction < 0.0
                || !spec.restitution.is_finite()
                || !(0.0..=1.0).contains(&spec.restitution)
                || spec
                    .pose
                    .translation
                    .iter()
                    .chain(spec.pose.rotation.iter())
                    .chain(spec.linear_velocity.iter())
                    .chain(spec.angular_velocity.iter())
                    .chain(spec.linear_velocity_spread.iter())
                    .chain(spec.angular_velocity_spread.iter())
                    .any(|v| !v.is_finite())
            {
                return Err(format!(
                    "rigid body {:?} has invalid physical values",
                    spec.id
                ));
            }
            let rotation = Quat::from_xyzw(
                spec.pose.rotation[0],
                spec.pose.rotation[1],
                spec.pose.rotation[2],
                spec.pose.rotation[3],
            );
            if rotation.length_squared() < 1.0e-12 {
                return Err(format!("rigid body {:?} has a zero rotation", spec.id));
            }
            let rotation = rotation.normalize();
            let position = Vector::new(
                spec.pose.translation[0],
                spec.pose.translation[1],
                spec.pose.translation[2],
            );
            let rb = match spec.mode {
                BodyMode::Dynamic => RigidBodyBuilder::dynamic(),
                BodyMode::Fixed => RigidBodyBuilder::fixed(),
                BodyMode::Kinematic => RigidBodyBuilder::kinematic_position_based(),
            }
            .translation(position)
            .rotation(rotation.to_scaled_axis())
            .linvel(Vector::new(
                spec.linear_velocity[0],
                spec.linear_velocity[1],
                spec.linear_velocity[2],
            ))
            .angvel(Vector::new(
                spec.angular_velocity[0],
                spec.angular_velocity[1],
                spec.angular_velocity[2],
            ))
            .additional_mass(spec.mass)
            .ccd_enabled(spec.ccd)
            .build();
            let handle = session.bodies.insert(rb);
            if let Some(shape) = &spec.shape {
                let builder = match shape {
                    ColliderShape::Ball { radius } if radius.is_finite() && *radius > 0.0 => {
                        ColliderBuilder::ball(*radius)
                    }
                    ColliderShape::Cuboid { half_extents }
                        if half_extents.iter().all(|v| v.is_finite() && *v > 0.0) =>
                    {
                        ColliderBuilder::cuboid(half_extents[0], half_extents[1], half_extents[2])
                    }
                    ColliderShape::CapsuleY {
                        half_height,
                        radius,
                    } if half_height.is_finite()
                        && *half_height >= 0.0
                        && radius.is_finite()
                        && *radius > 0.0 =>
                    {
                        ColliderBuilder::capsule_y(*half_height, *radius)
                    }
                    ColliderShape::ConvexHull {
                        points,
                        fallback_half_extents,
                    } if fallback_half_extents
                        .iter()
                        .all(|v| v.is_finite() && *v > 0.0) =>
                    {
                        let points = points
                            .iter()
                            .map(|point| Vector::new(point[0], point[1], point[2]))
                            .collect::<Vec<_>>();
                        ColliderBuilder::convex_hull(&points).unwrap_or_else(|| {
                            ColliderBuilder::cuboid(
                                fallback_half_extents[0],
                                fallback_half_extents[1],
                                fallback_half_extents[2],
                            )
                        })
                    }
                    _ => {
                        return Err(format!(
                            "rigid body {:?} has invalid collider dimensions",
                            spec.id
                        ));
                    }
                };
                session.colliders.insert_with_parent(
                    builder
                        .friction(spec.friction)
                        .restitution(spec.restitution)
                        // Extruded text has a very small depth relative to its
                        // face. A small contact skin keeps edge-on glyphs from
                        // losing contact due to tiny solver separation errors.
                        .contact_skin(TEXT_COLLIDER_CONTACT_SKIN)
                        .build(),
                    handle,
                    &mut session.bodies,
                );
            }
            session.handles.insert(spec.id.clone(), handle);
        }
        session.cache.insert(0, session.snapshot());
        Ok(session)
    }

    pub(crate) fn tick(&self) -> u64 {
        self.tick
    }
    pub(crate) fn cached(&self, tick: u64) -> Option<&PhysicsSample> {
        self.cache.get(&tick)
    }

    /// Advance by a bounded number of fixed steps, returning the newest sample.
    /// A caller should run this on a worker and publish the returned snapshot.
    pub(crate) fn advance(&mut self, steps: u32) -> Result<&PhysicsSample, String> {
        if steps > 240 {
            return Err("one physics advance may request at most 240 fixed steps".into());
        }
        for _ in 0..steps {
            self.pipeline.step(
                self.gravity,
                &self.integration,
                &mut self.islands,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.bodies,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                &mut self.soft_bodies,
                &mut self.ccd_solver,
                &(),
                &(),
            );
            self.tick = self.tick.checked_add(1).ok_or("physics tick overflow")?;
            let sample = self.snapshot();
            if self.cache.len() >= MAX_CACHED_SAMPLES {
                if let Some(oldest) = self.cache.keys().next().copied() {
                    self.cache.remove(&oldest);
                }
            }
            self.cache.insert(self.tick, sample);
        }
        Ok(self
            .cache
            .get(&self.tick)
            .expect("initial sample or newly advanced sample"))
    }

    pub(crate) fn sample_at(&mut self, target_tick: u64) -> Result<PhysicsSample, String> {
        if let Some(sample) = self.cache.get(&target_tick) {
            return Ok(sample.clone());
        }
        if target_tick < self.tick {
            return Err(
                "backward sampling requires restoring a checkpoint or resetting the session".into(),
            );
        }
        while self.tick < target_tick {
            let steps = (target_tick - self.tick).min(240) as u32;
            self.advance(steps)?;
        }
        Ok(self
            .cache
            .get(&target_tick)
            .expect("target tick was cached")
            .clone())
    }

    fn snapshot(&self) -> PhysicsSample {
        let bodies = self
            .handles
            .iter()
            .filter_map(|(id, handle)| {
                self.bodies.get(*handle).map(|body| {
                    let q = body.rotation();
                    (
                        id.clone(),
                        Pose {
                            translation: [
                                body.translation().x,
                                body.translation().y,
                                body.translation().z,
                            ],
                            rotation: [q.x, q.y, q.z, q.w],
                        },
                    )
                })
            })
            .collect();
        PhysicsSample {
            tick: self.tick,
            time_seconds: self.tick as f64 * self.step_seconds,
            bodies,
        }
    }
}

enum WorkerRequest {
    Advance(u64),
    Stop,
}

/// Owns the stateful solver on a worker thread. Preview requests are nonblocking;
/// while the worker catches up, callers keep showing the last sample and its time.
pub(crate) struct PhysicsWorker {
    request: SyncSender<WorkerRequest>,
    completed: Receiver<Result<PhysicsSample, String>>,
    latest: Option<PhysicsSample>,
    join: Option<JoinHandle<()>>,
}

impl PhysicsWorker {
    pub(crate) fn new(
        world: &super::model::PhysicsWorld,
        initial_frame: &super::model::Frame<'_>,
    ) -> Result<Self, String> {
        let specs = expand_instance_bodies(&world.bodies, initial_frame)?;
        let gravity = world.gravity;
        let step_hz = world.step_hz;
        // Validate synchronously so malformed projects fail during reload.
        let initial = PhysicsSession::with_rate(&specs, gravity, step_hz)?
            .cached(0)
            .expect("session contains initial state")
            .clone();
        let (request, requests) = mpsc::sync_channel(1);
        let (completed_tx, completed) = mpsc::sync_channel(1);
        let join = thread::Builder::new()
            .name("mogf-physics-preview".into())
            .spawn(move || {
                let mut session = match PhysicsSession::with_rate(&specs, gravity, step_hz) {
                    Ok(session) => session,
                    Err(error) => {
                        let _ = completed_tx.try_send(Err(error));
                        return;
                    }
                };
                while let Ok(message) = requests.recv() {
                    match message {
                        WorkerRequest::Stop => break,
                        WorkerRequest::Advance(tick) => {
                            if tick < session.tick() {
                                let Ok(reset) = PhysicsSession::with_rate(&specs, gravity, step_hz)
                                else {
                                    break;
                                };
                                session = reset;
                            }
                            let result = session.sample_at(tick);
                            match completed_tx.try_send(result) {
                                Ok(()) => {}
                                Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                                }
                            }
                        }
                    }
                }
            })
            .map_err(|error| format!("cannot start physics worker: {error}"))?;
        Ok(Self {
            request,
            completed,
            latest: Some(initial),
            join: Some(join),
        })
    }

    pub(crate) fn request_tick(&self, tick: u64) {
        match self.request.try_send(WorkerRequest::Advance(tick)) {
            Ok(()) | Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {}
        }
    }

    pub(crate) fn poll(&mut self) -> Result<Option<&PhysicsSample>, String> {
        loop {
            match self.completed.try_recv() {
                Ok(Ok(sample)) => self.latest = Some(sample),
                Ok(Err(error)) => return Err(error),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Err("physics worker stopped".into()),
            }
        }
        Ok(self.latest.as_ref())
    }
}

impl Drop for PhysicsWorker {
    fn drop(&mut self) {
        let _ = self.request.send(WorkerRequest::Stop);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

type Matrix4 = [[f64; 4]; 4];

pub(crate) fn apply_sample(
    frame: &mut super::model::Frame<'_>,
    sample: &PhysicsSample,
    world: &super::model::PhysicsWorld,
) {
    // A body moves its whole source object. Render instances such as text glyphs
    // must keep their authored offsets from that object's origin instead of all
    // being collapsed onto the rigid body's center.
    let deltas: BTreeMap<&str, Matrix4> = world
        .bodies
        .iter()
        .filter_map(|body| {
            let current = sample.bodies.get(&body.id)?;
            let initial_inverse = inverse_rigid_matrix(pose_matrix(body.pose));
            Some((
                body.id.as_str(),
                multiply_matrices(pose_matrix(*current), initial_inverse),
            ))
        })
        .collect();

    for instance in &mut frame.instances {
        if let Some(template) = world
            .bodies
            .iter()
            .find(|body| body.per_instance && body.id == instance.source_object)
        {
            let body_id = format!("{}::{}", template.id, instance.instance_id);
            if let Some(current) = sample.bodies.get(&body_id) {
                let initial = pose_from_matrix(instance.world_matrix);
                let delta = multiply_matrices(
                    pose_matrix(*current),
                    inverse_rigid_matrix(pose_matrix(initial)),
                );
                instance.world_matrix = multiply_matrices(delta, instance.world_matrix);
            }
            continue;
        }
        let Some(delta) = deltas.get(instance.source_object.as_str()) else {
            continue;
        };
        instance.world_matrix = multiply_matrices(*delta, instance.world_matrix);
    }
}

fn expand_instance_bodies(
    templates: &[RigidBodySpec],
    frame: &super::model::Frame<'_>,
) -> Result<Vec<RigidBodySpec>, String> {
    let mut expanded = Vec::new();
    let mut shape_cache = BTreeMap::<u64, ColliderShape>::new();
    for template in templates {
        if !template.per_instance {
            expanded.push(template.clone());
            continue;
        }
        let instances = frame
            .instances
            .iter()
            .filter(|instance| {
                instance.source_object == template.id
                    && instance.geometry_kind == "glyph_mesh"
                    && !instance
                        .glyph_mesh
                        .as_ref()
                        .is_some_and(|mesh| mesh.is_empty())
            })
            .collect::<Vec<_>>();
        if instances.is_empty() {
            return Err(format!(
                "per-instance physics body {:?} has no glyph instances",
                template.id
            ));
        }
        for instance in instances {
            let mut body = template.clone();
            body.id = format!("{}::{}", template.id, instance.instance_id);
            body.per_instance = false;
            body.pose = pose_from_matrix(instance.world_matrix);
            let mut points = std::collections::BTreeSet::new();
            if let Some(mesh) = &instance.glyph_mesh {
                for point in mesh.iter().flat_map(|triangle| triangle.vertices) {
                    points.insert(
                        [point[0] as f32, point[1] as f32, point[2] as f32].map(f32::to_bits),
                    );
                }
            }
            // Derive the fallback from the actual shaped mesh instead of the
            // advance-box size. Italics and overhanging glyphs can extend past
            // that layout box. Keep a minimum extrusion depth for effectively
            // planar glyphs so their collision is still stable when tilted.
            let mut fallback_half_extents = [0.01_f32; 3];
            for point in &points {
                let coordinates = point.map(f32::from_bits);
                for axis in 0..3 {
                    fallback_half_extents[axis] =
                        fallback_half_extents[axis].max(coordinates[axis].abs());
                }
            }
            fallback_half_extents[2] = fallback_half_extents[2].max(0.12);
            let generated_shape = if points.len() >= 4 {
                ColliderShape::ConvexHull {
                    points: points
                        .into_iter()
                        .map(|point| point.map(f32::from_bits))
                        .collect(),
                    fallback_half_extents,
                }
            } else {
                ColliderShape::Cuboid {
                    half_extents: fallback_half_extents,
                }
            };
            body.shape = Some(if let Some(hash) = instance.glyph_hash {
                shape_cache.entry(hash).or_insert(generated_shape).clone()
            } else {
                generated_shape
            });
            let (linear, angular) = deterministic_spread(&instance.instance_id, template.seed);
            for axis in 0..3 {
                body.linear_velocity[axis] += linear[axis] * template.linear_velocity_spread[axis];
                body.angular_velocity[axis] +=
                    angular[axis] * template.angular_velocity_spread[axis];
            }
            expanded.push(body);
        }
    }
    Ok(expanded)
}

fn deterministic_spread(id: &str, seed: u64) -> ([f32; 3], [f32; 3]) {
    let mut state = 0xcbf29ce484222325_u64 ^ seed;
    for byte in id.bytes() {
        state = (state ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    let mut next = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^= value >> 31;
        ((value >> 40) as f32 / ((1_u32 << 24) - 1) as f32) * 2.0 - 1.0
    };
    ([next(), next(), next()], [next(), next(), next()])
}

fn pose_from_matrix(matrix: Matrix4) -> Pose {
    let mut columns = [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ];
    for column in &mut columns {
        let length = (column[0] * column[0] + column[1] * column[1] + column[2] * column[2])
            .sqrt()
            .max(1.0e-12);
        for value in column {
            *value /= length;
        }
    }
    let m00 = columns[0][0];
    let m01 = columns[1][0];
    let m02 = columns[2][0];
    let m10 = columns[0][1];
    let m11 = columns[1][1];
    let m12 = columns[2][1];
    let m20 = columns[0][2];
    let m21 = columns[1][2];
    let m22 = columns[2][2];
    let trace = m00 + m11 + m22;
    let (x, y, z, w) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        ((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
    };
    Pose {
        translation: [
            matrix[0][3] as f32,
            matrix[1][3] as f32,
            matrix[2][3] as f32,
        ],
        rotation: [x as f32, y as f32, z as f32, w as f32],
    }
}

fn pose_matrix(pose: Pose) -> Matrix4 {
    let [mut x, mut y, mut z, mut w] = pose.rotation.map(f64::from);
    let norm = (x * x + y * y + z * z + w * w).sqrt();
    x /= norm;
    y /= norm;
    z /= norm;
    w /= norm;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            f64::from(pose.translation[0]),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            f64::from(pose.translation[1]),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
            f64::from(pose.translation[2]),
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn inverse_rigid_matrix(matrix: Matrix4) -> Matrix4 {
    let mut inverse = [
        [matrix[0][0], matrix[1][0], matrix[2][0], 0.0],
        [matrix[0][1], matrix[1][1], matrix[2][1], 0.0],
        [matrix[0][2], matrix[1][2], matrix[2][2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for values in inverse.iter_mut().take(3) {
        values[3] =
            -(values[0] * matrix[0][3] + values[1] * matrix[1][3] + values[2] * matrix[2][3]);
    }
    inverse
}

fn multiply_matrices(left: Matrix4, right: Matrix4) -> Matrix4 {
    let mut product = [[0.0; 4]; 4];
    for (row, values) in product.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = (0..4)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum();
        }
    }
    product
}

pub(crate) fn tick_for_frame(
    frame: u64,
    fps: &super::model::Rational,
    step_hz: u32,
) -> Result<u64, String> {
    let numerator: u128 = fps
        .n
        .parse()
        .map_err(|_| "invalid composition frame rate numerator")?;
    let denominator: u128 = fps
        .d
        .parse()
        .map_err(|_| "invalid composition frame rate denominator")?;
    if numerator == 0 || denominator == 0 {
        return Err("composition frame rate must be positive".into());
    }
    let ticks = u128::from(frame)
        .checked_mul(denominator)
        .and_then(|v| v.checked_mul(u128::from(step_hz)))
        .ok_or("physics sample time overflow")?
        / numerator;
    u64::try_from(ticks).map_err(|_| "physics sample tick overflow".into())
}
