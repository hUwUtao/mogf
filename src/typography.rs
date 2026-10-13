use lyon_tessellation::{
    FillOptions, FillTessellator, VertexBuffers, geometry_builder::simple_builder, math::Point,
    path::Path,
};
use rustybuzz::{BufferClusterLevel, Face as ShapeFace, UnicodeBuffer, Variation, shape};
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    path::{Path as FilePath, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};
use ttf_parser::{Face as OutlineFace, GlyphId, OutlineBuilder, Tag};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Clone, Debug)]
pub struct GlyphTriangle {
    pub vertices: [[f64; 3]; 3],
    pub shade: f64,
}

#[derive(Clone, Debug)]
pub struct PositionedGlyph {
    pub instance_id: String,
    pub character_ids: Vec<String>,
    pub word_id: Option<String>,
    pub origin: [f64; 2],
    pub size: [f64; 2],
    pub triangles: Arc<[GlyphTriangle]>,
    pub geometry_hash: u64,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct TextShapeKey {
    font_path: PathBuf,
    font_stamp: crate::imports::SourceStamp,
    text: String,
    token_ids: Vec<String>,
    em_size: u64,
    max_width: Option<u64>,
    font_weight: u32,
    extrusion_depth: u64,
    fill_rule: String,
    tolerance: u64,
}

struct CachedTextShape {
    glyphs: Arc<[PositionedGlyph]>,
    bytes: usize,
    last_used: u64,
}

#[derive(Default)]
struct TextShapeCache {
    entries: HashMap<TextShapeKey, CachedTextShape>,
    bytes: usize,
    serial: u64,
}

const MAX_TEXT_SHAPE_CACHE_BYTES: usize = 128 * 1024 * 1024;

pub fn shape_text_cached(
    text: &str,
    font_path: &FilePath,
    font_stamp: &crate::imports::SourceStamp,
    font_bytes: &[u8],
    em_size: f64,
    token_ids: &[String],
    max_width: Option<f64>,
    font_weight: f32,
    extrusion_depth: f64,
    fill_rule: &str,
    tolerance: f64,
) -> Result<Arc<[PositionedGlyph]>, String> {
    let key = TextShapeKey {
        font_path: font_path.to_path_buf(),
        font_stamp: font_stamp.clone(),
        text: text.to_owned(),
        token_ids: token_ids.to_vec(),
        em_size: em_size.to_bits(),
        max_width: max_width.map(f64::to_bits),
        font_weight: font_weight.to_bits(),
        extrusion_depth: extrusion_depth.to_bits(),
        fill_rule: fill_rule.to_owned(),
        tolerance: tolerance.to_bits(),
    };
    static CACHE: OnceLock<Mutex<TextShapeCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(TextShapeCache::default()));
    if let Ok(mut cache) = cache.lock() {
        cache.serial = cache.serial.wrapping_add(1);
        let serial = cache.serial;
        if let Some(entry) = cache.entries.get_mut(&key) {
            entry.last_used = serial;
            return Ok(Arc::clone(&entry.glyphs));
        }
    }

    let glyphs = Arc::<[PositionedGlyph]>::from(shape_text(
        text,
        font_bytes,
        em_size,
        token_ids,
        max_width,
        font_weight,
        extrusion_depth,
        fill_rule,
        tolerance,
    )?);
    let key_bytes = std::mem::size_of::<TextShapeKey>()
        + key.font_path.as_os_str().len()
        + key.text.len()
        + key.token_ids.iter().map(String::len).sum::<usize>()
        + key.fill_rule.len();
    let bytes = key_bytes
        + glyphs.len() * std::mem::size_of::<PositionedGlyph>()
        + glyphs
            .iter()
            .map(|glyph| {
                glyph.instance_id.len()
                    + glyph.character_ids.iter().map(String::len).sum::<usize>()
                    + glyph.word_id.as_ref().map_or(0, String::len)
                    + glyph.triangles.len() * std::mem::size_of::<GlyphTriangle>()
            })
            .sum::<usize>();
    if bytes <= MAX_TEXT_SHAPE_CACHE_BYTES
        && let Ok(mut cache) = cache.lock()
    {
        while cache.bytes.saturating_add(bytes) > MAX_TEXT_SHAPE_CACHE_BYTES {
            let Some(evict) = cache
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(removed) = cache.entries.remove(&evict) {
                cache.bytes = cache.bytes.saturating_sub(removed.bytes);
            }
        }
        cache.serial = cache.serial.wrapping_add(1);
        let last_used = cache.serial;
        cache.bytes = cache.bytes.saturating_add(bytes);
        cache.entries.insert(
            key,
            CachedTextShape {
                glyphs: Arc::clone(&glyphs),
                bytes,
                last_used,
            },
        );
    }
    Ok(glyphs)
}

struct ShapedLine {
    byte_start: usize,
    text: String,
    infos: Vec<rustybuzz::GlyphInfo>,
    positions: Vec<rustybuzz::GlyphPosition>,
    advance: [f64; 2],
    baseline_y: f64,
}

#[derive(Default)]
struct OutlineContours {
    contours: Vec<Vec<[f32; 2]>>,
    current: Vec<[f32; 2]>,
    cursor: [f32; 2],
}

impl OutlineContours {
    fn push_point(&mut self, point: [f32; 2]) {
        self.cursor = point;
        self.current.push(point);
    }

    fn finish_contour(&mut self) {
        if self.current.len() >= 3 {
            if self.current.first() == self.current.last() {
                self.current.pop();
            }
            if self.current.len() >= 3 {
                self.contours.push(std::mem::take(&mut self.current));
            }
        }
        self.current.clear();
    }
}

impl OutlineBuilder for OutlineContours {
    fn move_to(&mut self, x: f32, y: f32) {
        self.finish_contour();
        self.push_point([x, y]);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.push_point([x, y]);
    }

    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let start = self.cursor;
        for step in 1..=12 {
            let t = step as f32 / 12.0;
            let u = 1.0 - t;
            self.push_point([
                u * u * start[0] + 2.0 * u * t * cx + t * t * x,
                u * u * start[1] + 2.0 * u * t * cy + t * t * y,
            ]);
        }
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let start = self.cursor;
        for step in 1..=16 {
            let t = step as f32 / 16.0;
            let u = 1.0 - t;
            self.push_point([
                u.powi(3) * start[0] + 3.0 * u * u * t * x1 + 3.0 * u * t * t * x2 + t.powi(3) * x,
                u.powi(3) * start[1] + 3.0 * u * u * t * y1 + 3.0 * u * t * t * y2 + t.powi(3) * y,
            ]);
        }
    }

    fn close(&mut self) {
        self.finish_contour();
    }
}

pub fn shape_text(
    text: &str,
    font_bytes: &[u8],
    em_size: f64,
    token_ids: &[String],
    max_width: Option<f64>,
    font_weight: f32,
    extrusion_depth: f64,
    fill_rule: &str,
    tolerance: f64,
) -> Result<Vec<PositionedGlyph>, String> {
    if !em_size.is_finite() || em_size <= 0.0 {
        return Err("text size must be finite and positive".to_owned());
    }
    if !font_weight.is_finite() || !(100.0..=900.0).contains(&font_weight) {
        return Err("font weight must be in [100, 900]".to_owned());
    }
    if !extrusion_depth.is_finite() || extrusion_depth < 0.0 || extrusion_depth > 2.0 {
        return Err("text extrusion depth must be in [0, 2]".to_owned());
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err("outline tolerance must be finite and positive".to_owned());
    }
    if let Some(width) = max_width
        && (!width.is_finite() || width <= 0.0)
    {
        return Err("text max_width must be finite and positive".to_owned());
    }

    let graphemes: Vec<(usize, &str)> = text.grapheme_indices(true).collect();
    if token_ids.len() != graphemes.len() {
        return Err(format!(
            "text has {} grapheme clusters but token_ids has {} entries",
            graphemes.len(),
            token_ids.len()
        ));
    }
    for token in token_ids {
        if !valid_id(token) {
            return Err(format!("invalid text token ID {token:?}"));
        }
    }
    let tag = Tag::from_bytes(b"wght");
    let mut outline_face = OutlineFace::parse(font_bytes, 0)
        .map_err(|error| format!("cannot parse font outlines: {error:?}"))?;
    if outline_face.set_variation(tag, font_weight).is_none() && font_weight != 400.0 {
        return Err("font asset has no variable wght axis".to_owned());
    }
    let mut shape_face =
        ShapeFace::from_slice(font_bytes, 0).ok_or("cannot parse font shaping tables")?;
    shape_face.set_variations(&[Variation {
        tag,
        value: font_weight,
    }]);
    let units_per_em = f64::from(shape_face.units_per_em());
    if units_per_em <= 0.0 {
        return Err("font has invalid units_per_em".to_owned());
    }
    let scale = em_size / units_per_em;

    let mut word_for_grapheme = vec![None; graphemes.len()];
    for (word_start, word) in text.unicode_word_indices() {
        let word_end = word_start + word.len();
        let members: Vec<usize> = graphemes
            .iter()
            .enumerate()
            .filter_map(|(index, (start, _))| {
                (*start >= word_start && *start < word_end).then_some(index)
            })
            .collect();
        if let Some(first) = members.first() {
            let word_id = format!("word:{}", token_ids[*first]);
            for index in members {
                word_for_grapheme[index] = Some(word_id.clone());
            }
        }
    }

    let mut lines = wrap_lines(text, max_width, &shape_face, scale)?;
    for (index, line) in lines.iter_mut().enumerate() {
        let mut buffer = UnicodeBuffer::new();
        buffer.push_str(&line.text);
        buffer.set_cluster_level(BufferClusterLevel::MonotoneGraphemes);
        buffer.guess_segment_properties();
        let shaped = shape(&shape_face, &[], buffer);
        line.infos = shaped.glyph_infos().to_vec();
        line.positions = shaped.glyph_positions().to_vec();
        line.advance = line.positions.iter().fold([0.0, 0.0], |mut sum, position| {
            sum[0] += f64::from(position.x_advance) * scale;
            sum[1] += f64::from(position.y_advance) * scale;
            sum
        });
        line.baseline_y = index as f64 * em_size * 1.2;
    }

    let vertical_center = lines.len().saturating_sub(1) as f64 * em_size * 1.2 * 0.5;
    let mut glyphs = Vec::new();
    // A shaped text sample uses one font face, variation setting, scale, and
    // outline configuration. Within that sample, a glyph ID therefore maps to
    // the same centered prototype even when it appears at several positions.
    let mut glyph_prototypes: HashMap<u16, (Arc<[GlyphTriangle]>, [f64; 2], u64)> = HashMap::new();
    let mut glyph_ordinal = 0usize;
    for line in lines {
        let mut pen_x = -line.advance[0] * 0.5;
        let mut pen_y = vertical_center - line.baseline_y;
        let clusters: Vec<usize> = line
            .infos
            .iter()
            .map(|info| info.cluster as usize)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        for (info, position) in line.infos.iter().zip(&line.positions) {
            let cluster_start = info.cluster as usize;
            let cluster_end = clusters
                .iter()
                .copied()
                .find(|candidate| *candidate > cluster_start)
                .unwrap_or(line.text.len());
            let global_cluster = line.byte_start + cluster_start;
            let first_grapheme = graphemes
                .partition_point(|(byte, _)| *byte <= global_cluster)
                .saturating_sub(1)
                .min(graphemes.len());
            let mut char_indices: Vec<usize> = graphemes
                .iter()
                .enumerate()
                .filter_map(|(index, (byte, _))| {
                    (*byte >= global_cluster && *byte < line.byte_start + cluster_end)
                        .then_some(index)
                })
                .collect();
            if char_indices.is_empty() && first_grapheme < graphemes.len() {
                char_indices.push(first_grapheme);
            }
            let character_ids: Vec<String> = char_indices
                .iter()
                .map(|index| token_ids[*index].clone())
                .collect();
            let word_id = char_indices
                .iter()
                .find_map(|index| word_for_grapheme[*index].clone());
            let glyph_id = u16::try_from(info.glyph_id)
                .map_err(|_| "font glyph index exceeds supported TrueType range")?;
            let (triangles, size, geometry_hash) = if let Some((triangles, size, geometry_hash)) =
                glyph_prototypes.get(&glyph_id)
            {
                (Arc::clone(triangles), *size, *geometry_hash)
            } else {
                let (mut triangles, size) = extrude_glyph(
                    &mut outline_face,
                    GlyphId(glyph_id),
                    scale,
                    extrusion_depth,
                    fill_rule,
                    tolerance,
                )?;
                if !triangles.is_empty() {
                    let pivot = [size[0] * 0.5, size[1] * 0.5];
                    for triangle in &mut triangles {
                        for vertex in &mut triangle.vertices {
                            vertex[0] -= pivot[0];
                            vertex[1] -= pivot[1];
                        }
                    }
                }
                let geometry_hash = glyph_geometry_hash(&triangles);
                let triangles = Arc::<[GlyphTriangle]>::from(triangles);
                glyph_prototypes.insert(glyph_id, (Arc::clone(&triangles), size, geometry_hash));
                (triangles, size, geometry_hash)
            };
            if !triangles.is_empty() {
                let first_id = character_ids
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "glyph".to_owned());
                glyphs.push(PositionedGlyph {
                    instance_id: format!("glyph:{first_id}:{glyph_ordinal}"),
                    character_ids,
                    word_id,
                    origin: [
                        pen_x + f64::from(position.x_offset) * scale,
                        pen_y + f64::from(position.y_offset) * scale,
                    ],
                    size,
                    triangles,
                    geometry_hash,
                });
            }
            glyph_ordinal += 1;
            pen_x += f64::from(position.x_advance) * scale;
            pen_y += f64::from(position.y_advance) * scale;
        }
    }
    Ok(glyphs)
}

fn glyph_geometry_hash(triangles: &[GlyphTriangle]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    triangles.len().hash(&mut hasher);
    for triangle in triangles {
        triangle.shade.to_bits().hash(&mut hasher);
        for point in triangle.vertices {
            for value in point {
                value.to_bits().hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

fn wrap_lines(
    text: &str,
    max_width: Option<f64>,
    face: &ShapeFace<'_>,
    scale: f64,
) -> Result<Vec<ShapedLine>, String> {
    let mut lines = Vec::new();
    let mut paragraph_start = 0usize;
    for part in text.split_inclusive('\n') {
        let paragraph = part.strip_suffix('\n').unwrap_or(part);
        if let Some(max_width) = max_width {
            let mut line_start = paragraph_start;
            let mut line = String::new();
            let mut pending_space = String::new();
            for (offset, word_part) in paragraph.split_word_bound_indices() {
                if word_part.chars().all(char::is_whitespace) {
                    pending_space.push_str(word_part);
                    continue;
                }
                let mut candidate = line.clone();
                candidate.push_str(&pending_space);
                candidate.push_str(word_part);
                if !line.trim().is_empty() && shaped_advance(&candidate, face, scale) > max_width {
                    lines.push(empty_line(line_start, std::mem::take(&mut line)));
                    line_start = paragraph_start + offset;
                    pending_space.clear();
                    line.push_str(word_part);
                } else {
                    if !line.is_empty() {
                        line.push_str(&pending_space);
                    }
                    line.push_str(word_part);
                    pending_space.clear();
                }
            }
            if !line.trim().is_empty() {
                lines.push(empty_line(line_start, line));
            } else if paragraph.is_empty() {
                lines.push(empty_line(paragraph_start, String::new()));
            }
        } else {
            lines.push(empty_line(paragraph_start, paragraph.to_owned()));
        }
        paragraph_start += part.len();
    }
    if text.is_empty() {
        lines.push(empty_line(0, String::new()));
    }
    Ok(lines)
}

fn empty_line(byte_start: usize, text: String) -> ShapedLine {
    ShapedLine {
        byte_start,
        text,
        infos: Vec::new(),
        positions: Vec::new(),
        advance: [0.0; 2],
        baseline_y: 0.0,
    }
}

fn shaped_advance(text: &str, face: &ShapeFace<'_>, scale: f64) -> f64 {
    let mut buffer = UnicodeBuffer::new();
    buffer.push_str(text);
    buffer.guess_segment_properties();
    shape(face, &[], buffer)
        .glyph_positions()
        .iter()
        .map(|position| f64::from(position.x_advance) * scale)
        .sum()
}

fn extrude_glyph(
    face: &mut OutlineFace<'_>,
    glyph_id: GlyphId,
    scale: f64,
    depth: f64,
    fill_rule: &str,
    tolerance: f64,
) -> Result<(Vec<GlyphTriangle>, [f64; 2]), String> {
    let mut outline = OutlineContours::default();
    if face.outline_glyph(glyph_id, &mut outline).is_none() {
        return Ok((Vec::new(), [0.0; 2]));
    }
    outline.finish_contour();
    if outline.contours.is_empty() {
        return Ok((Vec::new(), [0.0; 2]));
    }

    let mut path_builder = Path::builder();
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    for contour in &outline.contours {
        let first = contour[0];
        path_builder.begin(Point::new(first[0] * scale as f32, first[1] * scale as f32));
        for point in contour.iter().skip(1) {
            path_builder.line_to(Point::new(point[0] * scale as f32, point[1] * scale as f32));
        }
        path_builder.end(true);
        for point in contour {
            min[0] = min[0].min(point[0]);
            min[1] = min[1].min(point[1]);
            max[0] = max[0].max(point[0]);
            max[1] = max[1].max(point[1]);
        }
    }
    let path = path_builder.build();
    let mut buffers: VertexBuffers<Point, u16> = VertexBuffers::new();
    let mut tessellator = FillTessellator::new();
    let options = FillOptions::default()
        .with_tolerance((tolerance * scale).clamp(0.0001, 1.0) as f32)
        .with_fill_rule(if fill_rule == "nonzero" {
            lyon_tessellation::FillRule::NonZero
        } else {
            lyon_tessellation::FillRule::EvenOdd
        });
    tessellator
        .tessellate_path(&path, &options, &mut simple_builder(&mut buffers))
        .map_err(|error| format!("glyph fill tessellation failed: {error:?}"))?;

    let front = depth * 0.5;
    let back = -front;
    let mut triangles = Vec::with_capacity(
        buffers.indices.len() * 2 + outline.contours.iter().map(Vec::len).sum::<usize>() * 2,
    );
    for indices in buffers.indices.chunks_exact(3) {
        let points = [
            buffers.vertices[indices[0] as usize],
            buffers.vertices[indices[1] as usize],
            buffers.vertices[indices[2] as usize],
        ];
        let cap = points.map(|point| [f64::from(point.x), f64::from(point.y), front]);
        triangles.push(GlyphTriangle {
            vertices: cap,
            shade: 1.0,
        });
        triangles.push(GlyphTriangle {
            vertices: [cap[2], cap[1], cap[0]].map(|v| [v[0], v[1], back]),
            shade: 0.76,
        });
    }
    if depth > 0.0 {
        for contour in &outline.contours {
            for index in 0..contour.len() {
                let a = contour[index];
                let b = contour[(index + 1) % contour.len()];
                let a = [f64::from(a[0]) * scale, f64::from(a[1]) * scale];
                let b = [f64::from(b[0]) * scale, f64::from(b[1]) * scale];
                let a_front = [a[0], a[1], front];
                let b_front = [b[0], b[1], front];
                let a_back = [a[0], a[1], back];
                let b_back = [b[0], b[1], back];
                triangles.push(GlyphTriangle {
                    vertices: [a_front, b_front, b_back],
                    shade: 0.42,
                });
                triangles.push(GlyphTriangle {
                    vertices: [a_front, b_back, a_back],
                    shade: 0.42,
                });
            }
        }
    }
    Ok((
        triangles,
        [
            (f64::from(max[0] - min[0]) * scale),
            (f64::from(max[1] - min[1]) * scale),
        ],
    ))
}

fn valid_id(id: &str) -> bool {
    let mut bytes = id.bytes();
    matches!(bytes.next(), Some(b'A'..=b'Z' | b'a'..=b'z'))
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || b"_.:-".contains(&byte))
        && !id.is_empty()
        && id.len() <= 128
}
