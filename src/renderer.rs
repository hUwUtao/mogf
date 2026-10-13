use super::Mat4;

pub(super) fn contact_sheet_ppm(
    frames: &[Vec<u8>],
    size: [u32; 2],
    columns: usize,
) -> Result<Vec<u8>, String> {
    if frames.is_empty() || columns == 0 {
        return Err("contact sheet needs frames and a positive column count".to_owned());
    }
    let width = usize::try_from(size[0]).map_err(|_| "contact sheet width overflow")?;
    let height = usize::try_from(size[1]).map_err(|_| "contact sheet height overflow")?;
    let rows = frames.len().div_ceil(columns);
    let sheet_width = width
        .checked_mul(columns)
        .ok_or("contact sheet width overflow")?;
    let sheet_height = height
        .checked_mul(rows)
        .ok_or("contact sheet height overflow")?;
    let pixels = sheet_width
        .checked_mul(sheet_height)
        .ok_or("contact sheet size overflow")?;
    if sheet_width > 4096 || sheet_height > 4096 || pixels > 16_777_216 {
        return Err("contact sheet exceeds 4096 per side or 16,777,216 pixels".to_owned());
    }
    let mut rgb = vec![
        7_u8;
        pixels
            .checked_mul(3)
            .ok_or("contact sheet byte size overflow")?
    ];
    let expected_len = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or("frame size overflow")?;
    for (frame_index, ppm) in frames.iter().enumerate() {
        let header_end = ppm
            .iter()
            .enumerate()
            .filter(|(_, byte)| **byte == b'\n')
            .nth(2)
            .map(|(index, _)| index + 1)
            .ok_or("rendered frame has an invalid PPM header")?;
        let payload = ppm
            .get(header_end..)
            .ok_or("rendered frame has an invalid PPM payload")?;
        if payload.len() != expected_len {
            return Err("rendered frame dimensions do not match the contact sheet".to_owned());
        }
        let origin_x = (frame_index % columns) * width;
        let origin_y = (frame_index / columns) * height;
        for row in 0..height {
            let source = row * width * 3;
            let target = ((origin_y + row) * sheet_width + origin_x) * 3;
            rgb[target..target + width * 3].copy_from_slice(&payload[source..source + width * 3]);
        }
        draw_frame_label(
            &mut rgb,
            sheet_width,
            origin_x + 6,
            origin_y + 6,
            frame_index as u64,
        );
    }
    let mut ppm = format!("P6\n{sheet_width} {sheet_height}\n255\n").into_bytes();
    ppm.extend(rgb);
    Ok(ppm)
}

fn draw_frame_label(rgb: &mut [u8], width: usize, mut x: usize, y: usize, number: u64) {
    const DIGITS: [[u8; 5]; 10] = [
        [0b111, 0b101, 0b101, 0b101, 0b111],
        [0b010, 0b110, 0b010, 0b010, 0b111],
        [0b111, 0b001, 0b111, 0b100, 0b111],
        [0b111, 0b001, 0b111, 0b001, 0b111],
        [0b101, 0b101, 0b111, 0b001, 0b001],
        [0b111, 0b100, 0b111, 0b001, 0b111],
        [0b111, 0b100, 0b111, 0b101, 0b111],
        [0b111, 0b001, 0b010, 0b010, 0b010],
        [0b111, 0b101, 0b111, 0b101, 0b111],
        [0b111, 0b101, 0b111, 0b001, 0b111],
    ];
    let label = number.to_string();
    let box_width = label.len() * 4 + 6;
    for py in y.saturating_sub(3)..y + 11 {
        for px in x.saturating_sub(3)..x + box_width {
            let pixel = (py * width + px) * 3;
            if let Some(color) = rgb.get_mut(pixel..pixel + 3) {
                color.copy_from_slice(&[10, 14, 24]);
            }
        }
    }
    x += 1;
    for byte in label.bytes() {
        let glyph = DIGITS[(byte - b'0') as usize];
        for (row, bits) in glyph.into_iter().enumerate() {
            for col in 0..3 {
                if bits & (1 << (2 - col)) == 0 {
                    continue;
                }
                let pixel = ((y + row) * width + x + col) * 3;
                if let Some(color) = rgb.get_mut(pixel..pixel + 3) {
                    color.copy_from_slice(&[255, 255, 255]);
                }
            }
        }
        x += 4;
    }
}

pub(super) fn plane_triangles(size: [f64; 3]) -> Vec<[[f64; 3]; 3]> {
    let x = size[0] * 0.5;
    let y = size[1] * 0.5;
    let a = [-x, -y, 0.0];
    let b = [x, -y, 0.0];
    let c = [x, y, 0.0];
    let d = [-x, y, 0.0];
    vec![[a, b, c], [a, c, d]]
}

pub(super) fn box_triangles(size: [f64; 3]) -> Vec<[[f64; 3]; 3]> {
    let [x, y, z] = [size[0] * 0.5, size[1] * 0.5, size[2] * 0.5];
    let faces = [
        [[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]],
        [[x, -y, -z], [-x, -y, -z], [-x, y, -z], [x, y, -z]],
        [[x, -y, z], [x, -y, -z], [x, y, -z], [x, y, z]],
        [[-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, -z]],
        [[-x, y, z], [x, y, z], [x, y, -z], [-x, y, -z]],
        [[-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z]],
    ];
    let mut triangles = Vec::with_capacity(12);
    for [a, b, c, d] in faces {
        triangles.push([a, b, c]);
        triangles.push([a, c, d]);
    }
    triangles
}

pub(super) fn transform_point(matrix: &Mat4, point: [f64; 3]) -> [f64; 3] {
    let input = [point[0], point[1], point[2], 1.0];
    let out: [f64; 4] =
        std::array::from_fn(|row| (0..4).map(|col| matrix[row][col] * input[col]).sum());
    [out[0] / out[3], out[1] / out[3], out[2] / out[3]]
}

pub(super) fn inverse(matrix: &Mat4) -> Result<Mat4, String> {
    let mut augmented = [[0.0; 8]; 4];
    for row in 0..4 {
        for col in 0..4 {
            augmented[row][col] = matrix[row][col];
        }
        augmented[row][4 + row] = 1.0;
    }
    for pivot in 0..4 {
        let best = (pivot..4)
            .max_by(|a, b| {
                augmented[*a][pivot]
                    .abs()
                    .total_cmp(&augmented[*b][pivot].abs())
            })
            .unwrap();
        if augmented[best][pivot].abs() < 1e-12 {
            return Err("camera transform is singular".to_owned());
        }
        augmented.swap(pivot, best);
        let divisor = augmented[pivot][pivot];
        for col in 0..8 {
            augmented[pivot][col] /= divisor;
        }
        for row in 0..4 {
            if row == pivot {
                continue;
            }
            let factor = augmented[row][pivot];
            for col in 0..8 {
                augmented[row][col] -= factor * augmented[pivot][col];
            }
        }
    }
    Ok(std::array::from_fn(|row| {
        std::array::from_fn(|col| augmented[row][col + 4])
    }))
}
