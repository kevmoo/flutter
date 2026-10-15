use crate::pslg::{
    extract_fill_contours, intern_vertex, is_simple_convex_contour, Pt2, AREA_EPS, INTERSECT_EPS,
    SNAP_EPS,
};
use impeller_geometry::Scalar;
use impeller_path::{Convexity, FillType, Path, PathBuilder};
use std::collections::BTreeSet;

/// Boolean path operation matching Skia `SkPathOp` (`0..=4`).
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathOp {
    Difference = 0,
    Intersect = 1,
    Union = 2,
    Xor = 3,
    ReverseDifference = 4,
}

impl PathOp {
    /// Converts a Skia `SkPathOp` integer code (`0..=4`) into a `PathOp`.
    pub const fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Difference),
            1 => Some(Self::Intersect),
            2 => Some(Self::Union),
            3 => Some(Self::Xor),
            4 => Some(Self::ReverseDifference),
            _ => None,
        }
    }
}

#[inline]
fn eval_op(op: PathOp, in_a: bool, in_b: bool) -> bool {
    match op {
        PathOp::Difference => in_a && !in_b,
        PathOp::Intersect => in_a && in_b,
        PathOp::Union => in_a || in_b,
        PathOp::Xor => in_a ^ in_b,
        PathOp::ReverseDifference => in_b && !in_a,
    }
}

fn point_in_contours(pt: Pt2, contours: &[Vec<Pt2>], fill_type: FillType) -> bool {
    let mut winding = 0i32;
    for contour in contours {
        let n = contour.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = contour[i];
            let b = contour[(i + 1) % n];
            if a.y <= pt.y {
                if b.y > pt.y && b.sub(a).cross(pt.sub(a)) > 0.0 {
                    winding += 1;
                }
            } else if b.y <= pt.y && b.sub(a).cross(pt.sub(a)) < 0.0 {
                winding -= 1;
            }
        }
    }
    match fill_type {
        FillType::NonZero => winding != 0,
        FillType::Odd => winding.rem_euclid(2) != 0,
    }
}

fn contours_to_path(contours: &[Vec<Pt2>], fill_type: FillType) -> Path {
    let mut pb = PathBuilder::new();
    for contour in contours {
        if contour.len() < 3 {
            continue;
        }
        pb.move_to(contour[0].to_point());
        for p in &contour[1..] {
            pb.line_to(p.to_point());
        }
        pb.close();
    }
    let mut out = pb.build(fill_type);
    if contours.len() == 1 && is_simple_convex_contour(&contours[0]) {
        out.set_convexity(Convexity::Convex);
    }
    out
}

fn simplify_contour(pts: &mut Vec<Pt2>) {
    loop {
        if pts.len() < 3 {
            break;
        }
        let before = pts.len();
        let mut dedup: Vec<Pt2> = Vec::with_capacity(pts.len());
        for &p in pts.iter() {
            if dedup
                .last()
                .is_none_or(|last| last.dist_sq(p) > SNAP_EPS * SNAP_EPS)
            {
                dedup.push(p);
            }
        }
        if dedup.len() >= 2
            && dedup.first().unwrap().dist_sq(*dedup.last().unwrap()) <= SNAP_EPS * SNAP_EPS
        {
            dedup.pop();
        }
        *pts = dedup;
        if pts.len() < 3 {
            break;
        }

        let mut removed = false;
        let n = pts.len();
        for i in 0..n {
            let a = pts[(i + n - 1) % n];
            let b = pts[i];
            let c = pts[(i + 1) % n];
            if a.dist_sq(c) <= SNAP_EPS * SNAP_EPS {
                pts.remove(i);
                removed = true;
                break;
            }
            let ab = b.sub(a);
            let bc = c.sub(b);
            let collinear_eps = (1e-8 * (ab.len_sq() * bc.len_sq()).sqrt()).max(1e-11);
            if ab.cross(bc).abs() <= collinear_eps && ab.dot(bc) > 0.0 {
                pts.remove(i);
                removed = true;
                break;
            }
        }
        if !removed && pts.len() == before {
            break;
        }
    }
}

fn contour_signed_area2(pts: &[Pt2]) -> f64 {
    let n = pts.len();
    let mut sum = 0.0;
    for i in 0..n {
        sum += pts[i].cross(pts[(i + 1) % n]);
    }
    sum
}

/// Computes an exact 2D boolean path combination (`Difference`, `Intersect`, `Union`, `Xor`,
/// or `ReverseDifference`) over flattened polygon contours in `f64` precision.
pub fn combine_paths(path_a: &Path, path_b: &Path, op: PathOp, scale: Scalar) -> Path {
    if op == PathOp::ReverseDifference {
        return combine_paths(path_b, path_a, PathOp::Difference, scale);
    }

    let tol_scale = if scale.is_finite() && scale > 0.0 {
        scale.max(1.0)
    } else {
        1.0
    };

    let bounds_a = path_a.get_bounding_box();
    let bounds_b = path_b.get_bounding_box();
    let a_empty = bounds_a.is_none_or(|r| r.get_width() <= 0.0 || r.get_height() <= 0.0);
    let b_empty = bounds_b.is_none_or(|r| r.get_width() <= 0.0 || r.get_height() <= 0.0);

    if a_empty && b_empty {
        return PathBuilder::new().build(FillType::NonZero);
    }
    if a_empty {
        return match op {
            PathOp::Difference | PathOp::Intersect => PathBuilder::new().build(FillType::NonZero),
            PathOp::Union | PathOp::Xor => path_b.clone(),
            PathOp::ReverseDifference => unreachable!(),
        };
    }
    if b_empty {
        return match op {
            PathOp::Intersect => PathBuilder::new().build(FillType::NonZero),
            PathOp::Difference | PathOp::Union | PathOp::Xor => path_a.clone(),
            PathOp::ReverseDifference => unreachable!(),
        };
    }

    let ra = bounds_a.unwrap();
    let rb = bounds_b.unwrap();
    let disjoint = ra.get_right() + 1e-4 < rb.get_left()
        || rb.get_right() + 1e-4 < ra.get_left()
        || ra.get_bottom() + 1e-4 < rb.get_top()
        || rb.get_bottom() + 1e-4 < ra.get_top();

    if disjoint {
        match op {
            PathOp::Intersect => return PathBuilder::new().build(FillType::NonZero),
            PathOp::Difference => return path_a.clone(),
            PathOp::Union | PathOp::Xor => {
                if path_a.fill_type == path_b.fill_type {
                    let mut pb = PathBuilder::new();
                    pb.add_path(path_a);
                    pb.add_path(path_b);
                    return pb.build(path_a.fill_type);
                }
            }
            PathOp::ReverseDifference => unreachable!(),
        }
    }

    let (contours_a, _) = extract_fill_contours(path_a, tol_scale);
    let (contours_b, _) = extract_fill_contours(path_b, tol_scale);

    if contours_a.is_empty() && contours_b.is_empty() {
        return PathBuilder::new().build(FillType::NonZero);
    }
    if contours_a.is_empty() {
        return match op {
            PathOp::Difference | PathOp::Intersect => PathBuilder::new().build(FillType::NonZero),
            PathOp::Union | PathOp::Xor => contours_to_path(&contours_b, path_b.fill_type),
            PathOp::ReverseDifference => unreachable!(),
        };
    }
    if contours_b.is_empty() {
        return match op {
            PathOp::Intersect => PathBuilder::new().build(FillType::NonZero),
            PathOp::Difference | PathOp::Union | PathOp::Xor => {
                contours_to_path(&contours_a, path_a.fill_type)
            }
            PathOp::ReverseDifference => unreachable!(),
        };
    }

    // 1. Intern vertices and collect raw segments from both paths.
    let mut verts: Vec<Pt2> = Vec::new();
    let mut raw_segments: Vec<(usize, usize)> = Vec::new();

    for contour in contours_a.iter().chain(contours_b.iter()) {
        if contour.len() < 3 {
            continue;
        }
        let mut ids = Vec::with_capacity(contour.len());
        for &pt in contour {
            let id = intern_vertex(&mut verts, pt);
            if ids.last().copied() != Some(id) {
                ids.push(id);
            }
        }
        if ids.len() >= 2 && ids.first() == ids.last() {
            ids.pop();
        }
        if ids.len() < 3 {
            continue;
        }
        for i in 0..ids.len() {
            let u = ids[i];
            let v = ids[(i + 1) % ids.len()];
            if u != v {
                raw_segments.push((u, v));
            }
        }
    }

    if raw_segments.is_empty() {
        return PathBuilder::new().build(FillType::NonZero);
    }

    // 2. Find all pairwise proper segment intersections across both paths and intern them.
    let seg_count = raw_segments.len();
    for i in 0..seg_count {
        let (u1, v1) = raw_segments[i];
        let a = verts[u1];
        let b = verts[v1];
        let min_x1 = a.x.min(b.x) - SNAP_EPS;
        let max_x1 = a.x.max(b.x) + SNAP_EPS;
        let min_y1 = a.y.min(b.y) - SNAP_EPS;
        let max_y1 = a.y.max(b.y) + SNAP_EPS;

        for &(u2, v2) in &raw_segments[(i + 1)..seg_count] {
            if u1 == u2 || u1 == v2 || v1 == u2 || v1 == v2 {
                continue;
            }
            let c = verts[u2];
            let d = verts[v2];
            if c.x.max(d.x) < min_x1
                || c.x.min(d.x) > max_x1
                || c.y.max(d.y) < min_y1
                || c.y.min(d.y) > max_y1
            {
                continue;
            }
            let r = b.sub(a);
            let s = d.sub(c);
            let rxs = r.cross(s);
            if rxs.abs() <= 1e-11 {
                continue;
            }
            let qp = c.sub(a);
            let t = qp.cross(s) / rxs;
            let u = qp.cross(r) / rxs;
            if t > INTERSECT_EPS
                && t < 1.0 - INTERSECT_EPS
                && u > INTERSECT_EPS
                && u < 1.0 - INTERSECT_EPS
            {
                let p_int = a.add(r.scale(t));
                intern_vertex(&mut verts, p_int);
            }
        }
    }

    // 3. Split every segment at all vertices lying on its open interior (including T-junctions
    // and collinear overlaps) into unique undirected atomic sub-edges.
    let mut undirected_edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut split_buf: Vec<(f64, usize)> = Vec::new();

    for &(u, v) in &raw_segments {
        let a = verts[u];
        let b = verts[v];
        let ab = b.sub(a);
        let len_sq = ab.len_sq();
        let len = len_sq.sqrt();
        if len <= 1e-9 {
            continue;
        }
        let min_x = a.x.min(b.x) - SNAP_EPS;
        let max_x = a.x.max(b.x) + SNAP_EPS;
        let min_y = a.y.min(b.y) - SNAP_EPS;
        let max_y = a.y.max(b.y) + SNAP_EPS;

        split_buf.clear();
        for (w, &pw) in verts.iter().enumerate() {
            if w == u || w == v {
                continue;
            }
            if pw.x < min_x || pw.x > max_x || pw.y < min_y || pw.y > max_y {
                continue;
            }
            let aw = pw.sub(a);
            let t = aw.dot(ab) / len_sq;
            if t <= 1e-6 || t >= 1.0 - 1e-6 {
                continue;
            }
            let perp = aw.cross(ab).abs() / len;
            if perp <= SNAP_EPS {
                split_buf.push((t, w));
            }
        }

        split_buf.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut prev = u;
        for &(_, w) in &split_buf {
            if w != prev {
                undirected_edges.insert((prev.min(w), prev.max(w)));
                prev = w;
            }
        }
        if prev != v {
            undirected_edges.insert((prev.min(v), prev.max(v)));
        }
    }

    // 4. Classify each atomic undirected sub-edge by evaluating `op(in_a, in_b)` on its left
    // and right midpoint normals, keeping directed edges with `true` on left and `false` on right.
    let mut boundary_edges: Vec<(usize, usize)> = Vec::new();

    for &(u, v) in &undirected_edges {
        let a = verts[u];
        let b = verts[v];
        let ab = b.sub(a);
        let len = ab.len_sq().sqrt();
        if len <= SNAP_EPS {
            continue;
        }
        let mid = Pt2 {
            x: (a.x + b.x) * 0.5,
            y: (a.y + b.y) * 0.5,
        };
        let delta = (len * 1e-4).clamp(1e-7, 1e-4);
        let nx = -ab.y / len * delta;
        let ny = ab.x / len * delta;

        let left_pt = Pt2 {
            x: mid.x + nx,
            y: mid.y + ny,
        };
        let right_pt = Pt2 {
            x: mid.x - nx,
            y: mid.y - ny,
        };

        let left_in = eval_op(
            op,
            point_in_contours(left_pt, &contours_a, path_a.fill_type),
            point_in_contours(left_pt, &contours_b, path_b.fill_type),
        );
        let right_in = eval_op(
            op,
            point_in_contours(right_pt, &contours_a, path_a.fill_type),
            point_in_contours(right_pt, &contours_b, path_b.fill_type),
        );

        if left_in && !right_in {
            boundary_edges.push((u, v));
        } else if right_in && !left_in {
            boundary_edges.push((v, u));
        }
    }

    if boundary_edges.is_empty() {
        return PathBuilder::new().build(FillType::NonZero);
    }

    // 5. Stitch directed boundary edges into closed polygon contours, picking the immediate
    // clockwise outgoing edge from the reverse incoming vector at touching vertices.
    let mut outgoing: Vec<Vec<(usize, f64)>> = vec![Vec::new(); verts.len()];
    for (e_idx, &(u, v)) in boundary_edges.iter().enumerate() {
        let d = verts[v].sub(verts[u]);
        outgoing[u].push((e_idx, d.y.atan2(d.x)));
    }
    for outs in &mut outgoing {
        outs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    let mut used = vec![false; boundary_edges.len()];
    let mut final_contours: Vec<Vec<Pt2>> = Vec::new();

    for start_e in 0..boundary_edges.len() {
        if used[start_e] {
            continue;
        }
        let start_u = boundary_edges[start_e].0;
        let mut cycle_verts: Vec<Pt2> = Vec::new();
        let mut curr_e = start_e;
        let mut closed = false;

        while !used[curr_e] {
            used[curr_e] = true;
            let (u, v) = boundary_edges[curr_e];
            cycle_verts.push(verts[u]);
            if v == start_u {
                closed = true;
                break;
            }
            let back = verts[u].sub(verts[v]);
            let back_angle = back.y.atan2(back.x);
            let outs = &outgoing[v];
            if outs.is_empty() {
                break;
            }
            let start_pos = outs.partition_point(|&(_, ang)| ang < back_angle - 1e-12);
            let m = outs.len();
            let mut next_edge = None;
            for k in 0..m {
                let idx = (start_pos + m - 1 - k) % m;
                let (cand_e, _) = outs[idx];
                if !used[cand_e] {
                    next_edge = Some(cand_e);
                    break;
                }
            }
            match next_edge {
                Some(ne) => curr_e = ne,
                None => break,
            }
        }

        if closed && cycle_verts.len() >= 3 {
            simplify_contour(&mut cycle_verts);
            if cycle_verts.len() >= 3 && contour_signed_area2(&cycle_verts).abs() > AREA_EPS * 2.0 {
                final_contours.push(cycle_verts);
            }
        }
    }

    contours_to_path(&final_contours, FillType::NonZero)
}
