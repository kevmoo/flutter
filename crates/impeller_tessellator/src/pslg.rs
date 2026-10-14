use crate::convex::{wangs_cubic_subdivisions, wangs_quadratic_subdivisions};
use crate::{FillTessellationKind, TessellatedMesh, TessellationResult, Vertex2D};
use impeller_geometry::{Point, Scalar};
use impeller_path::{CubicPathComponent, FillType, Path, PathComponent, QuadPathComponent};
use std::collections::BTreeMap;

const SNAP_EPS: f64 = 1e-5;
const INTERSECT_EPS: f64 = 1e-7;
const AREA_EPS: f64 = 1e-8;

#[derive(Copy, Clone, Debug, PartialEq)]
struct Pt2 {
    x: f64,
    y: f64,
}

impl Pt2 {
    #[inline]
    fn from_point(p: Point) -> Self {
        Self {
            x: p.x as f64,
            y: p.y as f64,
        }
    }

    #[inline]
    fn to_point(self) -> Point {
        Point::new(self.x as Scalar, self.y as Scalar)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    #[inline]
    fn scale(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }

    #[inline]
    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    fn cross(self, other: Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    #[inline]
    fn len_sq(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    #[inline]
    fn dist_sq(self, other: Self) -> f64 {
        self.sub(other).len_sq()
    }
}

#[inline]
fn solve_quad_f64(q: &QuadPathComponent, t: f64) -> Pt2 {
    let u = 1.0 - t;
    let w0 = u * u;
    let w1 = 2.0 * u * t;
    let w2 = t * t;
    Pt2 {
        x: (q.p1.x as f64) * w0 + (q.cp.x as f64) * w1 + (q.p2.x as f64) * w2,
        y: (q.p1.y as f64) * w0 + (q.cp.y as f64) * w1 + (q.p2.y as f64) * w2,
    }
}

#[inline]
fn solve_cubic_f64(c: &CubicPathComponent, t: f64) -> Pt2 {
    let u = 1.0 - t;
    let u2 = u * u;
    let t2 = t * t;
    let w0 = u2 * u;
    let w1 = 3.0 * u2 * t;
    let w2 = 3.0 * u * t2;
    let w3 = t2 * t;
    Pt2 {
        x: (c.p1.x as f64) * w0
            + (c.cp1.x as f64) * w1
            + (c.cp2.x as f64) * w2
            + (c.p2.x as f64) * w3,
        y: (c.p1.y as f64) * w0
            + (c.cp1.y as f64) * w1
            + (c.cp2.y as f64) * w2
            + (c.p2.y as f64) * w3,
    }
}

/// Extracts flattened fill contours from `Path` matching `PathTessellator::PathToFilledVertices`.
/// Returns `(contours, has_any_segment)`.
fn extract_fill_contours(path: &Path, tolerance_scale: Scalar) -> (Vec<Vec<Pt2>>, bool) {
    let scale = tolerance_scale.max(1.0);
    let mut contours: Vec<Vec<Pt2>> = Vec::new();
    let mut current_contour: Vec<Pt2> = Vec::new();
    let mut contour_origin = Point::ZERO;
    let mut current_point = Point::ZERO;
    let mut contour_has_segments = false;
    let mut has_any_segment = false;

    let finish_contour = |current_contour: &mut Vec<Pt2>, contours: &mut Vec<Vec<Pt2>>| {
        if current_contour.len() >= 3 {
            let mut cleaned: Vec<Pt2> = Vec::with_capacity(current_contour.len());
            for &p in current_contour.iter() {
                if cleaned
                    .last()
                    .is_none_or(|last| last.dist_sq(p) > SNAP_EPS * SNAP_EPS)
                {
                    cleaned.push(p);
                }
            }
            if cleaned.len() >= 2
                && cleaned.first().unwrap().dist_sq(*cleaned.last().unwrap()) <= SNAP_EPS * SNAP_EPS
            {
                cleaned.pop();
            }
            if cleaned.len() >= 3 {
                contours.push(cleaned);
            }
        }
        current_contour.clear();
    };

    for comp in &path.components {
        match *comp {
            PathComponent::Contour(c) => {
                if contour_has_segments {
                    finish_contour(&mut current_contour, &mut contours);
                }
                contour_origin = c.destination;
                current_point = c.destination;
                contour_has_segments = false;
            }
            PathComponent::Linear(l) => {
                has_any_segment = true;
                if !contour_has_segments {
                    current_contour.push(Pt2::from_point(contour_origin));
                    contour_has_segments = true;
                }
                if l.p2 != current_point {
                    current_contour.push(Pt2::from_point(l.p2));
                    current_point = l.p2;
                }
            }
            PathComponent::Quad(q) => {
                has_any_segment = true;
                if !contour_has_segments {
                    current_contour.push(Pt2::from_point(contour_origin));
                    contour_has_segments = true;
                }
                if q.cp == current_point || q.p2 == q.cp {
                    if q.p2 != current_point {
                        current_contour.push(Pt2::from_point(q.p2));
                        current_point = q.p2;
                    }
                } else {
                    let count =
                        wangs_quadratic_subdivisions(scale, q.p1, q.cp, q.p2).ceil() as usize;
                    let count = count.max(1);
                    let inv_count = 1.0 / (count as f64);
                    for i in 1..count {
                        let t = (i as f64) * inv_count;
                        current_contour.push(solve_quad_f64(&q, t));
                    }
                    current_contour.push(Pt2::from_point(q.p2));
                    current_point = q.p2;
                }
            }
            PathComponent::Cubic(c) => {
                has_any_segment = true;
                if !contour_has_segments {
                    current_contour.push(Pt2::from_point(contour_origin));
                    contour_has_segments = true;
                }
                if c.cp1 != current_point || c.cp2 != current_point || c.p2 != current_point {
                    let count =
                        wangs_cubic_subdivisions(scale, c.p1, c.cp1, c.cp2, c.p2).ceil() as usize;
                    let count = count.max(1);
                    let inv_count = 1.0 / (count as f64);
                    for i in 1..count {
                        let t = (i as f64) * inv_count;
                        current_contour.push(solve_cubic_f64(&c, t));
                    }
                    current_contour.push(Pt2::from_point(c.p2));
                    current_point = c.p2;
                }
            }
        }
    }

    if contour_has_segments {
        finish_contour(&mut current_contour, &mut contours);
    }

    (contours, has_any_segment)
}

fn intern_vertex(verts: &mut Vec<Pt2>, p: Pt2) -> usize {
    let eps_sq = SNAP_EPS * SNAP_EPS;
    for (i, v) in verts.iter().enumerate() {
        if v.dist_sq(p) <= eps_sq {
            return i;
        }
    }
    let idx = verts.len();
    verts.push(p);
    idx
}

/// Checks in $O(V)$ whether a single flattened contour is a strictly simple convex polygon
/// (winding number $\pm 1$, uniform turn sign, no self-intersections or star-polygon wraps).
fn is_simple_convex_contour(pts: &[Pt2]) -> bool {
    let n = pts.len();
    if n < 3 {
        return false;
    }

    let eps_sq = SNAP_EPS * SNAP_EPS;
    let mut signed_area2 = 0.0f64;
    let mut non_zero_edge_count = 0usize;
    let mut first_edge: Option<Pt2> = None;
    let mut prev_edge: Option<Pt2> = None;
    let mut has_pos_turn = false;
    let mut has_neg_turn = false;

    let mut first_x_sign: i8 = 0;
    let mut prev_x_sign: i8 = 0;
    let mut x_sign_flips = 0usize;

    let mut first_y_sign: i8 = 0;
    let mut prev_y_sign: i8 = 0;
    let mut y_sign_flips = 0usize;

    for i in 0..n {
        let p0 = pts[i];
        let p1 = pts[(i + 1) % n];
        signed_area2 += p0.cross(p1);
        let e = p1.sub(p0);
        if e.len_sq() <= eps_sq {
            continue;
        }
        non_zero_edge_count += 1;

        if let Some(pe) = prev_edge {
            let cross = pe.cross(e);
            let turn_eps = (2e-4 * (pe.len_sq() * e.len_sq()).sqrt()).max(AREA_EPS);
            if cross > turn_eps {
                has_pos_turn = true;
            } else if cross < -turn_eps {
                has_neg_turn = true;
            } else if pe.dot(e) < 0.0 {
                return false;
            }
            if has_pos_turn && has_neg_turn {
                return false;
            }
        } else {
            first_edge = Some(e);
        }
        prev_edge = Some(e);

        let sx: i8 = if e.x > SNAP_EPS {
            1
        } else if e.x < -SNAP_EPS {
            -1
        } else {
            0
        };
        if sx != 0 {
            if first_x_sign == 0 {
                first_x_sign = sx;
            } else if sx != prev_x_sign {
                x_sign_flips += 1;
            }
            prev_x_sign = sx;
        }

        let sy: i8 = if e.y > SNAP_EPS {
            1
        } else if e.y < -SNAP_EPS {
            -1
        } else {
            0
        };
        if sy != 0 {
            if first_y_sign == 0 {
                first_y_sign = sy;
            } else if sy != prev_y_sign {
                y_sign_flips += 1;
            }
            prev_y_sign = sy;
        }
    }

    if non_zero_edge_count < 3 || signed_area2.abs() <= AREA_EPS * 2.0 {
        return false;
    }

    if let (Some(pe), Some(fe)) = (prev_edge, first_edge) {
        let cross = pe.cross(fe);
        let turn_eps = (2e-4 * (pe.len_sq() * fe.len_sq()).sqrt()).max(AREA_EPS);
        if cross > turn_eps {
            has_pos_turn = true;
        } else if cross < -turn_eps {
            has_neg_turn = true;
        } else if pe.dot(fe) < 0.0 {
            return false;
        }
        if has_pos_turn && has_neg_turn {
            return false;
        }
    }

    if !(has_pos_turn || has_neg_turn) {
        return false;
    }

    if first_x_sign == 0 || first_y_sign == 0 {
        return false;
    }
    if prev_x_sign != first_x_sign {
        x_sign_flips += 1;
    }
    if prev_y_sign != first_y_sign {
        y_sign_flips += 1;
    }

    x_sign_flips <= 2 && y_sign_flips <= 2
}

/// Triangulates a single convex contour in $O(V)$ using an alternating zig-zag strip
/// (`0, 1, n-1, 2, n-2, ...`).
fn triangulate_convex_contour(pts: &[Pt2]) -> TessellatedMesh {
    let n = pts.len();
    if n < 3 {
        return TessellatedMesh::default();
    }
    let vertices: Vec<Vertex2D> = pts
        .iter()
        .map(|p| Vertex2D::from_point(p.to_point()))
        .collect();
    let mut indices = Vec::with_capacity((n - 2) * 3);
    let zig_zag = |k: usize| -> usize {
        if k == 0 {
            0
        } else if k % 2 == 1 {
            k.div_ceil(2)
        } else {
            n - (k / 2)
        }
    };

    for i in 0..(n - 2) {
        let (i0, i1, i2) = if i % 2 == 0 {
            (zig_zag(i), zig_zag(i + 1), zig_zag(i + 2))
        } else {
            (zig_zag(i + 1), zig_zag(i), zig_zag(i + 2))
        };
        let p0 = pts[i0];
        let p1 = pts[i1];
        let p2 = pts[i2];
        if p1.sub(p0).cross(p2.sub(p0)).abs() > AREA_EPS {
            indices.push(i0 as u32);
            indices.push(i1 as u32);
            indices.push(i2 as u32);
        }
    }

    TessellatedMesh { vertices, indices }
}

fn contour_aabb(pts: &[Pt2]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in pts {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    (
        min_x - SNAP_EPS,
        max_x + SNAP_EPS,
        min_y - SNAP_EPS,
        max_y + SNAP_EPS,
    )
}

fn try_tessellate_disjoint_convex(contours: &[Vec<Pt2>]) -> Option<TessellatedMesh> {
    if contours.len() <= 1 || contours.len() > 8 {
        return None;
    }
    let mut aabbs: [(f64, f64, f64, f64); 8] = [(0.0, 0.0, 0.0, 0.0); 8];
    for (i, c) in contours.iter().enumerate() {
        aabbs[i] = contour_aabb(c);
    }
    for i in 0..contours.len() {
        let (min_x1, max_x1, min_y1, max_y1) = aabbs[i];
        for &(min_x2, max_x2, min_y2, max_y2) in &aabbs[(i + 1)..contours.len()] {
            let disjoint = max_x1 < min_x2 || max_x2 < min_x1 || max_y1 < min_y2 || max_y2 < min_y1;
            if !disjoint {
                return None;
            }
        }
    }
    if !contours.iter().all(|c| is_simple_convex_contour(c)) {
        return None;
    }

    let total_verts: usize = contours.iter().map(|c| c.len()).sum();
    let total_indices: usize = contours.iter().map(|c| c.len().saturating_sub(2) * 3).sum();
    let mut vertices = Vec::with_capacity(total_verts);
    let mut indices = Vec::with_capacity(total_indices);

    for c in contours {
        let base = vertices.len() as u32;
        let mesh = triangulate_convex_contour(c);
        vertices.extend_from_slice(&mesh.vertices);
        indices.extend(mesh.indices.into_iter().map(|idx| base + idx));
    }

    Some(TessellatedMesh { vertices, indices })
}

/// Tessellates an arbitrary path (convex, concave, multi-contour with holes, or self-intersecting)
/// using a Planar Straight-Line Graph (PSLG) decomposition and winding-rule face filtering.
pub fn tessellate_pslg(
    path: &Path,
    tolerance_scale: Scalar,
) -> Result<TessellatedMesh, TessellationResult> {
    tessellate_pslg_classified(path, tolerance_scale).map(|(mesh, _)| mesh)
}

/// Tessellates an arbitrary path and returns the classification tier taken (`PslgSimpleConvex`,
/// `PslgDisjointConvex`, or `PslgGeneral`).
pub fn tessellate_pslg_classified(
    path: &Path,
    tolerance_scale: Scalar,
) -> Result<(TessellatedMesh, FillTessellationKind), TessellationResult> {
    let (contours, has_any_segment) = extract_fill_contours(path, tolerance_scale);
    if !has_any_segment {
        return Err(TessellationResult::InputError);
    }
    if contours.is_empty() {
        return Ok((
            TessellatedMesh::default(),
            FillTessellationKind::PslgSimpleConvex,
        ));
    }
    if contours.len() == 1 && is_simple_convex_contour(&contours[0]) {
        return Ok((
            triangulate_convex_contour(&contours[0]),
            FillTessellationKind::PslgSimpleConvex,
        ));
    }
    if let Some(mesh) = try_tessellate_disjoint_convex(&contours) {
        return Ok((mesh, FillTessellationKind::PslgDisjointConvex));
    }

    Ok((
        tessellate_pslg_general(&contours, path.fill_type),
        FillTessellationKind::PslgGeneral,
    ))
}

fn tessellate_pslg_general(contours: &[Vec<Pt2>], fill_type: FillType) -> TessellatedMesh {
    // 1. Intern initial vertices and build directed segments.
    let mut verts: Vec<Pt2> = Vec::new();
    let mut raw_segments: Vec<(usize, usize)> = Vec::new();

    for contour in contours {
        let n = contour.len();
        if n < 3 {
            continue;
        }
        let mut ids = Vec::with_capacity(n);
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
        return TessellatedMesh::default();
    }

    // 2. Find all pairwise proper segment intersections and intern intersection points.
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

    // 3. Split every directed segment at all vertices lying on its open interior.
    let mut edge_weights: BTreeMap<(usize, usize), i32> = BTreeMap::new();
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
                add_directed_edge(&mut edge_weights, prev, w);
                prev = w;
            }
        }
        if prev != v {
            add_directed_edge(&mut edge_weights, prev, v);
        }
    }

    // Remove zero-weight edges.
    let mut edges: Vec<(usize, usize, i32)> = edge_weights
        .into_iter()
        .filter_map(|((u, v), w)| if w != 0 { Some((u, v, w)) } else { None })
        .collect();

    // Iteratively prune any degree-1 vertices if numerical edge cases produced them.
    loop {
        let mut degree = vec![0usize; verts.len()];
        for &(u, v, _) in &edges {
            degree[u] += 1;
            degree[v] += 1;
        }
        let before = edges.len();
        edges.retain(|&(u, v, _)| degree[u] >= 2 && degree[v] >= 2);
        if edges.len() == before {
            break;
        }
    }

    if edges.is_empty() {
        return TessellatedMesh::default();
    }

    // 4. Connected components via Union-Find.
    let mut parent: Vec<usize> = (0..verts.len()).collect();
    for &(u, v, _) in &edges {
        uf_union(&mut parent, u, v);
    }

    // 5. Build half-edge structure and sort outgoing half-edges CCW around each vertex.
    let num_edges = edges.len();
    let mut he_origin = Vec::with_capacity(num_edges * 2);
    let mut he_dest = Vec::with_capacity(num_edges * 2);
    let mut outgoing: Vec<Vec<usize>> = vec![Vec::new(); verts.len()];

    for (e_idx, &(u, v, _)) in edges.iter().enumerate() {
        let h0 = e_idx * 2;
        let h1 = e_idx * 2 + 1;
        he_origin.push(u);
        he_dest.push(v);
        he_origin.push(v);
        he_dest.push(u);
        outgoing[u].push(h0);
        outgoing[v].push(h1);
    }

    for (u, outs) in outgoing.iter_mut().enumerate() {
        let pu = verts[u];
        outs.sort_by(|&h_a, &h_b| {
            let pa = verts[he_dest[h_a]].sub(pu);
            let pb = verts[he_dest[h_b]].sub(pu);
            let ang_a = pa.y.atan2(pa.x);
            let ang_b = pb.y.atan2(pb.x);
            ang_a
                .partial_cmp(&ang_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let mut next_he = vec![0usize; num_edges * 2];
    for h in 0..(num_edges * 2) {
        let v = he_dest[h];
        let twin = h ^ 1;
        let outs = &outgoing[v];
        let pos = outs.iter().position(|&x| x == twin).unwrap_or(0);
        let prev_ccw = if pos == 0 { outs.len() - 1 } else { pos - 1 };
        next_he[h] = outs[prev_ccw];
    }

    // 6. Trace all half-edge cycles.
    let mut visited = vec![false; num_edges * 2];
    let mut bounded_faces: Vec<BoundedFace> = Vec::new();
    let mut hole_cycles: Vec<HoleCycle> = Vec::new();

    for start_h in 0..(num_edges * 2) {
        if visited[start_h] {
            continue;
        }
        let mut cycle = Vec::new();
        let mut cur = start_h;
        while !visited[cur] {
            visited[cur] = true;
            cycle.push(he_origin[cur]);
            cur = next_he[cur];
        }
        if cycle.len() < 3 {
            continue;
        }
        let area2 = compute_signed_area2(&cycle, &verts);
        let comp_id = uf_find(&mut parent, cycle[0]);
        if area2 > AREA_EPS {
            bounded_faces.push(BoundedFace {
                comp_id,
                cycle,
                area2,
                holes: Vec::new(),
            });
        } else if area2 < -AREA_EPS {
            hole_cycles.push(HoleCycle {
                comp_id,
                cycle,
                area2_abs: -area2,
            });
        }
    }

    // 7. Assign outer CW cycles of inner components as holes of their immediate enclosing CCW face.
    for hole in hole_cycles {
        let sample_pt = verts[hole.cycle[0]];
        let mut best_face_idx: Option<usize> = None;
        let mut best_area2 = f64::INFINITY;

        for (f_idx, face) in bounded_faces.iter().enumerate() {
            if face.comp_id == hole.comp_id || face.area2 <= hole.area2_abs {
                continue;
            }
            if face.area2 < best_area2 && point_in_cycle(sample_pt, &face.cycle, &verts) {
                best_area2 = face.area2;
                best_face_idx = Some(f_idx);
            }
        }

        if let Some(f_idx) = best_face_idx {
            bounded_faces[f_idx].holes.push(hole.cycle);
        }
    }

    // 8. For each bounded face, bridge holes, triangulate, and test winding number!
    let mut out_vertices: Vec<Vertex2D> = Vec::new();
    let mut vert_remap: Vec<Option<u32>> = vec![None; verts.len()];
    let mut out_indices: Vec<u32> = Vec::new();

    for face in bounded_faces {
        let merged_poly = bridge_holes(face.cycle, face.holes, &verts);
        let tris = ear_clip_polygon(&merged_poly, &verts);
        if tris.is_empty() {
            continue;
        }

        // Pick centroid of the largest-area triangle in `tris` to evaluate exact winding number.
        let mut best_tri = tris[0];
        let mut best_tri_area = -1.0f64;
        for &tri in &tris {
            let a = verts[tri[0]];
            let b = verts[tri[1]];
            let c = verts[tri[2]];
            let area = b.sub(a).cross(c.sub(a)).abs();
            if area > best_tri_area {
                best_tri_area = area;
                best_tri = tri;
            }
        }

        let centroid = Pt2 {
            x: (verts[best_tri[0]].x + verts[best_tri[1]].x + verts[best_tri[2]].x) / 3.0,
            y: (verts[best_tri[0]].y + verts[best_tri[1]].y + verts[best_tri[2]].y) / 3.0,
        };

        let winding = compute_winding_number(centroid, &edges, &verts);
        let is_filled = match fill_type {
            FillType::NonZero => winding != 0,
            FillType::Odd => winding.rem_euclid(2) != 0,
        };

        if !is_filled {
            continue;
        }

        for tri in tris {
            for &vid in &tri {
                let mapped = match vert_remap[vid] {
                    Some(idx) => idx,
                    None => {
                        let idx = out_vertices.len() as u32;
                        out_vertices.push(Vertex2D::from_point(verts[vid].to_point()));
                        vert_remap[vid] = Some(idx);
                        idx
                    }
                };
                out_indices.push(mapped);
            }
        }
    }

    TessellatedMesh {
        vertices: out_vertices,
        indices: out_indices,
    }
}

struct BoundedFace {
    comp_id: usize,
    cycle: Vec<usize>,
    area2: f64,
    holes: Vec<Vec<usize>>,
}

struct HoleCycle {
    comp_id: usize,
    cycle: Vec<usize>,
    area2_abs: f64,
}

#[inline]
fn add_directed_edge(map: &mut BTreeMap<(usize, usize), i32>, u: usize, v: usize) {
    if u < v {
        *map.entry((u, v)).or_insert(0) += 1;
    } else if v < u {
        *map.entry((v, u)).or_insert(0) -= 1;
    }
}

fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn uf_union(parent: &mut [usize], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

fn compute_signed_area2(cycle: &[usize], verts: &[Pt2]) -> f64 {
    let n = cycle.len();
    let mut sum = 0.0;
    for i in 0..n {
        let p = verts[cycle[i]];
        let q = verts[cycle[(i + 1) % n]];
        sum += p.x * q.y - q.x * p.y;
    }
    sum
}

fn point_in_cycle(pt: Pt2, cycle: &[usize], verts: &[Pt2]) -> bool {
    let mut winding = 0i32;
    let n = cycle.len();
    for i in 0..n {
        let a = verts[cycle[i]];
        let b = verts[cycle[(i + 1) % n]];
        if a.y <= pt.y {
            if b.y > pt.y && b.sub(a).cross(pt.sub(a)) > 0.0 {
                winding += 1;
            }
        } else if b.y <= pt.y && b.sub(a).cross(pt.sub(a)) < 0.0 {
            winding -= 1;
        }
    }
    winding != 0
}

fn compute_winding_number(pt: Pt2, edges: &[(usize, usize, i32)], verts: &[Pt2]) -> i32 {
    let mut winding = 0i32;
    for &(u, v, weight) in edges {
        let a = verts[u];
        let b = verts[v];
        if a.y <= pt.y {
            if b.y > pt.y && b.sub(a).cross(pt.sub(a)) > 0.0 {
                winding += weight;
            }
        } else if b.y <= pt.y && b.sub(a).cross(pt.sub(a)) < 0.0 {
            winding -= weight;
        }
    }
    winding
}

/// Bridges CW `holes` into CCW `outer` polygon using Eberly's rightmost-vertex ray-casting method.
fn bridge_holes(mut poly: Vec<usize>, mut holes: Vec<Vec<usize>>, verts: &[Pt2]) -> Vec<usize> {
    if holes.is_empty() {
        return poly;
    }

    // Sort holes by maximum x-coordinate descending.
    holes.sort_by(|h1, h2| {
        let max_x1 = h1
            .iter()
            .map(|&v| verts[v].x)
            .fold(f64::NEG_INFINITY, f64::max);
        let max_x2 = h2
            .iter()
            .map(|&v| verts[v].x)
            .fold(f64::NEG_INFINITY, f64::max);
        max_x2
            .partial_cmp(&max_x1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for hole in holes {
        if hole.len() < 3 {
            continue;
        }
        // Find rightmost vertex of hole.
        let mut h_idx = 0;
        for i in 1..hole.len() {
            let p = verts[hole[i]];
            let best = verts[hole[h_idx]];
            if p.x > best.x || ((p.x - best.x).abs() <= 1e-12 && p.y < best.y) {
                h_idx = i;
            }
        }

        let m_vert = hole[h_idx];
        let m_pt = verts[m_vert];

        // Find closest ray intersection (+x from m_pt) with edges of `poly`.
        let n = poly.len();
        let mut best_ix = f64::INFINITY;
        let mut best_edge = 0usize;

        for i in 0..n {
            let j = (i + 1) % n;
            let a = verts[poly[i]];
            let b = verts[poly[j]];

            if (a.y - m_pt.y).abs() <= 1e-11 && (b.y - m_pt.y).abs() <= 1e-11 {
                // Horizontal edge on y == m_pt.y
                let min_ex = a.x.min(b.x);
                let max_ex = a.x.max(b.x);
                if max_ex >= m_pt.x - 1e-9 {
                    let ix = min_ex.max(m_pt.x);
                    if ix < best_ix {
                        best_ix = ix;
                        best_edge = i;
                    }
                }
                continue;
            }

            // Check if segment straddles m_pt.y (Wait: only consider upward or straddling edges that bound the interior to the left)
            let straddles = (a.y <= m_pt.y && b.y > m_pt.y) || (b.y <= m_pt.y && a.y > m_pt.y);
            if !straddles {
                continue;
            }
            let t = (m_pt.y - a.y) / (b.y - a.y);
            let ix = a.x + t * (b.x - a.x);
            if ix >= m_pt.x - 1e-9 && ix < best_ix {
                best_ix = ix;
                best_edge = i;
            }
        }

        // Determine visible vertex on `poly`.
        let e_i = best_edge;
        let e_j = (best_edge + 1) % n;
        let a = verts[poly[e_i]];
        let b = verts[poly[e_j]];

        let mut cand_idx = if (a.y - m_pt.y).abs() <= 1e-9 && (a.x - best_ix).abs() <= 1e-9 {
            e_i
        } else if (b.y - m_pt.y).abs() <= 1e-9 && (b.x - best_ix).abs() <= 1e-9 {
            e_j
        } else if a.x >= b.x {
            e_i
        } else {
            e_j
        };

        let i_pt = Pt2 {
            x: best_ix,
            y: m_pt.y,
        };
        let c_pt = verts[poly[cand_idx]];

        if (c_pt.y - m_pt.y).abs() > 1e-9 {
            let mut min_angle = f64::INFINITY;
            let mut min_dist_sq = f64::INFINITY;

            for (k, &vk) in poly.iter().enumerate() {
                if vk == poly[cand_idx] || vk == m_vert {
                    continue;
                }
                let r_pt = verts[vk];
                if point_in_triangle_inclusive(r_pt, m_pt, i_pt, c_pt, 1e-11) {
                    let d = r_pt.sub(m_pt);
                    let angle = d.y.abs().atan2(d.x);
                    let dist_sq = d.len_sq();
                    if angle < min_angle - 1e-11
                        || ((angle - min_angle).abs() <= 1e-11 && dist_sq < min_dist_sq)
                    {
                        min_angle = angle;
                        min_dist_sq = dist_sq;
                        cand_idx = k;
                    }
                }
            }
        }

        // Splice `hole` into `poly` at `cand_idx`.
        let mut new_poly = Vec::with_capacity(poly.len() + hole.len() + 2);
        new_poly.extend_from_slice(&poly[0..=cand_idx]);
        new_poly.extend_from_slice(&hole[h_idx..]);
        new_poly.extend_from_slice(&hole[0..=h_idx]);
        new_poly.extend_from_slice(&poly[cand_idx..]);
        poly = new_poly;
    }

    poly
}

fn clean_polygon_spikes(poly: &mut Vec<usize>, verts: &[Pt2]) {
    loop {
        let before = poly.len();
        if before < 3 {
            break;
        }
        // Remove consecutive identical vertices.
        let mut dedup = Vec::with_capacity(poly.len());
        for &v in poly.iter() {
            if dedup.last().copied() != Some(v) {
                dedup.push(v);
            }
        }
        if dedup.len() >= 2 && dedup.first() == dedup.last() {
            dedup.pop();
        }
        *poly = dedup;
        if poly.len() < 3 {
            break;
        }

        // Remove zero-area spikes (prev == next) or strictly collinear intermediate vertices.
        let mut found = false;
        let n = poly.len();
        for i in 0..n {
            let prev = poly[(i + n - 1) % n];
            let curr = poly[i];
            let next = poly[(i + 1) % n];
            if prev == next {
                poly.remove(i);
                found = true;
                break;
            }
            let a = verts[prev];
            let b = verts[curr];
            let c = verts[next];
            let cross = b.sub(a).cross(c.sub(b));
            let dot = b.sub(a).dot(c.sub(b));
            if cross.abs() <= 1e-11 && dot > 0.0 {
                poly.remove(i);
                found = true;
                break;
            }
        }
        if !found && poly.len() == before {
            break;
        }
    }
}

#[inline]
fn is_reflex_at(poly: &[usize], k: usize, verts: &[Pt2]) -> bool {
    let n = poly.len();
    let prev = poly[(k + n - 1) % n];
    let curr = poly[k];
    let next = poly[(k + 1) % n];
    if prev == curr || curr == next || prev == next {
        return true;
    }
    let a = verts[prev];
    let b = verts[curr];
    let c = verts[next];
    b.sub(a).cross(c.sub(a)) <= 1e-10
}

#[inline]
fn remove_poly_vertex(
    poly: &mut Vec<usize>,
    is_reflex: &mut Vec<bool>,
    reflex_verts: &mut Vec<usize>,
    idx: usize,
) {
    let vid = poly.remove(idx);
    let was_reflex = is_reflex.remove(idx);
    if was_reflex {
        if let Some(pos) = reflex_verts.iter().position(|&v| v == vid) {
            reflex_verts.swap_remove(pos);
        }
    }
}

#[inline]
fn refresh_reflex_at(
    poly: &[usize],
    is_reflex: &mut [bool],
    reflex_verts: &mut Vec<usize>,
    k: usize,
    verts: &[Pt2],
) {
    let new_r = is_reflex_at(poly, k, verts);
    let old_r = is_reflex[k];
    if old_r != new_r {
        is_reflex[k] = new_r;
        let vid = poly[k];
        if new_r {
            reflex_verts.push(vid);
        } else if let Some(pos) = reflex_verts.iter().position(|&v| v == vid) {
            reflex_verts.swap_remove(pos);
        }
    }
}

fn clean_local_around(
    poly: &mut Vec<usize>,
    is_reflex: &mut Vec<bool>,
    reflex_verts: &mut Vec<usize>,
    mut pivot: usize,
    verts: &[Pt2],
) -> usize {
    while poly.len() >= 3 {
        let m = poly.len();
        pivot %= m;
        let candidates = [(pivot + m - 1) % m, pivot, (pivot + 1) % m];
        let mut removed = None;
        for &idx in &candidates {
            let prev = poly[(idx + m - 1) % m];
            let curr = poly[idx];
            let next = poly[(idx + 1) % m];
            if curr == next || prev == curr || prev == next {
                removed = Some(idx);
                break;
            }
            let a = verts[prev];
            let b = verts[curr];
            let c = verts[next];
            let ab = b.sub(a);
            let bc = c.sub(b);
            if ab.cross(bc).abs() <= 1e-11 && ab.dot(bc) > 0.0 {
                removed = Some(idx);
                break;
            }
        }
        if let Some(idx) = removed {
            remove_poly_vertex(poly, is_reflex, reflex_verts, idx);
            if !poly.is_empty() {
                pivot = idx % poly.len();
            }
        } else {
            break;
        }
    }
    if poly.len() >= 3 {
        let m = poly.len();
        pivot %= m;
        let p0 = (pivot + m - 1) % m;
        let p1 = pivot;
        let p2 = (pivot + 1) % m;
        refresh_reflex_at(poly, is_reflex, reflex_verts, p0, verts);
        if p1 != p0 {
            refresh_reflex_at(poly, is_reflex, reflex_verts, p1, verts);
        }
        if p2 != p0 && p2 != p1 {
            refresh_reflex_at(poly, is_reflex, reflex_verts, p2, verts);
        }
    }
    pivot
}

fn ear_clip_polygon(initial_poly: &[usize], verts: &[Pt2]) -> Vec<[usize; 3]> {
    let mut poly = initial_poly.to_vec();
    clean_polygon_spikes(&mut poly, verts);
    if poly.len() < 3 {
        return Vec::new();
    }

    let n_init = poly.len();
    let mut is_reflex: Vec<bool> = (0..n_init).map(|k| is_reflex_at(&poly, k, verts)).collect();
    let mut reflex_verts: Vec<usize> = poly
        .iter()
        .enumerate()
        .filter_map(|(k, &vid)| if is_reflex[k] { Some(vid) } else { None })
        .collect();

    let mut tris = Vec::with_capacity(n_init - 2);
    let mut guard = n_init * 4 + 16;
    let mut cursor = 0usize;

    while poly.len() > 3 && guard > 0 {
        guard -= 1;

        let n = poly.len();
        cursor %= n;
        let mut best_ear: Option<usize> = None;
        let mut best_score = -1.0f64;

        for offset in 0..n {
            let i = (cursor + offset) % n;
            if is_reflex[i] {
                continue;
            }
            let prev = poly[(i + n - 1) % n];
            let curr = poly[i];
            let next = poly[(i + 1) % n];
            if prev == curr || curr == next || prev == next {
                continue;
            }
            let a = verts[prev];
            let b = verts[curr];
            let c = verts[next];
            let cross = b.sub(a).cross(c.sub(a));
            if cross <= 1e-11 {
                continue;
            }
            let perim_sq = a.dist_sq(b) + b.dist_sq(c) + c.dist_sq(a);
            let score = if perim_sq > 0.0 {
                cross / perim_sq
            } else {
                cross
            };
            if score <= best_score {
                continue;
            }

            if is_valid_ear(i, &poly, &reflex_verts, verts, [a, b, c], 1e-10) {
                best_score = score;
                best_ear = Some(i);
                if score >= 1e-3 {
                    break;
                }
            }
        }

        // Relaxed fallback if strict tolerance rejected all ears due to near-collinear bridge edges.
        if best_ear.is_none() {
            for offset in 0..n {
                let i = (cursor + offset) % n;
                let prev = poly[(i + n - 1) % n];
                let curr = poly[i];
                let next = poly[(i + 1) % n];
                if prev == curr || curr == next || prev == next {
                    continue;
                }
                let a = verts[prev];
                let b = verts[curr];
                let c = verts[next];
                let cross = b.sub(a).cross(c.sub(a));
                if cross > 0.0 && is_valid_ear(i, &poly, &reflex_verts, verts, [a, b, c], -1e-9) {
                    best_ear = Some(i);
                    break;
                }
            }
        }

        if let Some(ear_idx) = best_ear {
            let n = poly.len();
            let prev = poly[(ear_idx + n - 1) % n];
            let curr = poly[ear_idx];
            let next = poly[(ear_idx + 1) % n];
            tris.push([prev, curr, next]);
            remove_poly_vertex(&mut poly, &mut is_reflex, &mut reflex_verts, ear_idx);
            if !poly.is_empty() {
                let pivot = ear_idx % poly.len();
                cursor =
                    clean_local_around(&mut poly, &mut is_reflex, &mut reflex_verts, pivot, verts);
            }
        } else {
            break;
        }
    }

    if poly.len() == 3 {
        let a = verts[poly[0]];
        let b = verts[poly[1]];
        let c = verts[poly[2]];
        if b.sub(a).cross(c.sub(a)).abs() > 1e-11
            && poly[0] != poly[1]
            && poly[1] != poly[2]
            && poly[0] != poly[2]
        {
            tris.push([poly[0], poly[1], poly[2]]);
        }
    }

    tris
}

fn is_valid_ear(
    i: usize,
    poly: &[usize],
    reflex_verts: &[usize],
    verts: &[Pt2],
    [a, b, c]: [Pt2; 3],
    eps: f64,
) -> bool {
    let n = poly.len();
    let prev = poly[(i + n - 1) % n];
    let curr = poly[i];
    let next = poly[(i + 1) % n];

    let pad = eps.abs().max(1e-8);
    let tri_min_x = a.x.min(b.x).min(c.x) - pad;
    let tri_max_x = a.x.max(b.x).max(c.x) + pad;
    let tri_min_y = a.y.min(b.y).min(c.y) - pad;
    let tri_max_y = a.y.max(b.y).max(c.y) + pad;

    // 1. Check no distinct reflex vertex lies inside or on triangle (a, b, c).
    for &vk in reflex_verts {
        if vk == prev || vk == curr || vk == next {
            continue;
        }
        let p = verts[vk];
        if p.x < tri_min_x || p.x > tri_max_x || p.y < tri_min_y || p.y > tri_max_y {
            continue;
        }
        if point_in_triangle_inclusive(p, a, b, c, eps) {
            return false;
        }
    }

    // 2. Check that diagonal (prev, next) does not properly intersect any polygon edge.
    if !reflex_verts.is_empty() {
        let diag_min_x = a.x.min(c.x) - 1e-8;
        let diag_max_x = a.x.max(c.x) + 1e-8;
        let diag_min_y = a.y.min(c.y) - 1e-8;
        let diag_max_y = a.y.max(c.y) + 1e-8;

        for e in 0..n {
            let u = poly[e];
            let v = poly[(e + 1) % n];
            if u == prev || u == next || v == prev || v == next {
                continue;
            }
            let pu = verts[u];
            let pv = verts[v];
            if pu.x.max(pv.x) < diag_min_x
                || pu.x.min(pv.x) > diag_max_x
                || pu.y.max(pv.y) < diag_min_y
                || pu.y.min(pv.y) > diag_max_y
            {
                continue;
            }
            if segments_properly_intersect(a, c, pu, pv) {
                return false;
            }
        }
    }

    true
}

fn point_in_triangle_inclusive(p: Pt2, a: Pt2, b: Pt2, c: Pt2, eps: f64) -> bool {
    let area = b.sub(a).cross(c.sub(a));
    let (a, b, c) = if area < 0.0 { (a, c, b) } else { (a, b, c) };
    let c1 = b.sub(a).cross(p.sub(a));
    let c2 = c.sub(b).cross(p.sub(b));
    let c3 = a.sub(c).cross(p.sub(c));
    c1 >= -eps && c2 >= -eps && c3 >= -eps
}

fn segments_properly_intersect(a: Pt2, b: Pt2, c: Pt2, d: Pt2) -> bool {
    let r = b.sub(a);
    let s = d.sub(c);
    let rxs = r.cross(s);
    if rxs.abs() <= 1e-11 {
        return false;
    }
    let qp = c.sub(a);
    let t = qp.cross(s) / rxs;
    let u = qp.cross(r) / rxs;
    t > 1e-8 && t < 1.0 - 1e-8 && u > 1e-8 && u < 1.0 - 1e-8
}
