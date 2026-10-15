use crate::*;

fn assert_scalar_near(a: Scalar, b: Scalar) {
    assert!(
        (a - b).abs() <= K_EH_CLOSE_ENOUGH,
        "scalar mismatch: left={a}, right={b}, diff={}",
        (a - b).abs()
    );
}

fn assert_point_near(a: Point, b: Point) {
    assert!(
        (a.x - b.x).abs() <= K_EH_CLOSE_ENOUGH && (a.y - b.y).abs() <= K_EH_CLOSE_ENOUGH,
        "point mismatch: left={a:?}, right={b:?}"
    );
}

fn assert_vector3_near(a: Vector3, b: Vector3) {
    assert!(
        (a.x - b.x).abs() <= K_EH_CLOSE_ENOUGH
            && (a.y - b.y).abs() <= K_EH_CLOSE_ENOUGH
            && (a.z - b.z).abs() <= K_EH_CLOSE_ENOUGH,
        "vector3 mismatch: left={a:?}, right={b:?}"
    );
}

fn assert_vector4_near(a: Vector4, b: Vector4) {
    assert!(
        (a.x - b.x).abs() <= K_EH_CLOSE_ENOUGH
            && (a.y - b.y).abs() <= K_EH_CLOSE_ENOUGH
            && (a.z - b.z).abs() <= K_EH_CLOSE_ENOUGH
            && (a.w - b.w).abs() <= K_EH_CLOSE_ENOUGH,
        "vector4 mismatch: left={a:?}, right={b:?}"
    );
}

fn assert_quaternion_near(a: Quaternion, b: Quaternion) {
    assert!(
        (a.x - b.x).abs() <= K_EH_CLOSE_ENOUGH
            && (a.y - b.y).abs() <= K_EH_CLOSE_ENOUGH
            && (a.z - b.z).abs() <= K_EH_CLOSE_ENOUGH
            && (a.w - b.w).abs() <= K_EH_CLOSE_ENOUGH,
        "quaternion mismatch: left={a:?}, right={b:?}"
    );
}

fn assert_matrix_near(a: Matrix, b: Matrix) {
    for i in 0..16 {
        assert!(
            (a.m[i] - b.m[i]).abs() <= K_EH_CLOSE_ENOUGH,
            "matrix[{i}] mismatch: left={}, right={}, full left={a:?}, full right={b:?}",
            a.m[i],
            b.m[i]
        );
    }
}

fn assert_color_near(a: Color, b: Color) {
    assert!(
        (a.red - b.red).abs() <= K_EH_CLOSE_ENOUGH
            && (a.green - b.green).abs() <= K_EH_CLOSE_ENOUGH
            && (a.blue - b.blue).abs() <= K_EH_CLOSE_ENOUGH
            && (a.alpha - b.alpha).abs() <= K_EH_CLOSE_ENOUGH,
        "color mismatch: left={a:?}, right={b:?}"
    );
}

fn assert_rect_near(a: Rect, b: Rect) {
    assert_scalar_near(a.get_left(), b.get_left());
    assert_scalar_near(a.get_top(), b.get_top());
    assert_scalar_near(a.get_right(), b.get_right());
    assert_scalar_near(a.get_bottom(), b.get_bottom());
}

/// C++ source: wangs_formula_unittests.cc:Cubic
#[test]
fn wangs_formula_cubic() {
    let p0 = Point::new(300.0, 0.0);
    let p1 = Point::new(0.0, 0.0);
    let p2 = Point::new(0.0, 0.0);
    let p3 = Point::new(0.0, 300.0);
    let result = compute_cubic_subdivisions(1.0, p0, p1, p2, p3);
    assert_eq!(result, 30.0);
}

/// C++ source: wangs_formula_unittests.cc:Quadratic
#[test]
fn wangs_formula_quadratic() {
    let p0 = Point::new(15.0, 0.0);
    let p1 = Point::new(0.0, 0.0);
    let p2 = Point::new(0.0, 20.0);
    let result = compute_quadratic_subdivisions(1.0, p0, p1, p2);
    assert_eq!(result, 5.0);
    assert_eq!(compute_quadrature_subdivisions(1.0, p0, p1, p2), 5.0);
}

/// C++ source: geometry_unittests.cc:ScalarNearlyEqual
#[test]
fn scalar_nearly_equal_test() {
    assert!(!scalar_nearly_equal(0.0021, 0.001, K_EH_CLOSE_ENOUGH));
    assert!(scalar_nearly_equal(0.0019, 0.001, K_EH_CLOSE_ENOUGH));
    assert!(scalar_nearly_equal(0.002, 0.001, 0.0011));
    assert!(!scalar_nearly_equal(0.002, 0.001, 0.0009));
    assert!(scalar_nearly_equal(
        1.0,
        1.0 + f32::EPSILON * 4.0,
        K_EH_CLOSE_ENOUGH
    ));
}

/// C++ source: geometry_unittests.cc:CanConvertBetweenDegressAndRadians
#[test]
fn can_convert_between_degrees_and_radians() {
    let deg = Degrees(90.0);
    let rad: Radians = deg.into();
    assert_scalar_near(rad.0, K_PI_OVER_2);
    let back: Degrees = rad.into();
    assert_scalar_near(back.0, 90.0);
}

/// C++ source: trig_unittests.cc:TrigAngles
#[test]
fn trig_angles() {
    let t0 = Trig::from_degrees(Degrees(0.0));
    assert_eq!(t0.cos, 1.0);
    assert_eq!(t0.sin, 0.0);

    let r0 = Trig::new(Radians(0.0));
    assert_eq!(r0.cos, 1.0);
    assert_eq!(r0.sin, 0.0);

    let t30 = Trig::from_degrees(Degrees(30.0));
    assert!((t30.cos - 0.75_f64.sqrt()).abs() <= K_EH_CLOSE_ENOUGH as f64);
    assert!((t30.sin - 0.5).abs() <= K_EH_CLOSE_ENOUGH as f64);

    let t60 = Trig::from_degrees(Degrees(60.0));
    assert!((t60.cos - 0.5).abs() <= K_EH_CLOSE_ENOUGH as f64);
    assert!((t60.sin - 0.75_f64.sqrt()).abs() <= K_EH_CLOSE_ENOUGH as f64);

    let t90 = Trig::from_degrees(Degrees(90.0));
    assert!(t90.cos.abs() <= K_EH_CLOSE_ENOUGH as f64);
    assert!((t90.sin - 1.0).abs() <= K_EH_CLOSE_ENOUGH as f64);
}

/// C++ source: trig_unittests.cc:MultiplyByScalarRadius
#[test]
fn trig_multiply_by_scalar_radius() {
    for deg in (0..=360).step_by(30) {
        let radius = 15.0_f32;
        let trig = Trig::from_degrees(Degrees(deg as Scalar));
        let p = trig * radius;
        let rad = (deg as Scalar) * K_PI / 180.0;
        assert_point_near(p, Point::new(radius * rad.cos(), radius * rad.sin()));
    }
}

/// C++ source: point_unittests.cc:Length
#[test]
fn point_length() {
    for i in 0..21 {
        let fi = i as Scalar;
        assert_eq!(Point::new(fi, 0.0).get_length_squared(), fi * fi);
        assert_eq!(Point::new(0.0, fi).get_length_squared(), fi * fi);
        assert_eq!(Point::new(-fi, 0.0).get_length_squared(), fi * fi);
        assert_eq!(Point::new(0.0, -fi).get_length_squared(), fi * fi);

        assert_eq!(Point::new(fi, 0.0).get_length(), fi);
        assert_eq!(Point::new(0.0, fi).get_length(), fi);

        assert_eq!(Point::new(fi, fi).get_length_squared(), 2.0 * fi * fi);
        assert_scalar_near(Point::new(fi, fi).get_length(), K_SQRT_2 * fi);
    }
}

/// C++ source: point_unittests.cc:Distance
#[test]
fn point_distance() {
    for j in 0..10 {
        for i in 0..10 {
            let fi = i as Scalar;
            let fj = j as Scalar;
            let d_sq = fi * fi + fj * fj;
            assert_eq!(
                Point::new(fi, 0.0).get_distance_squared(Point::new(0.0, fj)),
                d_sq
            );
            assert_scalar_near(
                Point::new(fi, 0.0).get_distance(Point::new(0.0, fj)),
                d_sq.sqrt(),
            );
        }
    }
}

/// C++ source: point_unittests.cc:PerpendicularLeft
#[test]
fn point_perpendicular_left() {
    assert_eq!(
        Point::new(1.0, 0.0).perpendicular_left(),
        Point::new(0.0, -1.0)
    );
    assert_eq!(
        Point::new(0.0, 1.0).perpendicular_left(),
        Point::new(1.0, 0.0)
    );
    assert_eq!(
        Point::new(-1.0, 0.0).perpendicular_left(),
        Point::new(0.0, 1.0)
    );
    assert_eq!(
        Point::new(0.0, -1.0).perpendicular_left(),
        Point::new(-1.0, 0.0)
    );
}

/// C++ source: point_unittests.cc:PerpendicularRight
#[test]
fn point_perpendicular_right() {
    assert_eq!(
        Point::new(1.0, 0.0).perpendicular_right(),
        Point::new(0.0, 1.0)
    );
    assert_eq!(
        Point::new(0.0, 1.0).perpendicular_right(),
        Point::new(-1.0, 0.0)
    );
    assert_eq!(
        Point::new(-1.0, 0.0).perpendicular_right(),
        Point::new(0.0, -1.0)
    );
    assert_eq!(
        Point::new(0.0, -1.0).perpendicular_right(),
        Point::new(1.0, 0.0)
    );
}

/// C++ source: point_unittests.cc:PointToSegment
#[test]
fn point_to_segment() {
    let seg0 = Point::new(10.0, 10.0);
    let seg1 = Point::new(20.0, 10.0);
    // Left of segment
    assert_scalar_near(
        Point::new(5.0, 10.0).get_distance_to_segment(seg0, seg1),
        5.0,
    );
    // On segment
    assert_scalar_near(
        Point::new(15.0, 10.0).get_distance_to_segment(seg0, seg1),
        0.0,
    );
    // 5 pixels above segment
    assert_scalar_near(
        Point::new(15.0, 5.0).get_distance_to_segment(seg0, seg1),
        5.0,
    );
    assert_scalar_near(
        Point::new(15.0, 5.0).get_distance_to_segment_squared(seg0, seg1),
        25.0,
    );
}

/// C++ source: point_unittests.cc:CrossProductThreePoints
#[test]
fn cross_product_three_points() {
    assert_scalar_near(
        Point::cross_three_points(
            Point::new(-1.0, 0.0),
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
        ),
        0.0,
    );
    assert_scalar_near(
        Point::cross_three_points(
            Point::new(-1.0, 0.0),
            Point::new(0.0, 0.0),
            Point::new(0.0, 1.0),
        ),
        1.0,
    );
    assert_scalar_near(
        Point::cross_three_points(
            Point::new(-1.0, 0.0),
            Point::new(0.0, 0.0),
            Point::new(0.0, -1.0),
        ),
        -1.0,
    );
    assert_scalar_near(
        Point::cross_three_points(
            Point::new(3.0, -4.0),
            Point::new(0.0, 0.0),
            Point::new(1.0, 2.0),
        ),
        -10.0,
    );
}

/// C++ source: geometry_unittests.cc:PointDotProduct
#[test]
fn point_dot_product() {
    assert_scalar_near(Point::new(1.0, 0.0).dot(Point::new(-1.0, 0.0)), -1.0);
    assert_scalar_near(Point::new(0.0, -1.0).dot(Point::new(-1.0, 0.0)), 0.0);
    assert_scalar_near(Point::new(1.0, 2.0).dot(Point::new(3.0, -4.0)), -5.0);
}

/// C++ source: geometry_unittests.cc:PointCrossProduct
#[test]
fn point_cross_product() {
    assert_scalar_near(Point::new(1.0, 0.0).cross(Point::new(-1.0, 0.0)), 0.0);
    assert_scalar_near(Point::new(0.0, -1.0).cross(Point::new(-1.0, 0.0)), -1.0);
    assert_scalar_near(Point::new(1.0, 2.0).cross(Point::new(3.0, -4.0)), -10.0);
}

/// C++ source: geometry_unittests.cc:PointReflect
#[test]
fn point_reflect() {
    let axis = Point::new(0.0, 1.0);
    assert_point_near(Point::new(2.0, 3.0).reflect(axis), Point::new(2.0, -3.0));

    let axis2 = Point::new(1.0, 1.0).normalize();
    assert_point_near(Point::new(1.0, 0.0).reflect(axis2), Point::new(0.0, -1.0));
}

/// C++ source: geometry_unittests.cc:PointRotate
#[test]
fn point_rotate() {
    let a = Point::new(1.0, 0.0);
    assert_point_near(a.rotate(Radians(K_PI_OVER_2)), Point::new(0.0, 1.0));
    assert_point_near(a.rotate(Radians(-K_PI_OVER_2)), Point::new(0.0, -1.0));
    assert_point_near(a.rotate(Radians(K_PI)), Point::new(-1.0, 0.0));
}

/// C++ source: geometry_unittests.cc:PointAngleTo
#[test]
fn point_angle_to() {
    assert_scalar_near(
        Point::new(1.0, 1.0).angle_to(Point::new(1.0, -1.0)).0,
        -K_PI / 2.0,
    );
    assert_scalar_near(
        Point::new(1.0, -1.0).angle_to(Point::new(1.0, 1.0)).0,
        K_PI / 2.0,
    );
    assert_scalar_near(
        Point::new(100.0, -100.0).angle_to(Point::new(0.01, 0.01)).0,
        K_PI / 2.0,
    );
}

/// C++ source: geometry_unittests.cc:PointLerp
#[test]
fn point_and_vector_lerp() {
    assert_point_near(
        Point::new(1.0, 2.0).lerp(Point::new(5.0, 10.0), 0.75),
        Point::new(4.0, 8.0),
    );
    assert_vector3_near(
        Vector3::new(1.0, 2.0, 3.0).lerp(Vector3::new(5.0, 10.0, 15.0), 0.75),
        Vector3::new(4.0, 8.0, 12.0),
    );
    assert_vector4_near(
        Vector4::new(1.0, 2.0, 3.0, 4.0).lerp(Vector4::new(5.0, 10.0, 15.0, 20.0), 0.75),
        Vector4::new(4.0, 8.0, 12.0, 16.0),
    );
}

/// C++ source: geometry_unittests.cc:CanGenerateMipCounts
#[test]
fn can_generate_mip_counts() {
    assert_eq!(Size::new(128.0, 128.0).mip_count(), 7);
    assert_eq!(Size::new(128.0, 256.0).mip_count(), 7);
    assert_eq!(Size::new(128.0, 130.0).mip_count(), 7);
    assert_eq!(Size::new(128.0, 257.0).mip_count(), 7);
    assert_eq!(Size::new(257.0, 128.0).mip_count(), 7);
    assert_eq!(Size::new(128.0, 0.0).mip_count(), 1);
    assert_eq!(Size::new(128.0, -25.0).mip_count(), 1);
    assert_eq!(Size::new(-128.0, 25.0).mip_count(), 1);
    assert_eq!(Size::new(1.0, 1.0).mip_count(), 1);
    assert_eq!(Size::new(0.0, 0.0).mip_count(), 1);
}

/// C++ source: geometry_unittests.cc:MakeColumn
#[test]
fn matrix_make_column() {
    let matrix = Matrix::make_column(
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    );
    assert_eq!(
        matrix.m,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    );
}

/// C++ source: geometry_unittests.cc:MakeRow
#[test]
fn matrix_make_row() {
    let matrix = Matrix::make_row(
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    );
    let expect = Matrix::make_column(
        1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0,
    );
    assert_eq!(matrix, expect);
}

/// C++ source: geometry_unittests.cc:RotationMatrix
#[test]
fn matrix_rotation_z() {
    let rotation = Matrix::make_rotation_z(Radians(K_PI_OVER_4));
    let expect = Matrix::make_column(
        K_1_OVER_SQRT_2,
        K_1_OVER_SQRT_2,
        0.0,
        0.0,
        -K_1_OVER_SQRT_2,
        K_1_OVER_SQRT_2,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    assert_matrix_near(rotation, expect);
}

/// C++ source: geometry_unittests.cc:InvertMultMatrix
#[test]
fn matrix_invert_mult() {
    let rotation = Matrix::make_rotation_z(Radians(K_PI_OVER_4));
    let invert = rotation.invert();
    let expect = Matrix::make_column(
        K_1_OVER_SQRT_2,
        -K_1_OVER_SQRT_2,
        0.0,
        0.0,
        K_1_OVER_SQRT_2,
        K_1_OVER_SQRT_2,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    assert_matrix_near(invert, expect);
    assert_matrix_near(rotation * invert, Matrix::IDENTITY);

    let scale = Matrix::make_scale(2.0, 4.0, 1.0);
    let scale_inv = scale.invert();
    let expect_scale = Matrix::make_column(
        0.5, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    );
    assert_matrix_near(scale_inv, expect_scale);
}

/// C++ source: geometry_unittests.cc:MatrixBasis
#[test]
fn matrix_basis() {
    let matrix = Matrix::make_column(
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    );
    let basis = matrix.basis();
    let expect = Matrix::make_column(
        1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 7.0, 0.0, 9.0, 10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    );
    assert_matrix_near(basis, expect);
}

/// C++ source: geometry_unittests.cc:DeterminantTest
#[test]
fn matrix_determinant_test() {
    let matrix = Matrix::make_column(
        3.0, 4.0, 14.0, 155.0, 2.0, 1.0, 3.0, 4.0, 2.0, 3.0, 2.0, 1.0, 1.0, 2.0, 4.0, 2.0,
    );
    assert_eq!(matrix.determinant(), -1889.0);
}

/// C++ source: geometry_unittests.cc:InvertMatrix
#[test]
fn matrix_invert() {
    let inverted = Matrix::make_column(
        10.0, -9.0, -12.0, 8.0, 7.0, -12.0, 11.0, 22.0, -10.0, 10.0, 3.0, 6.0, -2.0, 22.0, 2.0, 1.0,
    )
    .invert();
    let result = Matrix::make_column(
        438.0 / 85123.0,
        1751.0 / 85123.0,
        -7783.0 / 85123.0,
        4672.0 / 85123.0,
        393.0 / 85123.0,
        -178.0 / 85123.0,
        -570.0 / 85123.0,
        4192.0 / 85123.0,
        -5230.0 / 85123.0,
        2802.0 / 85123.0,
        -3461.0 / 85123.0,
        962.0 / 85123.0,
        2690.0 / 85123.0,
        1814.0 / 85123.0,
        3896.0 / 85123.0,
        319.0 / 85123.0,
    );
    assert_matrix_near(inverted, result);
}

/// C++ source: geometry_unittests.cc:TestDecomposition2
#[test]
fn matrix_decomposition_and_recomposition() {
    let rotated = Matrix::make_rotation_z(Radians(K_PI_OVER_4));
    let scaled = Matrix::make_scale(2.0, 3.0, 1.0);
    let translated = Matrix::make_translation(-200.0, 750.0, 20.0);
    let combined = translated * rotated * scaled;

    let res = combined.decompose().expect("should decompose");
    let quaternion = Quaternion::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), K_PI_OVER_4);
    assert_quaternion_near(res.rotation, quaternion);
    assert_scalar_near(res.translation[0], -200.0);
    assert_scalar_near(res.translation[1], 750.0);
    assert_scalar_near(res.translation[2], 20.0);
    assert_scalar_near(res.scale[0], 2.0);
    assert_scalar_near(res.scale[1], 3.0);
    assert_scalar_near(res.scale[2], 1.0);

    let recomposed = Matrix::from_decomposition(&res);
    assert_matrix_near(combined, recomposed);
}

/// C++ source: geometry_unittests.cc:MatrixVectorMultiplication
#[test]
fn matrix_vector_multiplication() {
    let matrix = Matrix::make_translation(100.0, 100.0, 100.0)
        * Matrix::make_rotation_z(Radians(K_PI_OVER_2))
        * Matrix::make_scale(2.0, 2.0, 2.0);

    assert_vector4_near(
        matrix * Vector4::new(10.0, 20.0, 30.0, 2.0),
        Vector4::new(160.0, 220.0, 260.0, 2.0),
    );
    assert_vector3_near(
        matrix * Vector3::new(10.0, 20.0, 30.0),
        Vector3::new(60.0, 120.0, 160.0),
    );
    assert_point_near(matrix * Point::new(10.0, 20.0), Point::new(60.0, 120.0));

    let persp = Matrix::make_perspective(Radians(K_PI_OVER_2), 1.0, 1.0, 100.0);
    assert_vector3_near(
        persp * Vector3::new(3.0, 3.0, -3.0),
        Vector3::new(-1.0, -1.0, 1.3468),
    );
    assert_point_near(
        (persp * Matrix::make_translation(0.0, 0.0, -3.0)) * Point::new(3.0, 3.0),
        Point::new(-1.0, -1.0),
    );
    assert_point_near(persp * Point::new(3.0, 3.0), Point::new(0.0, 0.0));
}

/// C++ source: geometry_unittests.cc:MatrixMakeRotationFromQuaternion
#[test]
fn matrix_make_rotation_from_quaternion() {
    assert_matrix_near(
        Matrix::make_rotation_quaternion(Quaternion::from_axis_angle(
            Vector3::new(1.0, 0.0, 0.0),
            K_PI_OVER_2,
        )),
        Matrix::make_rotation_x(Radians(K_PI_OVER_2)),
    );
    assert_matrix_near(
        Matrix::make_rotation_quaternion(Quaternion::from_axis_angle(
            Vector3::new(0.0, 1.0, 0.0),
            K_PI_OVER_2,
        )),
        Matrix::make_rotation_y(Radians(K_PI_OVER_2)),
    );
    assert_matrix_near(
        Matrix::make_rotation_quaternion(Quaternion::from_axis_angle(
            Vector3::new(0.0, 0.0, 1.0),
            K_PI_OVER_2,
        )),
        Matrix::make_rotation_z(Radians(K_PI_OVER_2)),
    );
}

/// C++ source: geometry_unittests.cc:MatrixTransformDirection
#[test]
fn matrix_transform_direction() {
    let matrix = Matrix::make_translation(100.0, 100.0, 100.0)
        * Matrix::make_rotation_z(Radians(K_PI_OVER_2))
        * Matrix::make_scale(2.0, 2.0, 2.0);
    assert_vector4_near(
        matrix.transform_direction_4d(Vector4::new(10.0, 20.0, 30.0, 2.0)),
        Vector4::new(-40.0, 20.0, 60.0, 2.0),
    );
    assert_vector3_near(
        matrix.transform_direction_3d(Vector3::new(10.0, 20.0, 30.0)),
        Vector3::new(-40.0, 20.0, 60.0),
    );
    assert_point_near(
        matrix.transform_direction(Point::new(10.0, 20.0)),
        Point::new(-40.0, 20.0),
    );
}

/// C++ source: geometry_unittests.cc:MatrixGetMaxBasisLengthXY
#[test]
fn matrix_get_max_basis_length_xy() {
    let m = Matrix::make_scale(3.0, 1.0, 1.0);
    assert_eq!(m.get_max_basis_length_xy(), 3.0);

    let m2 = m * Matrix::make_skew(0.0, 4.0);
    assert_eq!(m2.get_max_basis_length_xy(), 5.0);

    let m3 = Matrix::make_scale(-3.0, 4.0, 7.0);
    assert_eq!(m3.get_max_basis_length_xy(), 4.0);
}

/// C++ source: geometry_unittests.cc:MatrixMakeOrthographic
#[test]
fn matrix_make_orthographic() {
    let m = Matrix::make_orthographic(Size::new(100.0, 200.0));
    let expect = Matrix::make_column(
        0.02, 0.0, 0.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.5, 1.0,
    );
    assert_matrix_near(m, expect);
}

/// C++ source: geometry_unittests.cc:MatrixMakePerspective
#[test]
fn matrix_make_perspective() {
    let m = Matrix::make_perspective_size(Degrees(60.0).into(), Size::new(100.0, 200.0), 1.0, 10.0);
    let expect = Matrix::make_column(
        3.4641, 0.0, 0.0, 0.0, 0.0, 1.73205, 0.0, 0.0, 0.0, 0.0, 1.11111, 1.0, 0.0, 0.0, -1.11111,
        0.0,
    );
    assert_matrix_near(m, expect);

    let m2 = Matrix::make_perspective(Radians(1.0), 2.0, 10.0, 20.0);
    let expect2 = Matrix::make_column(
        0.915244, 0.0, 0.0, 0.0, 0.0, 1.83049, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, -20.0, 0.0,
    );
    assert_matrix_near(m2, expect2);
}

/// C++ source: geometry_unittests.cc:MatrixLookAt
#[test]
fn matrix_look_at() {
    let m = Matrix::make_look_at(
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 1.0, 0.0),
    );
    let expected = Matrix::make_column(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    );
    assert_matrix_near(m, expected);

    let m2 = Matrix::make_look_at(
        Vector3::ZERO,
        Vector3::new(10.0, -10.0, 0.0),
        Vector3::new(0.0, 0.0, -1.0),
    );
    let expected2 = Matrix::make_column(
        -K_1_OVER_SQRT_2,
        0.0,
        K_1_OVER_SQRT_2,
        0.0,
        -K_1_OVER_SQRT_2,
        0.0,
        -K_1_OVER_SQRT_2,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    assert_matrix_near(m2, expected2);
}

/// C++ source: geometry_unittests.cc:QuaternionLerp
#[test]
fn quaternion_slerp() {
    let q1 = Quaternion::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), 0.0);
    let q2 = Quaternion::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), K_PI_OVER_4);
    let q3 = q1.slerp(q2, 0.5);
    let expected = Quaternion::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), K_PI_OVER_4 / 2.0);
    assert_quaternion_near(q3, expected);
}

/// C++ source: geometry_unittests.cc:QuaternionVectorMultiply
#[test]
fn quaternion_vector_multiply() {
    let q = Quaternion::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), K_PI_OVER_4);
    assert_vector3_near(
        q * Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(-K_1_OVER_SQRT_2, K_1_OVER_SQRT_2, 0.0),
    );

    let q2 = Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 1.0).normalize(), K_PI);
    assert_vector3_near(
        q2 * Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(-1.0, 0.0, 0.0),
    );
}

/// C++ source: matrix_unittests.cc:HasPerspective2D
#[test]
fn matrix_has_perspective_2d() {
    assert!(!Matrix::IDENTITY.has_perspective_2d());
    for index in 0..16 {
        let mut matrix = Matrix::IDENTITY;
        matrix.m[index] = 0.5;
        let expect = matches!(index, 3 | 7 | 15);
        assert_eq!(matrix.has_perspective_2d(), expect, "index={index}");
    }
}

/// C++ source: matrix_unittests.cc:IsInvertibleGetDeterminant
#[test]
fn matrix_is_invertible_get_determinant() {
    assert!(Matrix::IDENTITY.is_invertible());
    assert!(Matrix::make_translation(100.0, 100.0, 0.0).is_invertible());
    assert!(Matrix::make_scale(100.0, 100.0, 1.0).is_invertible());
    assert!(Matrix::make_rotation_x(Degrees(30.0).into()).is_invertible());
    assert!(Matrix::make_rotation_y(Degrees(30.0).into()).is_invertible());
    assert!(Matrix::make_rotation_z(Degrees(30.0).into()).is_invertible());

    assert!(!Matrix::make_scale(0.0, 1.0, 1.0).is_invertible());
    assert!(!Matrix::make_scale(1.0, 0.0, 1.0).is_invertible());
    assert!(!Matrix::make_scale(1.0, 1.0, 0.0).is_invertible());
}

/// C++ source: matrix_unittests.cc:IsTranslationScaleOnly
#[test]
fn matrix_is_translation_scale_only() {
    assert!(Matrix::IDENTITY.is_translation_only());
    assert!(Matrix::IDENTITY.is_translation_scale_only());
    assert!(Matrix::make_translation(100.0, 50.0, 0.0).is_translation_only());
    assert!(Matrix::make_translation(100.0, 50.0, 0.0).is_translation_scale_only());
    assert!(!Matrix::make_scale(2.0, 3.0, 1.0).is_translation_only());
    assert!(Matrix::make_scale(2.0, 3.0, 1.0).is_translation_scale_only());
    assert!(!Matrix::make_rotation_z(Degrees(10.0).into()).is_translation_scale_only());
}

/// C++ source: matrix_unittests.cc:IsAligned
#[test]
fn matrix_is_aligned() {
    assert!(Matrix::IDENTITY.is_aligned(0.0));
    assert!(Matrix::make_translation(1.0, 2.0, 3.0).is_aligned(0.0));
    assert!(Matrix::make_scale(1.0, 1.0, 2.0).is_aligned(0.0));
    assert!(!Matrix::make_rotation_z(Degrees(123.0).into()).is_aligned(0.0));

    for i in -4..=4 {
        let m = Matrix::make_rotation_z(Degrees((i * 90) as Scalar).into());
        assert!(m.is_aligned(K_EH_CLOSE_ENOUGH), "quadrant i={i}");
        assert!(m.is_aligned_2d(K_EH_CLOSE_ENOUGH), "quadrant 2d i={i}");
    }
}

/// C++ source: matrix_unittests.cc:TransformHomogenous
#[test]
fn matrix_transform_homogenous() {
    let matrix = Matrix::make_column(
        2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0, 37.0, 41.0, 43.0, 47.0, 53.0,
    );
    assert_eq!(
        matrix.transform_homogenous(Point::new(1.0, -1.0)),
        Vector3::new(32.0, 33.0, 41.0)
    );
}

/// C++ source: geometry_unittests.cc:ColorPremultiply
#[test]
fn color_premultiply_and_unpremultiply() {
    let a = Color::new(1.0, 0.5, 0.2, 0.5);
    assert_color_near(a.premultiply(), Color::new(0.5, 0.25, 0.1, 0.5));
    assert_color_near(
        Color::new(0.5, 0.25, 0.1, 0.5).unpremultiply(),
        Color::new(1.0, 0.5, 0.2, 0.5),
    );
    assert_color_near(
        Color::new(0.5, 0.25, 0.1, 0.0).unpremultiply(),
        Color::TRANSPARENT,
    );
}

/// C++ source: geometry_unittests.cc:ColorR8G8B8A8
#[test]
fn color_r8g8b8a8() {
    assert_eq!(
        Color::new(1.0, 0.5, 0.2, 0.5).to_r8g8b8a8(),
        [255, 128, 51, 128]
    );
    assert_eq!(Color::TRANSPARENT.to_r8g8b8a8(), [0, 0, 0, 0]);
    assert_eq!(Color::WHITE.to_r8g8b8a8(), [255, 255, 255, 255]);
}

/// C++ source: geometry_unittests.cc:ColorLerp
#[test]
fn color_lerp() {
    let a = Color::new(0.2, 0.4, 1.0, 0.5);
    let b = Color::new(0.4, 1.0, 0.2, 0.3);
    assert_color_near(Color::lerp(a, b, 0.5), Color::new(0.3, 0.7, 0.6, 0.4));
    assert_color_near(Color::lerp(a, b, 0.0), a);
    assert_color_near(Color::lerp(a, b, 1.0), b);
    assert_color_near(Color::lerp(a, b, 0.2), Color::new(0.24, 0.52, 0.84, 0.46));
}

/// C++ source: geometry_unittests.cc:ColorLinearToSRGB
#[test]
fn color_linear_to_srgb() {
    assert_color_near(Color::WHITE.linear_to_srgb(), Color::WHITE);
    assert_color_near(Color::TRANSPARENT.linear_to_srgb(), Color::TRANSPARENT);
    assert_color_near(
        Color::new(0.2, 0.4, 0.6, 0.8).linear_to_srgb(),
        Color::new(0.484529, 0.665185, 0.797738, 0.8),
    );
}

/// C++ source: geometry_unittests.cc:ColorSRGBToLinear
#[test]
fn color_srgb_to_linear() {
    assert_color_near(Color::WHITE.srgb_to_linear(), Color::WHITE);
    assert_color_near(Color::TRANSPARENT.srgb_to_linear(), Color::TRANSPARENT);
    assert_color_near(
        Color::new(0.2, 0.4, 0.6, 0.8).srgb_to_linear(),
        Color::new(0.0331048, 0.132868, 0.318547, 0.8),
    );
}

/// C++ source: geometry_unittests.cc:ColorBlendReturnsExpectedResults
#[test]
fn color_blend_returns_expected_results() {
    let dst = Color::CORNFLOWER_BLUE.with_alpha(0.75);
    let sources = [
        Color::WHITE.with_alpha(0.75),
        Color::LIME_GREEN.with_alpha(0.75),
        Color::BLACK.with_alpha(0.75),
    ];
    let expected: [[Color; 29]; 3] = [
        [
            Color::new(0.0, 0.0, 0.0, 0.0),
            Color::new(1.0, 1.0, 1.0, 0.75),
            Color::new(0.392157, 0.584314, 0.929412, 0.75),
            Color::new(0.878431, 0.916863, 0.985882, 0.9375),
            Color::new(0.513726, 0.667451, 0.943529, 0.9375),
            Color::new(1.0, 1.0, 1.0, 0.5625),
            Color::new(0.392157, 0.584314, 0.929412, 0.5625),
            Color::new(1.0, 1.0, 1.0, 0.1875),
            Color::new(0.392157, 0.584314, 0.929412, 0.1875),
            Color::new(0.848039, 0.896078, 0.982353, 0.75),
            Color::new(0.544118, 0.688235, 0.947059, 0.75),
            Color::new(0.696078, 0.792157, 0.964706, 0.375),
            Color::new(1.0, 1.0, 1.0, 1.0),
            Color::new(0.392157, 0.584314, 0.929412, 0.5625),
            Color::new(0.878431, 0.916863, 0.985882, 0.9375),
            Color::new(0.74902, 0.916863, 0.985882, 0.9375),
            Color::new(0.513726, 0.667451, 0.943529, 0.9375),
            Color::new(0.878431, 0.916863, 0.985882, 0.9375),
            Color::new(0.878431, 0.916863, 0.985882, 0.9375),
            Color::new(0.513725, 0.667451, 0.943529, 0.9375),
            Color::new(0.878431, 0.916863, 0.985882, 0.9375),
            Color::new(0.654166, 0.775505, 0.964318, 0.9375),
            Color::new(0.643137, 0.566275, 0.428235, 0.9375),
            Color::new(0.643137, 0.566275, 0.428235, 0.9375),
            Color::new(0.513726, 0.667451, 0.943529, 0.9375),
            Color::new(0.617208, 0.655639, 0.724659, 0.9375),
            Color::new(0.617208, 0.655639, 0.724659, 0.9375),
            Color::new(0.617208, 0.655639, 0.724659, 0.9375),
            Color::new(0.878431, 0.916863, 0.985882, 0.9375),
        ],
        [
            Color::new(0.0, 0.0, 0.0, 0.0),
            Color::new(0.196078, 0.803922, 0.196078, 0.75),
            Color::new(0.392157, 0.584314, 0.929412, 0.75),
            Color::new(0.235294, 0.76, 0.342745, 0.9375),
            Color::new(0.352941, 0.628235, 0.782745, 0.9375),
            Color::new(0.196078, 0.803922, 0.196078, 0.5625),
            Color::new(0.392157, 0.584314, 0.929412, 0.5625),
            Color::new(0.196078, 0.803922, 0.196078, 0.1875),
            Color::new(0.392157, 0.584314, 0.929412, 0.1875),
            Color::new(0.245098, 0.74902, 0.379412, 0.75),
            Color::new(0.343137, 0.639216, 0.746078, 0.75),
            Color::new(0.294118, 0.694118, 0.562745, 0.375),
            Color::new(0.441176, 1.0, 0.844118, 1.0),
            Color::new(0.0768935, 0.469742, 0.182238, 0.5625),
            Color::new(0.424452, 0.828743, 0.79105, 0.9375),
            Color::new(0.209919, 0.779839, 0.757001, 0.9375),
            Color::new(0.235294, 0.628235, 0.342745, 0.9375),
            Color::new(0.352941, 0.76, 0.782745, 0.9375),
            Color::new(0.41033, 0.877647, 0.825098, 0.9375),
            Color::new(0.117647, 0.567403, 0.609098, 0.9375),
            Color::new(0.209919, 0.779839, 0.443783, 0.9375),
            Color::new(0.266006, 0.693915, 0.758818, 0.9375),
            Color::new(0.235294, 0.409412, 0.665098, 0.9375),
            Color::new(0.378316, 0.546897, 0.681707, 0.9375),
            Color::new(0.163783, 0.559493, 0.334441, 0.9375),
            Color::new(0.266235, 0.748588, 0.373686, 0.9375),
            Color::new(0.339345, 0.629787, 0.811502, 0.9375),
            Color::new(0.241247, 0.765953, 0.348698, 0.9375),
            Color::new(0.346988, 0.622282, 0.776792, 0.9375),
        ],
        [
            Color::new(0.0, 0.0, 0.0, 0.0),
            Color::new(0.0, 0.0, 0.0, 0.75),
            Color::new(0.392157, 0.584314, 0.929412, 0.75),
            Color::new(0.0784314, 0.116863, 0.185882, 0.9375),
            Color::new(0.313726, 0.467451, 0.743529, 0.9375),
            Color::new(0.0, 0.0, 0.0, 0.5625),
            Color::new(0.392157, 0.584314, 0.929412, 0.5625),
            Color::new(0.0, 0.0, 0.0, 0.1875),
            Color::new(0.392157, 0.584314, 0.929412, 0.1875),
            Color::new(0.0980392, 0.146078, 0.232353, 0.75),
            Color::new(0.294118, 0.438235, 0.697059, 0.75),
            Color::new(0.196078, 0.292157, 0.464706, 0.375),
            Color::new(0.294118, 0.438235, 0.697059, 1.0),
            Color::new(0.0, 0.0, 0.0, 0.5625),
            Color::new(0.313726, 0.467451, 0.743529, 0.9375),
            Color::new(0.0784314, 0.218039, 0.701176, 0.9375),
            Color::new(0.0784314, 0.116863, 0.185882, 0.9375),
            Color::new(0.313726, 0.467451, 0.743529, 0.9375),
            Color::new(0.313726, 0.467451, 0.743529, 0.9375),
            Color::new(0.0784314, 0.116863, 0.185882, 0.9375),
            Color::new(0.0784314, 0.116863, 0.185882, 0.9375),
            Color::new(0.170704, 0.321716, 0.704166, 0.9375),
            Color::new(0.313726, 0.467451, 0.743529, 0.9375),
            Color::new(0.313726, 0.467451, 0.743529, 0.9375),
            Color::new(0.0784314, 0.116863, 0.185882, 0.9375),
            Color::new(0.417208, 0.455639, 0.524659, 0.9375),
            Color::new(0.417208, 0.455639, 0.524659, 0.9375),
            Color::new(0.417208, 0.455639, 0.524659, 0.9375),
            Color::new(0.0784314, 0.116863, 0.185882, 0.9375),
        ],
    ];

    for (src_idx, src) in sources.iter().copied().enumerate() {
        for (mode_idx, mode) in BlendMode::ALL.iter().copied().enumerate() {
            let actual = dst.blend(src, mode);
            let exp = expected[src_idx][mode_idx];
            assert!(
                (actual.red - exp.red).abs() <= K_EH_CLOSE_ENOUGH
                    && (actual.green - exp.green).abs() <= K_EH_CLOSE_ENOUGH
                    && (actual.blue - exp.blue).abs() <= K_EH_CLOSE_ENOUGH
                    && (actual.alpha - exp.alpha).abs() <= K_EH_CLOSE_ENOUGH,
                "src={src_idx}, mode={mode:?}: actual={actual:?}, expected={exp:?}"
            );
        }
    }
}

/// C++ source: geometry_unittests.cc:ToIColor
#[test]
fn color_to_icolor_and_hsla() {
    assert_eq!(Color::to_icolor(Color::TRANSPARENT), 0);
    assert_eq!(Color::to_icolor(Color::WHITE), 0xFFFF_FFFF);
    assert_eq!(
        Color::to_icolor(Color::new(0.5, 0.5, 1.0, 1.0)),
        0xFF80_80FF
    );

    let hsla = ColorHSLA::from_color(Color::CORNFLOWER_BLUE);
    let roundtrip = hsla.to_color();
    assert_color_near(roundtrip, Color::CORNFLOWER_BLUE);
}

/// C++ source: rect_unittests.cc:RectEmptyDeclaration
#[test]
fn rect_empty_declaration() {
    let rect = Rect::default();
    assert_eq!(rect.get_left(), 0.0);
    assert_eq!(rect.get_top(), 0.0);
    assert_eq!(rect.get_right(), 0.0);
    assert_eq!(rect.get_bottom(), 0.0);
    assert_eq!(rect.get_x(), 0.0);
    assert_eq!(rect.get_y(), 0.0);
    assert_eq!(rect.get_width(), 0.0);
    assert_eq!(rect.get_height(), 0.0);
    assert!(rect.is_empty());
    assert!(rect.is_finite());
}

/// C++ source: rect_unittests.cc:RectSimpleLTRB
#[test]
fn rect_simple_ltrb_and_xywh() {
    let rect = Rect::from_ltrb(5.125, 10.25, 20.625, 25.375);
    assert_eq!(rect.get_left(), 5.125);
    assert_eq!(rect.get_top(), 10.25);
    assert_eq!(rect.get_right(), 20.625);
    assert_eq!(rect.get_bottom(), 25.375);
    assert_eq!(rect.get_width(), 15.5);
    assert_eq!(rect.get_height(), 15.125);
    assert!(!rect.is_empty());

    let rect2 = Rect::from_xywh(5.125, 10.25, 15.5, 15.125);
    assert_eq!(rect, rect2);
}

/// C++ source: rect_unittests.cc:RectMakeCircleBounds
#[test]
fn rect_make_circle_bounds() {
    let rect = Rect::make_circle_bounds(Point::new(100.0, 200.0), 20.0);
    assert!(!rect.is_empty());
    assert_eq!(rect.get_left(), 80.0);
    assert_eq!(rect.get_right(), 120.0);
    assert_eq!(rect.get_top(), 180.0);
    assert_eq!(rect.get_bottom(), 220.0);
}

/// C++ source: rect_unittests.cc:RectScale
#[test]
fn rect_scale() {
    let rect = Rect::from_ltrb(10.0, 15.0, 100.0, 150.0);
    assert_rect_near(
        rect.scale(2.5),
        Rect::from_xywh(25.0, 37.5, 90.0 * 2.5, 135.0 * 2.5),
    );
    assert_rect_near(
        rect.scale_xy(2.5, 3.5),
        Rect::from_xywh(25.0, 52.5, 90.0 * 2.5, 135.0 * 3.5),
    );
}

/// C++ source: rect_unittests.cc:RectGetNormalizingTransform
#[test]
fn rect_get_normalizing_transform() {
    let r = Rect::from_ltrb(300.0, 500.0, 400.0, 700.0);
    let m = r.get_normalizing_transform();
    assert_point_near(m * Point::new(300.0, 500.0), Point::new(0.0, 0.0));
    assert_point_near(m * Point::new(400.0, 500.0), Point::new(1.0, 0.0));
    assert_point_near(m * Point::new(400.0, 700.0), Point::new(1.0, 1.0));
    assert_point_near(m * Point::new(300.0, 700.0), Point::new(0.0, 1.0));
    assert_point_near(m * Point::new(350.0, 600.0), Point::new(0.5, 0.5));
}

/// C++ source: rect_unittests.cc:MakePointBoundsQuad
#[test]
fn rect_make_point_bounds_quad() {
    let quad: Quad = [
        Point::new(10.0, 10.0),
        Point::new(20.0, 10.0),
        Point::new(10.0, 20.0),
        Point::new(20.0, 20.0),
    ];
    let bounds = Rect::make_point_bounds(&quad).expect("non-empty quad");
    assert_rect_near(bounds, Rect::from_ltrb(10.0, 10.0, 20.0, 20.0));
}

/// C++ source: rect_unittests.cc:RectExpand
#[test]
fn rect_expand_and_union_intersection() {
    let rect = Rect::from_ltrb(100.0, 100.0, 200.0, 200.0);
    assert_eq!(
        rect.expand_all(10.0),
        Rect::from_ltrb(90.0, 90.0, 210.0, 210.0)
    );
    assert_eq!(
        rect.expand(10.0, 20.0, 30.0, 40.0),
        Rect::from_ltrb(90.0, 80.0, 230.0, 240.0)
    );

    let r2 = Rect::from_ltrb(150.0, 150.0, 250.0, 250.0);
    assert_eq!(rect.union(r2), Rect::from_ltrb(100.0, 100.0, 250.0, 250.0));
    assert_eq!(
        rect.intersection(r2),
        Some(Rect::from_ltrb(150.0, 150.0, 200.0, 200.0))
    );
}

#[derive(Default)]
struct SpyPathReceiver {
    moved: Vec<(Point, bool)>,
    lines: Vec<Point>,
    conics: Vec<(Point, Point, Scalar)>,
    cubics: Vec<(Point, Point, Point)>,
    closed_count: usize,
}

impl PathReceiver for SpyPathReceiver {
    fn move_to(&mut self, p2: Point, will_be_closed: bool) {
        self.moved.push((p2, will_be_closed));
    }

    fn line_to(&mut self, p2: Point) {
        self.lines.push(p2);
    }

    fn conic_to(&mut self, cp: Point, p2: Point, weight: Scalar) -> bool {
        self.conics.push((cp, p2, weight));
        true
    }

    fn cubic_to(&mut self, cp1: Point, cp2: Point, p2: Point) {
        self.cubics.push((cp1, cp2, p2));
    }

    fn close(&mut self) {
        self.closed_count += 1;
    }
}

fn check_point_with_offset(rse: &RoundSuperellipse, p: Point, outward_offset: Point) {
    assert!(
        rse.contains(p),
        "expected {p:?} to be inside RoundSuperellipse"
    );
    assert!(
        !rse.contains(p + outward_offset),
        "expected {:?} (p + {outward_offset:?}) to be outside RoundSuperellipse",
        p + outward_offset
    );
}

/// C++ source: round_superellipse_unittests.cc:EmptyDeclaration
#[test]
fn round_superellipse_empty_declaration_and_default() {
    let rse = RoundSuperellipse::new();
    assert!(rse.is_empty());
    assert!(!rse.is_rect());
    assert!(!rse.is_oval());
    assert!(rse.is_finite());
    assert!(rse.get_bounds().is_empty());
    assert_eq!(rse.get_bounds(), Rect::ZERO);
    assert_eq!(rse.get_radii().top_left, Size::ZERO);
    assert_eq!(rse.get_radii().top_right, Size::ZERO);
    assert_eq!(rse.get_radii().bottom_left, Size::ZERO);
    assert_eq!(rse.get_radii().bottom_right, Size::ZERO);

    let rse_def = RoundSuperellipse::default();
    assert_eq!(rse, rse_def);
}

/// C++ source: round_superellipse_unittests.cc:RectConstructor
#[test]
fn round_superellipse_rect_and_inverted_rect_construction() {
    let empty_rect = RoundSuperellipse::make_rect(Rect::from_ltrb(20.0, 20.0, 20.0, 20.0));
    assert!(empty_rect.is_empty());
    assert!(!empty_rect.is_rect());
    assert!(!empty_rect.is_oval());
    assert!(empty_rect.is_finite());

    let rse = RoundSuperellipse::make_rect(Rect::from_ltrb(10.0, 10.0, 20.0, 20.0));
    assert!(!rse.is_empty());
    assert!(rse.is_rect());
    assert!(!rse.is_oval());
    assert!(rse.is_finite());
    assert_eq!(rse.get_bounds(), Rect::from_ltrb(10.0, 10.0, 20.0, 20.0));
    assert_eq!(rse.get_radii(), RoundingRadii::ZERO);

    let inv = RoundSuperellipse::make_rect(Rect::from_ltrb(20.0, 20.0, 10.0, 10.0));
    assert!(!inv.is_empty());
    assert!(inv.is_rect());
    assert_eq!(inv.get_bounds(), Rect::from_ltrb(10.0, 10.0, 20.0, 20.0));
}

/// C++ source: round_superellipse_unittests.cc:OvalConstructor
#[test]
fn round_superellipse_oval_and_inverted_oval_construction() {
    let empty_oval =
        RoundSuperellipse::make_rect_xy(Rect::from_ltrb(20.0, 20.0, 20.0, 20.0), 10.0, 10.0);
    assert!(empty_oval.is_empty());
    assert!(!empty_oval.is_oval());

    let oval = RoundSuperellipse::make_oval(Rect::from_ltrb(10.0, 10.0, 20.0, 20.0));
    assert!(!oval.is_empty());
    assert!(!oval.is_rect());
    assert!(oval.is_oval());
    assert!(oval.is_finite());
    assert_eq!(oval.get_bounds(), Rect::from_ltrb(10.0, 10.0, 20.0, 20.0));
    assert_eq!(oval.get_radii().top_left, Size::new(5.0, 5.0));
    assert_eq!(oval.get_radii().top_right, Size::new(5.0, 5.0));
    assert_eq!(oval.get_radii().bottom_left, Size::new(5.0, 5.0));
    assert_eq!(oval.get_radii().bottom_right, Size::new(5.0, 5.0));

    let inv_oval =
        RoundSuperellipse::make_rect_xy(Rect::from_ltrb(20.0, 20.0, 10.0, 10.0), 10.0, 10.0);
    assert!(inv_oval.is_oval());
    assert_eq!(inv_oval.get_radii().top_left, Size::new(5.0, 5.0));
}

/// C++ source: round_superellipse_unittests.cc:RectRadiiOverflowWidthConstructor
#[test]
fn round_superellipse_rect_radii_and_overflow_scaling() {
    let rse_r = RoundSuperellipse::make_rect_radius(Rect::from_ltrb(10.0, 10.0, 20.0, 20.0), 2.0);
    assert_eq!(rse_r.get_radii().top_left, Size::new(2.0, 2.0));

    let rse_xy = RoundSuperellipse::make_rect_xy(Rect::from_ltrb(10.0, 10.0, 20.0, 20.0), 2.0, 3.0);
    assert_eq!(rse_xy.get_radii().top_left, Size::new(2.0, 3.0));

    let rse_size = RoundSuperellipse::make_rect_size(
        Rect::from_ltrb(10.0, 10.0, 20.0, 20.0),
        Size::new(2.0, 3.0),
    );
    assert_eq!(rse_size, rse_xy);

    let overflow_radii = RoundingRadii::from_radii(
        Size::new(1.0, 2.0),
        Size::new(3.0, 4.0),
        Size::new(5.0, 6.0),
        Size::new(7.0, 8.0),
    );

    let rse_w =
        RoundSuperellipse::make_rect_radii(Rect::from_xywh(10.0, 10.0, 6.0, 30.0), overflow_radii);
    assert_eq!(rse_w.get_bounds(), Rect::from_ltrb(10.0, 10.0, 16.0, 40.0));
    assert_eq!(rse_w.get_radii().top_left, Size::new(0.5, 1.0));
    assert_eq!(rse_w.get_radii().top_right, Size::new(1.5, 2.0));
    assert_eq!(rse_w.get_radii().bottom_left, Size::new(2.5, 3.0));
    assert_eq!(rse_w.get_radii().bottom_right, Size::new(3.5, 4.0));

    let rse_h =
        RoundSuperellipse::make_rect_radii(Rect::from_xywh(10.0, 10.0, 30.0, 6.0), overflow_radii);
    assert_eq!(rse_h.get_bounds(), Rect::from_ltrb(10.0, 10.0, 40.0, 16.0));
    assert_eq!(rse_h.get_radii().top_left, Size::new(0.5, 1.0));
    assert_eq!(rse_h.get_radii().top_right, Size::new(1.5, 2.0));
    assert_eq!(rse_h.get_radii().bottom_left, Size::new(2.5, 3.0));
    assert_eq!(rse_h.get_radii().bottom_right, Size::new(3.5, 4.0));
}

/// C++ source: round_superellipse_unittests.cc:ContractAndRequireRadiiAdjustment
#[test]
fn round_superellipse_shift_expand_and_contract() {
    let radii = RoundingRadii::from_radii(
        Size::new(1.0, 2.0),
        Size::new(3.0, 4.0),
        Size::new(5.0, 6.0),
        Size::new(7.0, 8.0),
    );
    let rse = RoundSuperellipse::make_rect_radii(Rect::from_xywh(10.0, 10.0, 30.0, 30.0), radii);

    let shifted = rse.shift(5.0, 6.0);
    assert_eq!(
        shifted.get_bounds(),
        Rect::from_ltrb(15.0, 16.0, 45.0, 46.0)
    );
    assert_eq!(shifted.get_radii(), radii);

    assert_eq!(
        rse.expand(5.0).get_bounds(),
        Rect::from_ltrb(5.0, 5.0, 45.0, 45.0)
    );
    assert_eq!(
        rse.expand_xy(5.0, 6.0).get_bounds(),
        Rect::from_ltrb(5.0, 4.0, 45.0, 46.0)
    );
    assert_eq!(
        rse.expand_ltrb(5.0, 6.0, 7.0, 8.0).get_bounds(),
        Rect::from_ltrb(5.0, 4.0, 47.0, 48.0)
    );

    assert_eq!(
        rse.expand(-2.0).get_bounds(),
        Rect::from_ltrb(12.0, 12.0, 38.0, 38.0)
    );
    assert_eq!(
        rse.expand_xy(-1.0, -2.0).get_bounds(),
        Rect::from_ltrb(11.0, 12.0, 39.0, 38.0)
    );
    assert_eq!(
        rse.expand_ltrb(-1.0, -1.5, -2.0, -2.5).get_bounds(),
        Rect::from_ltrb(11.0, 11.5, 38.0, 37.5)
    );

    let contracted_adj = rse.expand(-12.0);
    assert_eq!(
        contracted_adj.get_bounds(),
        Rect::from_ltrb(22.0, 22.0, 28.0, 28.0)
    );
    assert_eq!(contracted_adj.get_radii().top_left, Size::new(0.5, 1.0));
    assert_eq!(contracted_adj.get_radii().top_right, Size::new(1.5, 2.0));
    assert_eq!(contracted_adj.get_radii().bottom_left, Size::new(2.5, 3.0));
    assert_eq!(contracted_adj.get_radii().bottom_right, Size::new(3.5, 4.0));
}

/// C++ source: round_superellipse_unittests.cc:NoCornerRoundSuperellipseContains
#[test]
fn round_superellipse_no_corner_and_tiny_corner_contains() {
    let bounds = Rect::from_ltrb(-50.0, -50.0, 50.0, 50.0);
    let no_corners =
        RoundSuperellipse::make_rect_radii(bounds, RoundingRadii::make_radii(Size::new(0.0, 0.0)));

    assert!(no_corners.contains(Point::new(-50.0, -50.0)));
    assert!(no_corners.contains(Point::new(-50.0, 49.99)));
    assert!(no_corners.contains(Point::new(49.99, -50.0)));
    assert!(no_corners.contains(Point::new(49.99, 49.99)));
    assert!(!no_corners.contains(Point::new(-50.01, -50.0)));
    assert!(!no_corners.contains(Point::new(-50.0, -50.01)));
    assert!(!no_corners.contains(Point::new(-50.01, 50.0)));
    assert!(!no_corners.contains(Point::new(-50.0, 50.01)));
    assert!(!no_corners.contains(Point::new(50.01, -50.0)));
    assert!(!no_corners.contains(Point::new(50.0, -50.01)));
    assert!(!no_corners.contains(Point::new(50.01, 50.0)));
    assert!(!no_corners.contains(Point::new(50.0, 50.01)));

    let tiny_corners = RoundSuperellipse::make_rect_radii(
        bounds,
        RoundingRadii::make_radii(Size::new(0.01, 0.01)),
    );
    assert!(!tiny_corners.contains(Point::new(-50.0, -50.0)));
    assert!(!tiny_corners.contains(Point::new(-50.0, 50.0)));
    assert!(!tiny_corners.contains(Point::new(50.0, -50.0)));
    assert!(!tiny_corners.contains(Point::new(50.0, 50.0)));
}

/// C++ source: round_superellipse_unittests.cc:UniformSquareContains
#[test]
fn round_superellipse_uniform_square_contains() {
    let bounds = Rect::from_ltrb(-50.0, -50.0, 50.0, 50.0);
    let rr =
        RoundSuperellipse::make_rect_radii(bounds, RoundingRadii::make_radii(Size::new(5.0, 5.0)));

    let check_mirrors = |p: Point| {
        check_point_with_offset(&rr, p, Point::new(0.02, 0.02));
        check_point_with_offset(&rr, p * Point::new(1.0, -1.0), Point::new(0.02, -0.02));
        check_point_with_offset(&rr, p * Point::new(-1.0, 1.0), Point::new(-0.02, 0.02));
        check_point_with_offset(&rr, p * Point::new(-1.0, -1.0), Point::new(-0.02, -0.02));
    };

    check_mirrors(Point::new(0.0, 49.995));
    check_mirrors(Point::new(44.245, 49.95));
    check_mirrors(Point::new(45.72, 49.87));
    check_mirrors(Point::new(48.53, 48.53));
    check_mirrors(Point::new(49.87, 45.72));
    check_mirrors(Point::new(49.95, 44.245));
    check_mirrors(Point::new(49.995, 0.0));
}

/// C++ source: round_superellipse_unittests.cc:UniformEllipticalContains
#[test]
fn round_superellipse_uniform_elliptical_contains() {
    let bounds = Rect::from_ltrb(-50.0, -50.0, 50.0, 50.0);
    let rr =
        RoundSuperellipse::make_rect_radii(bounds, RoundingRadii::make_radii(Size::new(5.0, 10.0)));

    let check_mirrors = |p: Point| {
        check_point_with_offset(&rr, p, Point::new(0.02, 0.02));
        check_point_with_offset(&rr, p * Point::new(1.0, -1.0), Point::new(0.02, -0.02));
        check_point_with_offset(&rr, p * Point::new(-1.0, 1.0), Point::new(-0.02, 0.02));
        check_point_with_offset(&rr, p * Point::new(-1.0, -1.0), Point::new(-0.02, -0.02));
    };

    check_mirrors(Point::new(0.0, 49.995));
    check_mirrors(Point::new(44.245, 49.911));
    check_mirrors(Point::new(45.72, 49.75));
    check_mirrors(Point::new(48.51, 47.07));
    check_mirrors(Point::new(49.87, 41.44));
    check_mirrors(Point::new(49.95, 38.49));
    check_mirrors(Point::new(49.995, 0.0));
}

/// C++ source: round_superellipse_unittests.cc:UniformRectangularContains
#[test]
fn round_superellipse_uniform_rectangular_contains() {
    let bounds = Rect::from_ltrb(0.0, 0.0, 50.0, 100.0);
    let rr = RoundSuperellipse::make_rect_radii(
        bounds,
        RoundingRadii::make_radii(Size::new(23.0, 30.0)),
    );
    let center = bounds.get_center();

    let check_mirrors = |p: Point| {
        check_point_with_offset(
            &rr,
            (p - center) * Point::new(1.0, 1.0) + center,
            Point::new(0.02, 0.02),
        );
        check_point_with_offset(
            &rr,
            (p - center) * Point::new(1.0, -1.0) + center,
            Point::new(0.02, -0.02),
        );
        check_point_with_offset(
            &rr,
            (p - center) * Point::new(-1.0, 1.0) + center,
            Point::new(-0.02, 0.02),
        );
        check_point_with_offset(
            &rr,
            (p - center) * Point::new(-1.0, -1.0) + center,
            Point::new(-0.02, -0.02),
        );
    };

    check_mirrors(Point::new(24.99, 99.99));
    check_mirrors(Point::new(29.99, 99.64));
    check_mirrors(Point::new(34.99, 98.06));
    check_mirrors(Point::new(39.99, 94.73));
    check_mirrors(Point::new(44.13, 89.99));
    check_mirrors(Point::new(48.46, 79.99));
    check_mirrors(Point::new(49.70, 69.99));
    check_mirrors(Point::new(49.97, 59.99));
    check_mirrors(Point::new(49.99, 49.99));
}

/// C++ source: round_superellipse_unittests.cc:SlimDiagonalContains
#[test]
fn round_superellipse_slim_diagonal_contains() {
    let bounds = Rect::from_ltrb(-50.0, -50.0, 50.0, 50.0);
    let rr = RoundSuperellipse::make_rect_radii(
        bounds,
        RoundingRadii::from_radii(
            Size::new(1.0, 1.0),
            Size::new(99.0, 99.0),
            Size::new(99.0, 99.0),
            Size::new(1.0, 1.0),
        ),
    );

    assert!(rr.contains(Point::new(0.0, 0.0)));
    assert!(!rr.contains(Point::new(-49.999, -49.999)));
    assert!(!rr.contains(Point::new(-49.999, 49.999)));
    assert!(!rr.contains(Point::new(49.999, 49.999)));
    assert!(!rr.contains(Point::new(49.999, -49.999)));

    check_point_with_offset(&rr, Point::new(-49.70, -49.70), Point::new(-0.02, -0.02));
    check_point_with_offset(&rr, Point::new(49.70, 49.70), Point::new(0.02, 0.02));

    let check_diagonal = |p: Point| {
        check_point_with_offset(&rr, p, Point::new(0.02, -0.02));
        check_point_with_offset(&rr, p * Point::new(-1.0, -1.0), Point::new(-0.02, 0.02));
    };

    check_diagonal(Point::new(-40.0, -49.59));
    check_diagonal(Point::new(-20.0, -45.64));
    check_diagonal(Point::new(0.0, -37.01));
    check_diagonal(Point::new(20.0, -21.96));
    check_diagonal(Point::new(21.05, -20.92));
    check_diagonal(Point::new(40.0, 5.68));
}

/// C++ source: round_superellipse_unittests.cc:PointsOutsideOfSharpCorner
#[test]
fn round_superellipse_points_outside_of_sharp_corner() {
    let bounds = Rect::from_ltrb(196.0, 0.0, 294.0, 28.0);
    let rr = RoundSuperellipseParam::make_bounds_radii(
        bounds,
        RoundingRadii::from_radii(
            Size::new(0.0, 0.0),
            Size::new(3.0, 3.0),
            Size::new(0.0, 0.0),
            Size::new(3.0, 3.0),
        ),
    );
    assert!(!rr.contains(Point::new(147.0, 14.0)));
}

/// C++ source: round_superellipse_unittests.cc:PathForRectangularRseWithShapeCornersShouldBeWithinBounds
#[test]
fn round_superellipse_path_for_rectangular_rse_with_sharp_corners_within_bounds() {
    let bounds = Rect::from_ltrb(34.0, 242.0, 766.0, 358.0);
    let rr = RoundSuperellipseParam::make_bounds_radii(
        bounds,
        RoundingRadii::from_radii(
            Size::new(14.0, 14.0),
            Size::new(14.0, 14.0),
            Size::new(0.0, 0.0),
            Size::new(0.0, 0.0),
        ),
    );
    let mut spy = SpyPathReceiver::default();
    rr.dispatch(&mut spy);

    for p2 in spy.lines {
        assert!(
            bounds.contains_inclusive(p2),
            "line_to endpoint {p2:?} outside {bounds:?}"
        );
    }
    for (_cp1, _cp2, p2) in spy.cubics {
        assert!(
            bounds.contains_inclusive(p2),
            "cubic_to endpoint {p2:?} outside {bounds:?}"
        );
    }
    assert_eq!(spy.closed_count, 1);
}

/// C++ source: round_superellipse_unittests.cc:PathForLongRseShouldBeCorrect
#[test]
fn round_superellipse_path_for_long_rse_within_bounds() {
    let bounds = Rect::from_ltrb(0.0, 0.0, 300.0, 100_000.0);
    let rr = RoundSuperellipseParam::make_bounds_radius(bounds, 100.0);
    let mut spy = SpyPathReceiver::default();
    rr.dispatch(&mut spy);

    for p2 in spy.lines {
        assert!(bounds.contains_inclusive(p2));
    }
    for (cp1, cp2, p2) in spy.cubics {
        assert!(bounds.contains_inclusive(cp1));
        assert!(bounds.contains_inclusive(cp2));
        assert!(bounds.contains_inclusive(p2));
    }
    for (cp, p2, _w) in spy.conics {
        assert!(bounds.contains_inclusive(cp));
        assert!(bounds.contains_inclusive(p2));
    }
    assert_eq!(spy.closed_count, 1);
}

/// C++ source: round_superellipse_param.cc:ComputeOctant
#[test]
fn round_superellipse_octant_evaluate_lame_matches_joint_points() {
    let bounds = Rect::from_ltrb(0.0, 0.0, 100.0, 100.0);
    let param = RoundSuperellipseParam::make_bounds_radius(bounds, 20.0);
    let octant = param.top_right.top;
    assert_point_near(octant.evaluate_lame(0.0), Point::new(0.0, octant.se_a));
    assert_point_near(
        octant.evaluate_lame(octant.se_max_theta),
        octant.circle_start,
    );
}
