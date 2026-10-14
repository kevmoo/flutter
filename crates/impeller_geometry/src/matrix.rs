use crate::point::{Point, Quad, Vector2};
use crate::quaternion::Quaternion;
use crate::scalar::{scalar_nearly_equal, scalar_nearly_zero, Radians, Scalar, K_EH_CLOSE_ENOUGH};
use crate::shear::Shear;
use crate::size::Size;
use crate::vector::{Vector3, Vector4};
use bytemuck::{Pod, Zeroable};
use std::ops::{Add, Mul, Sub};

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct MatrixDecomposition {
    pub translation: [Scalar; 3],
    pub scale: [Scalar; 3],
    pub skew: [Scalar; 3],
    pub perspective: [Scalar; 4],
    pub rotation: Quaternion,
}

impl MatrixDecomposition {
    pub const COMPONENT_TRANSLATION: u64 = 1 << 0;
    pub const COMPONENT_SCALE: u64 = 1 << 1;
    pub const COMPONENT_SHEAR: u64 = 1 << 2;
    pub const COMPONENT_PERSPECTIVE: u64 = 1 << 3;
    pub const COMPONENT_ROTATION: u64 = 1 << 4;

    #[inline]
    pub const fn shear(&self) -> Shear {
        Shear::from_array(self.skew)
    }

    #[inline]
    pub fn get_components_mask(&self) -> u64 {
        let mut mask = 0u64;
        if self.rotation != Quaternion::IDENTITY {
            mask |= Self::COMPONENT_ROTATION;
        }
        if self.perspective != [0.0, 0.0, 0.0, 1.0] {
            mask |= Self::COMPONENT_PERSPECTIVE;
        }
        if self.skew != [0.0, 0.0, 0.0] {
            mask |= Self::COMPONENT_SHEAR;
        }
        if self.scale != [1.0, 1.0, 1.0] {
            mask |= Self::COMPONENT_SCALE;
        }
        if self.translation != [0.0, 0.0, 0.0] {
            mask |= Self::COMPONENT_TRANSLATION;
        }
        mask
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Matrix {
    pub m: [Scalar; 16],
}

impl Default for Matrix {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Matrix {
    pub const IDENTITY: Self = Self {
        m: [
            1.0, 0.0, 0.0, 0.0, // col 0
            0.0, 1.0, 0.0, 0.0, // col 1
            0.0, 0.0, 1.0, 0.0, // col 2
            0.0, 0.0, 0.0, 1.0, // col 3
        ],
    };

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub const fn make_column(
        m0: Scalar,
        m1: Scalar,
        m2: Scalar,
        m3: Scalar,
        m4: Scalar,
        m5: Scalar,
        m6: Scalar,
        m7: Scalar,
        m8: Scalar,
        m9: Scalar,
        m10: Scalar,
        m11: Scalar,
        m12: Scalar,
        m13: Scalar,
        m14: Scalar,
        m15: Scalar,
    ) -> Self {
        Self {
            m: [
                m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15,
            ],
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub const fn make_row(
        m0: Scalar,
        m1: Scalar,
        m2: Scalar,
        m3: Scalar,
        m4: Scalar,
        m5: Scalar,
        m6: Scalar,
        m7: Scalar,
        m8: Scalar,
        m9: Scalar,
        m10: Scalar,
        m11: Scalar,
        m12: Scalar,
        m13: Scalar,
        m14: Scalar,
        m15: Scalar,
    ) -> Self {
        Self {
            m: [
                m0, m4, m8, m12, m1, m5, m9, m13, m2, m6, m10, m14, m3, m7, m11, m15,
            ],
        }
    }

    #[inline]
    pub const fn e(&self, col: usize, row: usize) -> Scalar {
        self.m[col * 4 + row]
    }

    #[inline]
    pub fn set_e(&mut self, col: usize, row: usize, val: Scalar) {
        self.m[col * 4 + row] = val;
    }

    #[inline]
    pub const fn make_translation(tx: Scalar, ty: Scalar, tz: Scalar) -> Self {
        Self {
            m: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                tx, ty, tz, 1.0,
            ],
        }
    }

    #[inline]
    pub const fn make_translation_vector(t: Vector3) -> Self {
        Self::make_translation(t.x, t.y, t.z)
    }

    #[inline]
    pub const fn make_scale(sx: Scalar, sy: Scalar, sz: Scalar) -> Self {
        Self {
            m: [
                sx, 0.0, 0.0, 0.0, //
                0.0, sy, 0.0, 0.0, //
                0.0, 0.0, sz, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    #[inline]
    pub const fn make_scale_vector(s: Vector3) -> Self {
        Self::make_scale(s.x, s.y, s.z)
    }

    #[inline]
    pub const fn make_scale_2d(s: Vector2) -> Self {
        Self::make_scale(s.x, s.y, 1.0)
    }

    #[inline]
    pub const fn make_translate_scale(s: Vector3, t: Vector3) -> Self {
        Self {
            m: [
                s.x, 0.0, 0.0, 0.0, //
                0.0, s.y, 0.0, 0.0, //
                0.0, 0.0, s.z, 0.0, //
                t.x, t.y, t.z, 1.0,
            ],
        }
    }

    #[inline]
    pub const fn make_skew(sx: Scalar, sy: Scalar) -> Self {
        Self {
            m: [
                1.0, sy, 0.0, 0.0, //
                sx, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    #[inline]
    pub fn cos_sin(radians: Radians) -> Vector2 {
        let sin = radians.0.sin();
        if sin.abs() == 1.0 {
            Vector2::new(0.0, sin)
        } else {
            let cos = radians.0.cos();
            if cos.abs() == 1.0 {
                Vector2::new(cos, 0.0)
            } else {
                Vector2::new(cos, sin)
            }
        }
    }

    #[inline]
    pub fn make_rotation_quaternion(q: Quaternion) -> Self {
        Self::make_column(
            1.0 - 2.0 * q.y * q.y - 2.0 * q.z * q.z,
            2.0 * q.x * q.y + 2.0 * q.z * q.w,
            2.0 * q.x * q.z - 2.0 * q.y * q.w,
            0.0,
            2.0 * q.x * q.y - 2.0 * q.z * q.w,
            1.0 - 2.0 * q.x * q.x - 2.0 * q.z * q.z,
            2.0 * q.y * q.z + 2.0 * q.x * q.w,
            0.0,
            2.0 * q.x * q.z + 2.0 * q.y * q.w,
            2.0 * q.y * q.z - 2.0 * q.x * q.w,
            1.0 - 2.0 * q.x * q.x - 2.0 * q.y * q.y,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    }

    #[inline]
    pub fn make_rotation(radians: Radians, r: Vector4) -> Self {
        let v = r.normalize();
        let cs = Self::cos_sin(radians);
        let cosine = cs.x;
        let cosp = 1.0 - cosine;
        let sine = cs.y;

        Self::make_column(
            cosine + cosp * v.x * v.x,
            cosp * v.x * v.y + v.z * sine,
            cosp * v.x * v.z - v.y * sine,
            0.0,
            cosp * v.x * v.y - v.z * sine,
            cosine + cosp * v.y * v.y,
            cosp * v.y * v.z + v.x * sine,
            0.0,
            cosp * v.x * v.z + v.y * sine,
            cosp * v.y * v.z - v.x * sine,
            cosine + cosp * v.z * v.z,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    }

    #[inline]
    pub fn make_rotation_x(r: Radians) -> Self {
        let cs = Self::cos_sin(r);
        let cosine = cs.x;
        let sine = cs.y;
        Self::make_column(
            1.0, 0.0, 0.0, 0.0, //
            0.0, cosine, sine, 0.0, //
            0.0, -sine, cosine, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        )
    }

    #[inline]
    pub fn make_rotation_y(r: Radians) -> Self {
        let cs = Self::cos_sin(r);
        let cosine = cs.x;
        let sine = cs.y;
        Self::make_column(
            cosine, 0.0, -sine, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            sine, 0.0, cosine, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        )
    }

    #[inline]
    pub fn make_rotation_z(radians: Radians) -> Self {
        let cs = Self::cos_sin(radians);
        let cosine = cs.x;
        let sine = cs.y;
        Self::make_column(
            cosine, sine, 0.0, 0.0, //
            -sine, cosine, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        )
    }

    #[inline]
    pub fn make_orthographic(size: Size) -> Self {
        let scale = Self::make_scale(2.0 / size.width, -2.0 / size.height, 0.0);
        let translate = Self::make_translation(-1.0, 1.0, 0.5);
        translate * scale
    }

    #[inline]
    pub fn make_perspective(
        fov_y: Radians,
        aspect_ratio: Scalar,
        z_near: Scalar,
        z_far: Scalar,
    ) -> Self {
        let height = (fov_y.0 * 0.5).tan();
        let width = height * aspect_ratio;
        Self::make_column(
            1.0 / width,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / height,
            0.0,
            0.0,
            0.0,
            0.0,
            z_far / (z_far - z_near),
            1.0,
            0.0,
            0.0,
            -(z_far * z_near) / (z_far - z_near),
            0.0,
        )
    }

    #[inline]
    pub fn make_perspective_size(
        fov_y: Radians,
        size: Size,
        z_near: Scalar,
        z_far: Scalar,
    ) -> Self {
        Self::make_perspective(fov_y, size.width / size.height, z_near, z_far)
    }

    #[inline]
    pub fn make_look_at(position: Vector3, target: Vector3, up: Vector3) -> Self {
        let forward = (target - position).normalize();
        let right = up.cross(forward);
        let up = forward.cross(right);

        Self::make_column(
            right.x,
            up.x,
            forward.x,
            0.0,
            right.y,
            up.y,
            forward.y,
            0.0,
            right.z,
            up.z,
            forward.z,
            0.0,
            -right.dot(position),
            -up.dot(position),
            -forward.dot(position),
            1.0,
        )
    }

    pub fn from_decomposition(d: &MatrixDecomposition) -> Self {
        let mut mat = Self::IDENTITY;

        // Apply perspective.
        for i in 0..4 {
            mat.set_e(i, 3, d.perspective[i]);
        }

        // Apply translation.
        for i in 0..3 {
            for j in 0..3 {
                let cur = mat.e(3, i);
                mat.set_e(3, i, cur + d.translation[j] * mat.e(j, i));
            }
        }

        // Apply rotation.
        let mut rotation = Self::IDENTITY;
        let x = -d.rotation.x;
        let y = -d.rotation.y;
        let z = -d.rotation.z;
        let w = d.rotation.w;

        rotation.set_e(0, 0, 1.0 - 2.0 * (y * y + z * z));
        rotation.set_e(0, 1, 2.0 * (x * y - z * w));
        rotation.set_e(0, 2, 2.0 * (x * z + y * w));
        rotation.set_e(1, 0, 2.0 * (x * y + z * w));
        rotation.set_e(1, 1, 1.0 - 2.0 * (x * x + z * z));
        rotation.set_e(1, 2, 2.0 * (y * z - x * w));
        rotation.set_e(2, 0, 2.0 * (x * z - y * w));
        rotation.set_e(2, 1, 2.0 * (y * z + x * w));
        rotation.set_e(2, 2, 1.0 - 2.0 * (x * x + y * y));

        mat = mat * rotation;

        // Apply shear (stored in d.skew as [xy, xz, yz]).
        let mut shear = Self::IDENTITY;
        if d.skew[2] != 0.0 {
            shear.set_e(2, 1, d.skew[2]);
            mat = mat * shear;
        }
        if d.skew[1] != 0.0 {
            shear.set_e(2, 1, 0.0);
            shear.set_e(2, 0, d.skew[1]);
            mat = mat * shear;
        }
        if d.skew[0] != 0.0 {
            shear.set_e(2, 0, 0.0);
            shear.set_e(1, 0, d.skew[0]);
            mat = mat * shear;
        }

        // Apply scale.
        for i in 0..3 {
            for j in 0..3 {
                let cur = mat.e(i, j);
                mat.set_e(i, j, cur * d.scale[i]);
            }
        }

        mat
    }

    #[inline]
    pub const fn basis(&self) -> Self {
        Self::make_column(
            self.m[0], self.m[1], self.m[2], 0.0, //
            self.m[4], self.m[5], self.m[6], 0.0, //
            self.m[8], self.m[9], self.m[10], 0.0, //
            0.0, 0.0, 0.0, 1.0,
        )
    }

    #[inline]
    pub const fn to_3x3(&self) -> Self {
        Self::make_column(
            self.m[0], self.m[1], 0.0, self.m[3], //
            self.m[4], self.m[5], 0.0, self.m[7], //
            0.0, 0.0, 1.0, 0.0, //
            self.m[12], self.m[13], 0.0, self.m[15],
        )
    }

    #[inline]
    pub fn translate(&self, t: Vector3) -> Self {
        Self::make_column(
            self.m[0],
            self.m[1],
            self.m[2],
            self.m[3],
            self.m[4],
            self.m[5],
            self.m[6],
            self.m[7],
            self.m[8],
            self.m[9],
            self.m[10],
            self.m[11],
            self.m[0] * t.x + self.m[4] * t.y + self.m[8] * t.z + self.m[12],
            self.m[1] * t.x + self.m[5] * t.y + self.m[9] * t.z + self.m[13],
            self.m[2] * t.x + self.m[6] * t.y + self.m[10] * t.z + self.m[14],
            self.m[3] * t.x + self.m[7] * t.y + self.m[11] * t.z + self.m[15],
        )
    }

    #[inline]
    pub fn scale(&self, s: Vector3) -> Self {
        Self::make_column(
            self.m[0] * s.x,
            self.m[1] * s.x,
            self.m[2] * s.x,
            self.m[3] * s.x,
            self.m[4] * s.y,
            self.m[5] * s.y,
            self.m[6] * s.y,
            self.m[7] * s.y,
            self.m[8] * s.z,
            self.m[9] * s.z,
            self.m[10] * s.z,
            self.m[11] * s.z,
            self.m[12],
            self.m[13],
            self.m[14],
            self.m[15],
        )
    }

    #[inline]
    pub const fn transpose(&self) -> Self {
        Self::make_column(
            self.m[0], self.m[4], self.m[8], self.m[12], //
            self.m[1], self.m[5], self.m[9], self.m[13], //
            self.m[2], self.m[6], self.m[10], self.m[14], //
            self.m[3], self.m[7], self.m[11], self.m[15],
        )
    }

    pub fn determinant(&self) -> Scalar {
        let a00 = self.e(0, 0);
        let a01 = self.e(0, 1);
        let a02 = self.e(0, 2);
        let a03 = self.e(0, 3);
        let a10 = self.e(1, 0);
        let a11 = self.e(1, 1);
        let a12 = self.e(1, 2);
        let a13 = self.e(1, 3);
        let a20 = self.e(2, 0);
        let a21 = self.e(2, 1);
        let a22 = self.e(2, 2);
        let a23 = self.e(2, 3);
        let a30 = self.e(3, 0);
        let a31 = self.e(3, 1);
        let a32 = self.e(3, 2);
        let a33 = self.e(3, 3);

        let b00 = a00 * a11 - a01 * a10;
        let b01 = a00 * a12 - a02 * a10;
        let b02 = a00 * a13 - a03 * a10;
        let b03 = a01 * a12 - a02 * a11;
        let b04 = a01 * a13 - a03 * a11;
        let b05 = a02 * a13 - a03 * a12;
        let b06 = a20 * a31 - a21 * a30;
        let b07 = a20 * a32 - a22 * a30;
        let b08 = a20 * a33 - a23 * a30;
        let b09 = a21 * a32 - a22 * a31;
        let b10 = a21 * a33 - a23 * a31;
        let b11 = a22 * a33 - a23 * a32;

        b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
    }

    #[inline]
    pub fn get_determinant(&self) -> Scalar {
        self.determinant()
    }

    #[inline]
    pub fn is_invertible(&self) -> bool {
        self.determinant() != 0.0
    }

    pub fn invert(&self) -> Self {
        let m = &self.m;
        let tmp = [
            m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
                + m[9] * m[7] * m[14]
                + m[13] * m[6] * m[11]
                - m[13] * m[7] * m[10],
            -m[1] * m[10] * m[15]
                + m[1] * m[11] * m[14]
                + m[9] * m[2] * m[15]
                - m[9] * m[3] * m[14]
                - m[13] * m[2] * m[11]
                + m[13] * m[3] * m[10],
            m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
                + m[5] * m[3] * m[14]
                + m[13] * m[2] * m[7]
                - m[13] * m[3] * m[6],
            -m[1] * m[6] * m[11]
                + m[1] * m[7] * m[10]
                + m[5] * m[2] * m[11]
                - m[5] * m[3] * m[10]
                - m[9] * m[2] * m[7]
                + m[9] * m[3] * m[6],
            -m[4] * m[10] * m[15]
                + m[4] * m[11] * m[14]
                + m[8] * m[6] * m[15]
                - m[8] * m[7] * m[14]
                - m[12] * m[6] * m[11]
                + m[12] * m[7] * m[10],
            m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
                + m[8] * m[3] * m[14]
                + m[12] * m[2] * m[11]
                - m[12] * m[3] * m[10],
            -m[0] * m[6] * m[15]
                + m[0] * m[7] * m[14]
                + m[4] * m[2] * m[15]
                - m[4] * m[3] * m[14]
                - m[12] * m[2] * m[7]
                + m[12] * m[3] * m[6],
            m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
                + m[4] * m[3] * m[10]
                + m[8] * m[2] * m[7]
                - m[8] * m[3] * m[6],
            m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
                + m[8] * m[7] * m[13]
                + m[12] * m[5] * m[11]
                - m[12] * m[7] * m[9],
            -m[0] * m[9] * m[15]
                + m[0] * m[11] * m[13]
                + m[8] * m[1] * m[15]
                - m[8] * m[3] * m[13]
                - m[12] * m[1] * m[11]
                + m[12] * m[3] * m[9],
            m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
                + m[4] * m[3] * m[13]
                + m[12] * m[1] * m[7]
                - m[12] * m[3] * m[5],
            -m[0] * m[5] * m[11]
                + m[0] * m[7] * m[9]
                + m[4] * m[1] * m[11]
                - m[4] * m[3] * m[9]
                - m[8] * m[1] * m[7]
                + m[8] * m[3] * m[5],
            -m[4] * m[9] * m[14]
                + m[4] * m[10] * m[13]
                + m[8] * m[5] * m[14]
                - m[8] * m[6] * m[13]
                - m[12] * m[5] * m[10]
                + m[12] * m[6] * m[9],
            m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
                + m[8] * m[2] * m[13]
                + m[12] * m[1] * m[10]
                - m[12] * m[2] * m[9],
            -m[0] * m[5] * m[14]
                + m[0] * m[6] * m[13]
                + m[4] * m[1] * m[14]
                - m[4] * m[2] * m[13]
                - m[12] * m[1] * m[6]
                + m[12] * m[2] * m[5],
            m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
                + m[4] * m[2] * m[9]
                + m[8] * m[1] * m[6]
                - m[8] * m[2] * m[5],
        ];

        let mut det =
            m[0] * tmp[0] + m[1] * tmp[4] + m[2] * tmp[8] + m[3] * tmp[12];
        if det == 0.0 {
            return Self::IDENTITY;
        }
        det = 1.0 / det;

        let mut out = [0.0; 16];
        for i in 0..16 {
            out[i] = tmp[i] * det;
        }
        Self { m: out }
    }

    #[inline]
    pub fn get_max_basis_length_xy(&self) -> Scalar {
        if self.e(0, 1) == 0.0 && self.e(1, 0) == 0.0 {
            return self.e(0, 0).abs().max(self.e(1, 1).abs());
        }
        let b0_sq = self.e(0, 0) * self.e(0, 0) + self.e(0, 1) * self.e(0, 1);
        let b1_sq = self.e(1, 0) * self.e(1, 0) + self.e(1, 1) * self.e(1, 1);
        b0_sq.max(b1_sq).sqrt()
    }

    pub fn get_scales_2d(&self) -> Option<(Scalar, Scalar)> {
        if self.has_perspective_2d() {
            return None;
        }

        let a = self.m[0] as f64;
        let b = self.m[1] as f64;
        let c = self.m[4] as f64;
        let d = self.m[5] as f64;

        if b == 0.0 && c == 0.0 {
            return Some((a.abs() as Scalar, d.abs() as Scalar));
        }

        if a == 0.0 && d == 0.0 {
            return Some((b.abs() as Scalar, c.abs() as Scalar));
        }

        let a2 = a * a + b * b;
        let b2 = a * c + b * d;
        let c2 = b2;
        let d2 = c * c + d * d;

        let minus_b = a2 + d2;
        let cap_c = a2 * d2 - b2 * c2;
        let b_squared_minus_4ac = minus_b * minus_b - 4.0 * cap_c;

        let quadratic_sqrt = if b_squared_minus_4ac <= 0.0 {
            debug_assert!(b_squared_minus_4ac + (K_EH_CLOSE_ENOUGH as f64) >= 0.0);
            0.0
        } else {
            b_squared_minus_4ac.sqrt()
        };

        let s1 = ((minus_b - quadratic_sqrt) / 2.0).max(0.0).sqrt() as Scalar;
        let s2 = ((minus_b + quadratic_sqrt) / 2.0).max(0.0).sqrt() as Scalar;
        Some((s1, s2))
    }

    #[inline]
    pub fn get_min_scale_2d(&self) -> Option<Scalar> {
        self.get_scales_2d().map(|(a, b)| a.min(b))
    }

    #[inline]
    pub fn get_max_scale_2d(&self) -> Option<Scalar> {
        self.get_scales_2d().map(|(a, b)| a.max(b))
    }

    #[inline]
    pub const fn get_basis_x(&self) -> Vector3 {
        Vector3::new(self.m[0], self.m[1], self.m[2])
    }

    #[inline]
    pub const fn get_basis_y(&self) -> Vector3 {
        Vector3::new(self.m[4], self.m[5], self.m[6])
    }

    #[inline]
    pub const fn get_basis_z(&self) -> Vector3 {
        Vector3::new(self.m[8], self.m[9], self.m[10])
    }

    #[inline]
    pub fn get_scale(&self) -> Vector3 {
        Vector3::new(
            self.get_basis_x().get_length(),
            self.get_basis_y().get_length(),
            self.get_basis_z().get_length(),
        )
    }

    #[inline]
    pub const fn get_basis_x_2d(&self) -> Vector2 {
        Vector2::new(self.m[0], self.m[1])
    }

    #[inline]
    pub const fn get_basis_y_2d(&self) -> Vector2 {
        Vector2::new(self.m[4], self.m[5])
    }

    #[inline]
    pub fn get_basis_scale_xy(&self) -> Vector2 {
        Vector2::new(
            self.get_basis_x_2d().get_length(),
            self.get_basis_y_2d().get_length(),
        )
    }

    #[inline]
    pub fn get_transformed_pixel_size(&self) -> Vector2 {
        if self.m[1] == 0.0 && self.m[4] == 0.0 {
            if self.m[0] == 0.0 || self.m[5] == 0.0 {
                return Vector2::ZERO;
            }
            return Vector2::new(1.0 / self.m[0].abs(), 1.0 / self.m[5].abs());
        }
        let det = self.m[0] * self.m[5] - self.m[1] * self.m[4];
        if scalar_nearly_zero(det, K_EH_CLOSE_ENOUGH) {
            return Vector2::ZERO;
        }
        let abs_det = det.abs();
        Vector2::new(
            self.get_basis_y_2d().get_length() / abs_det,
            self.get_basis_x_2d().get_length() / abs_det,
        )
    }

    #[inline]
    pub fn get_direction_scale(&self, direction: Vector3) -> Scalar {
        let len = (self.basis().invert() * direction.normalize()).get_length();
        (1.0 / len) * direction.get_length()
    }

    #[inline]
    pub fn is_finite(&self) -> bool {
        self.m.iter().all(|v| v.is_finite())
    }

    #[inline]
    pub fn is_affine(&self) -> bool {
        self.m[2] == 0.0
            && self.m[3] == 0.0
            && self.m[6] == 0.0
            && self.m[7] == 0.0
            && self.m[8] == 0.0
            && self.m[9] == 0.0
            && self.m[10] == 1.0
            && self.m[11] == 0.0
            && self.m[14] == 0.0
            && self.m[15] == 1.0
    }

    #[inline]
    pub fn has_perspective_2d(&self) -> bool {
        self.m[3] != 0.0 || self.m[7] != 0.0 || self.m[15] != 1.0
    }

    #[inline]
    pub fn has_perspective(&self) -> bool {
        self.m[3] != 0.0 || self.m[7] != 0.0 || self.m[11] != 0.0 || self.m[15] != 1.0
    }

    #[inline]
    pub fn has_translation(&self) -> bool {
        self.m[12] != 0.0 || self.m[13] != 0.0
    }

    #[inline]
    pub fn is_aligned_2d(&self, tolerance: Scalar) -> bool {
        if self.has_perspective_2d() {
            return false;
        }
        if scalar_nearly_zero(self.m[1], tolerance) && scalar_nearly_zero(self.m[4], tolerance) {
            return true;
        }
        if scalar_nearly_zero(self.m[0], tolerance) && scalar_nearly_zero(self.m[5], tolerance) {
            return true;
        }
        false
    }

    pub fn is_aligned(&self, tolerance: Scalar) -> bool {
        if self.has_perspective() {
            return false;
        }
        let v = [
            i32::from(!scalar_nearly_zero(self.m[0], tolerance)),
            i32::from(!scalar_nearly_zero(self.m[1], tolerance)),
            i32::from(!scalar_nearly_zero(self.m[2], tolerance)),
            i32::from(!scalar_nearly_zero(self.m[4], tolerance)),
            i32::from(!scalar_nearly_zero(self.m[5], tolerance)),
            i32::from(!scalar_nearly_zero(self.m[6], tolerance)),
            i32::from(!scalar_nearly_zero(self.m[8], tolerance)),
            i32::from(!scalar_nearly_zero(self.m[9], tolerance)),
            i32::from(!scalar_nearly_zero(self.m[10], tolerance)),
        ];
        if v[0] + v[1] + v[2] != 1 || v[3] + v[4] + v[5] != 1 || v[6] + v[7] + v[8] != 1 {
            return false;
        }
        if v[0] + v[3] + v[6] != 1 || v[1] + v[4] + v[7] != 1 || v[2] + v[5] + v[8] != 1 {
            return false;
        }
        true
    }

    #[inline]
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    #[inline]
    pub fn is_translation_only(&self) -> bool {
        self.m[0] == 1.0
            && self.m[1] == 0.0
            && self.m[2] == 0.0
            && self.m[3] == 0.0
            && self.m[4] == 0.0
            && self.m[5] == 1.0
            && self.m[6] == 0.0
            && self.m[7] == 0.0
            && self.m[8] == 0.0
            && self.m[9] == 0.0
            && self.m[10] == 1.0
            && self.m[11] == 0.0
            && self.m[15] == 1.0
    }

    #[inline]
    pub fn is_translation_scale_only(&self) -> bool {
        self.m[0] != 0.0
            && self.m[1] == 0.0
            && self.m[2] == 0.0
            && self.m[3] == 0.0
            && self.m[4] == 0.0
            && self.m[5] != 0.0
            && self.m[6] == 0.0
            && self.m[7] == 0.0
            && self.m[8] == 0.0
            && self.m[9] == 0.0
            && self.m[10] != 0.0
            && self.m[11] == 0.0
            && self.m[15] == 1.0
    }

    pub fn decompose(&self) -> Option<MatrixDecomposition> {
        let mut mat = *self;
        let m33 = mat.e(3, 3);
        if m33 == 0.0 {
            return None;
        }

        for i in 0..4 {
            for j in 0..4 {
                let v = mat.e(i, j) / m33;
                mat.set_e(i, j, v);
            }
        }

        let mut perspective_matrix = mat;
        for i in 0..3 {
            perspective_matrix.set_e(i, 3, 0.0);
        }
        perspective_matrix.set_e(3, 3, 1.0);

        if !perspective_matrix.is_invertible() {
            return None;
        }

        let mut result = MatrixDecomposition {
            translation: [0.0; 3],
            scale: [1.0; 3],
            skew: [0.0; 3],
            perspective: [0.0, 0.0, 0.0, 1.0],
            rotation: Quaternion::IDENTITY,
        };

        // 1. Isolate perspective.
        if mat.e(0, 3) != 0.0 || mat.e(1, 3) != 0.0 || mat.e(2, 3) != 0.0 {
            let rhs = Vector4::new(mat.e(0, 3), mat.e(1, 3), mat.e(2, 3), mat.e(3, 3));
            let p = perspective_matrix.invert().transpose() * rhs;
            result.perspective = p.to_array();
            mat.set_e(0, 3, 0.0);
            mat.set_e(1, 3, 0.0);
            mat.set_e(2, 3, 0.0);
            mat.set_e(3, 3, 1.0);
        }

        // 2. Translation.
        result.translation = [mat.e(3, 0), mat.e(3, 1), mat.e(3, 2)];
        mat.set_e(3, 0, 0.0);
        mat.set_e(3, 1, 0.0);
        mat.set_e(3, 2, 0.0);

        // 3. Scale and shear.
        let mut row = [
            Vector3::new(mat.e(0, 0), mat.e(0, 1), mat.e(0, 2)),
            Vector3::new(mat.e(1, 0), mat.e(1, 1), mat.e(1, 2)),
            Vector3::new(mat.e(2, 0), mat.e(2, 1), mat.e(2, 2)),
        ];

        let mut scale_x = row[0].get_length();
        row[0] = row[0].normalize();

        let mut shear_xy = row[0].dot(row[1]);
        row[1] = Vector3::combine(row[1], 1.0, row[0], -shear_xy);

        let mut scale_y = row[1].get_length();
        row[1] = row[1].normalize();
        shear_xy /= scale_y;

        let mut shear_xz = row[0].dot(row[2]);
        row[2] = Vector3::combine(row[2], 1.0, row[0], -shear_xz);
        let mut shear_yz = row[1].dot(row[2]);
        row[2] = Vector3::combine(row[2], 1.0, row[1], -shear_yz);

        let mut scale_z = row[2].get_length();
        row[2] = row[2].normalize();

        shear_xz /= scale_z;
        shear_yz /= scale_z;

        // Check for coordinate system flip.
        if row[0].dot(row[1].cross(row[2])) < 0.0 {
            scale_x = -scale_x;
            scale_y = -scale_y;
            scale_z = -scale_z;
            for r in &mut row {
                *r = -*r;
            }
        }

        result.scale = [scale_x, scale_y, scale_z];
        result.skew = [shear_xy, shear_xz, shear_yz];

        // 4. Rotation.
        let mut qx = 0.5 * (1.0 + row[0].x - row[1].y - row[2].z).max(0.0).sqrt();
        let mut qy = 0.5 * (1.0 - row[0].x + row[1].y - row[2].z).max(0.0).sqrt();
        let mut qz = 0.5 * (1.0 - row[0].x - row[1].y + row[2].z).max(0.0).sqrt();
        let qw = 0.5 * (1.0 + row[0].x + row[1].y + row[2].z).max(0.0).sqrt();

        if row[2].y > row[1].z {
            qx = -qx;
        }
        if row[0].z > row[2].x {
            qy = -qy;
        }
        if row[1].x > row[0].y {
            qz = -qz;
        }

        result.rotation = Quaternion::new(qx, qy, qz, qw);
        Some(result)
    }

    #[inline]
    pub fn equals(&self, other: &Self, epsilon: Scalar) -> bool {
        self.m
            .iter()
            .zip(other.m.iter())
            .all(|(a, b)| scalar_nearly_equal(*a, *b, epsilon))
    }

    #[inline]
    pub fn transform_point(&self, p: Point) -> Point {
        let mut w = p.x * self.m[3] + p.y * self.m[7] + self.m[15];
        let result = Point::new(
            p.x * self.m[0] + p.y * self.m[4] + self.m[12],
            p.x * self.m[1] + p.y * self.m[5] + self.m[13],
        );
        if w != 0.0 {
            w = 1.0 / w;
        }
        result * w
    }

    #[inline]
    pub fn transform_vector3(&self, v: Vector3) -> Vector3 {
        let mut w = v.x * self.m[3] + v.y * self.m[7] + v.z * self.m[11] + self.m[15];
        let result = Vector3::new(
            v.x * self.m[0] + v.y * self.m[4] + v.z * self.m[8] + self.m[12],
            v.x * self.m[1] + v.y * self.m[5] + v.z * self.m[9] + self.m[13],
            v.x * self.m[2] + v.y * self.m[6] + v.z * self.m[10] + self.m[14],
        );
        if w != 0.0 {
            w = 1.0 / w;
        }
        result * w
    }

    #[inline]
    pub const fn transform_vector4(&self, v: Vector4) -> Vector4 {
        Vector4::new(
            v.x * self.m[0] + v.y * self.m[4] + v.z * self.m[8] + v.w * self.m[12],
            v.x * self.m[1] + v.y * self.m[5] + v.z * self.m[9] + v.w * self.m[13],
            v.x * self.m[2] + v.y * self.m[6] + v.z * self.m[10] + v.w * self.m[14],
            v.x * self.m[3] + v.y * self.m[7] + v.z * self.m[11] + v.w * self.m[15],
        )
    }

    #[inline]
    pub const fn transform_homogenous(&self, v: Point) -> Vector3 {
        Vector3::new(
            v.x * self.m[0] + v.y * self.m[4] + self.m[12],
            v.x * self.m[1] + v.y * self.m[5] + self.m[13],
            v.x * self.m[3] + v.y * self.m[7] + self.m[15],
        )
    }

    #[inline]
    pub const fn transform_direction(&self, v: Vector2) -> Vector2 {
        Vector2::new(
            v.x * self.m[0] + v.y * self.m[4],
            v.x * self.m[1] + v.y * self.m[5],
        )
    }

    #[inline]
    pub const fn transform_direction_3d(&self, v: Vector3) -> Vector3 {
        Vector3::new(
            v.x * self.m[0] + v.y * self.m[4] + v.z * self.m[8],
            v.x * self.m[1] + v.y * self.m[5] + v.z * self.m[9],
            v.x * self.m[2] + v.y * self.m[6] + v.z * self.m[10],
        )
    }

    #[inline]
    pub const fn transform_direction_4d(&self, v: Vector4) -> Vector4 {
        Vector4::new(
            v.x * self.m[0] + v.y * self.m[4] + v.z * self.m[8],
            v.x * self.m[1] + v.y * self.m[5] + v.z * self.m[9],
            v.x * self.m[2] + v.y * self.m[6] + v.z * self.m[10],
            v.w,
        )
    }

    #[inline]
    pub fn transform_quad(&self, quad: Quad) -> Quad {
        [
            *self * quad[0],
            *self * quad[1],
            *self * quad[2],
            *self * quad[3],
        ]
    }
}

impl From<MatrixDecomposition> for Matrix {
    #[inline]
    fn from(d: MatrixDecomposition) -> Self {
        Self::from_decomposition(&d)
    }
}

impl Mul for Matrix {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = &self.m;
        let b = &rhs.m;
        let mut out = [0.0; 16];
        for c in 0..4 {
            for r in 0..4 {
                out[c * 4 + r] = a[r] * b[c * 4]
                    + a[4 + r] * b[c * 4 + 1]
                    + a[8 + r] * b[c * 4 + 2]
                    + a[12 + r] * b[c * 4 + 3];
            }
        }
        Self { m: out }
    }
}

impl Add for Matrix {
    type Output = Self;

    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            m: std::array::from_fn(|i| self.m[i] + o.m[i]),
        }
    }
}

impl Add<Vector3> for Matrix {
    type Output = Self;

    #[inline]
    fn add(self, t: Vector3) -> Self {
        self.translate(t)
    }
}

impl Sub<Vector3> for Matrix {
    type Output = Self;

    #[inline]
    fn sub(self, t: Vector3) -> Self {
        self.translate(-t)
    }
}

impl Mul<Point> for Matrix {
    type Output = Point;

    #[inline]
    fn mul(self, rhs: Point) -> Point {
        self.transform_point(rhs)
    }
}

impl Mul<Vector3> for Matrix {
    type Output = Vector3;

    #[inline]
    fn mul(self, rhs: Vector3) -> Vector3 {
        self.transform_vector3(rhs)
    }
}

impl Mul<Vector4> for Matrix {
    type Output = Vector4;

    #[inline]
    fn mul(self, rhs: Vector4) -> Vector4 {
        self.transform_vector4(rhs)
    }
}
