#![forbid(unsafe_code)]

pub mod color;
pub mod matrix;
pub mod point;
pub mod quaternion;
pub mod rect;
pub mod round_superellipse;
pub mod scalar;
pub mod shear;
pub mod size;
pub mod trig;
pub mod vector;
pub mod wangs_formula;

pub use color::*;
pub use matrix::*;
pub use point::*;
pub use quaternion::*;
pub use rect::*;
pub use round_superellipse::*;
pub use scalar::*;
pub use shear::*;
pub use size::*;
pub use trig::*;
pub use vector::*;
pub use wangs_formula::*;

#[cfg(test)]
mod tests;
