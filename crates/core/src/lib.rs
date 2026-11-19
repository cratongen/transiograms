//! Shared types and numerics for transiograms: category codes, drillhole sequences, transition and
//! rate matrices, grids, seeded random streams, special functions and the memory budget.

pub mod codes;
pub mod error;
pub mod grid;
pub mod matrix;
pub mod memory;
pub mod numerics;
pub mod rng;
pub mod sequence;
pub mod serde_mat;
pub mod special;
pub mod synthetic;

pub use nalgebra;
pub type Mat = nalgebra::DMatrix<f64>;
pub const ROW_TOL: f64 = 1e-12;
pub use codes::{Categories, Code, MISSING};
pub use error::{Error, Result, convergence, invalid};
pub use grid::{Grid, Prediction, Realizations, Snapped};
pub use matrix::{EmptyRow, Projection, RateMatrix, TransitionMatrix};
pub use rng::Stream;
pub use sequence::{Bed, Direction, Hole, Run, SequenceSet};
