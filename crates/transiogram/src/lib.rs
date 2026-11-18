#![allow(unused_variables)]

//! Experimental transiograms (points, downhole intervals, grids), 1D transiogram models and the
//! 3D transition model with rotation, rate interpolation, closeness and offset tables.

pub mod experimental;
pub mod fit;
pub mod model1d;
pub mod model3d;
pub mod table;

pub use experimental::{Experimental, PairFilter, PairSpec, downhole, experimental, on_grid};
pub use model1d::{Entry, Model1D, Structure};
pub use model3d::TransitionModel;
pub use table::Table;
