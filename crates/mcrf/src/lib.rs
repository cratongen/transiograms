#![allow(unused_variables)]

//! Markov chain random field simulation and Markov-type categorical prediction (Li and Zhang
//! Eq. 14) on grids.

pub mod conditional;
pub mod predict;
pub mod search;
pub mod simulate;

pub use conditional::{conditional, conditional_eq13};
pub use predict::predict;
pub use simulate::simulate;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Neighbourhood {
    /// Quadrants in 2D, octants in 3D.
    Sectors,
    Nearest(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub neighbourhood: Neighbourhood,
    pub closeness_limit: f64,
    pub n_realizations: usize,
    pub seed: u64,
    pub memory: Option<u64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            neighbourhood: Neighbourhood::Sectors,
            closeness_limit: 0.01,
            n_realizations: 1,
            seed: 0,
            memory: None,
        }
    }
}
