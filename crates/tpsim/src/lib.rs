#![allow(unused_variables)]

//! Transition-probability cokriging, sequential indicator simulation and zero-temperature
//! quenching on grids (T-PROGS).

pub mod cokriging;
pub mod quench;
pub mod search;
pub mod sis;

pub use cokriging::{cokrige, correct, predict};
pub use quench::{objective, quench, quench_lags};
pub use sis::simulate;

#[derive(Clone, Debug, PartialEq)]
pub struct QuenchOptions {
    pub max_iter: usize,
    pub limit: f64,
    pub tol: f64,
    pub proportion_weight: f64,
}

impl Default for QuenchOptions {
    fn default() -> Self {
        Self {
            max_iter: 3,
            limit: 0.4,
            tol: 1e-3,
            proportion_weight: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub n_realizations: usize,
    pub seed: u64,
    /// Even.
    pub max_data: usize,
    pub closeness_limit: f64,
    /// 0 = no cap.
    pub max_per_octant: usize,
    /// Relative to the largest singular value.
    pub svd_cutoff: f64,
    pub quench: Option<QuenchOptions>,
    pub memory: Option<u64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            n_realizations: 1,
            seed: 0,
            max_data: 12,
            closeness_limit: 0.01,
            max_per_octant: 0,
            svd_cutoff: 1e-8,
            quench: Some(QuenchOptions::default()),
            memory: None,
        }
    }
}
