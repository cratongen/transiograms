//! Hidden semi-Markov models with explicit durations and right censoring, and state-count
//! selection by AIC, BIC, ICL and cross-validated likelihood.

pub mod duration;
pub mod em;
pub mod forward_backward;
pub mod model;
pub mod selection;
pub mod viterbi;

pub use duration::{Duration, DurationDist, DurationFamily};
pub use forward_backward::SegmentPosterior;
pub use model::Hsmm;
pub use selection::{
    Model, ModelKind, SelectionSpec, SelectionTable, aic, bic, icl, select_states,
};

#[cfg(test)]
mod tests;
