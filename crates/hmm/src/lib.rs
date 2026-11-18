//! Hidden Markov models: emission families, scaled and log forward-backward, Viterbi and
//! posterior decoding, Baum-Welch over many sequences, sampling and information criteria.

pub mod em;
pub mod emission;
pub mod forward_backward;
pub mod init;
pub mod model;
pub mod sequence;
pub mod viterbi;

pub use em::update_emission;
pub use emission::{
    Categorical, CovarianceType, Emission, Family, Gaussian, GaussianMixture, Poisson,
};
pub use forward_backward::Posterior;
pub use init::kmeans;
pub use model::{
    Decoder, FitOptions, FitReport, Hmm, Implementation, Ordering, Params, Priors,
    emission_log_prior, n_observed,
};
pub use sequence::{Clamp, Sequence, pooled_moments};
