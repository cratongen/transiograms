//! Markov chains along drillholes: embedded, fixed-step and continuous-lag fits, statistical
//! tests, entropy, sojourn statistics and 1D simulation with bridges.

pub mod entropy;
pub mod fit;
pub mod simulate;
pub mod sojourn;
pub mod tests;

pub use entropy::Entropy;
pub use fit::{FitOptions, Kind, MarkovChain};
pub use simulate::{Hard, Simulated, Start, Target};
pub use sojourn::{Sojourn, sojourn};
pub use tests::{Method, OrderTable, TestResult, homogeneity_test, order_test, select_order};
