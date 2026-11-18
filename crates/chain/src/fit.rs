use serde::{Deserialize, Serialize};
use varvite_core::{
    Direction, EmptyRow, Mat, Projection, RateMatrix, Result, SequenceSet, TransitionMatrix,
};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Kind {
    Embedded,
    Fixed {
        step: f64,
    },
    /// Only for chains built from rates.
    Continuous,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitOptions {
    pub kind: Kind,
    pub direction: Direction,
    pub gap: f64,
    /// Embedded only.
    pub self_transitions: bool,
    pub pseudocount: f64,
    pub on_empty_row: EmptyRow,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            kind: Kind::Embedded,
            direction: Direction::Up,
            gap: 1e-6,
            self_transitions: false,
            pseudocount: 0.0,
            on_empty_row: EmptyRow::Raise,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarkovChain {
    pub kind: Kind,
    pub direction: Direction,
    /// Embedded: bed transitions; Fixed: step transitions.
    #[serde(with = "varvite_core::serde_mat::option")]
    pub counts: Option<Mat>,
    /// Total thickness `T_j` of each category (Embedded).
    pub exposure: Option<Vec<f64>>,
    /// Pi (Embedded, Continuous) or P(step) (Fixed).
    pub transition: TransitionMatrix,
    pub generator: Option<RateMatrix>,
    pub proportions: Vec<f64>,
    pub mean_lengths: Option<Vec<f64>>,
    pub bed_frequencies: Vec<f64>,
    /// Fixed: how far `logm(P) / step` moved to be a generator.
    pub projection: Option<Projection>,
}

impl MarkovChain {
    pub fn fit(seqs: &SequenceSet, opts: &FitOptions) -> Result<Self> {
        todo!()
    }

    pub fn from_transition(
        p: TransitionMatrix,
        kind: Kind,
        mean_lengths: Option<Vec<f64>>,
        direction: Direction,
    ) -> Result<Self> {
        todo!()
    }

    /// Kind Continuous.
    pub fn from_rates(q: RateMatrix, direction: Direction) -> Result<Self> {
        todo!()
    }

    /// Embedded given: `RateMatrix::from_partial` with `partial_ij = pi_ij / L_i` (i != q);
    /// symmetric: only i < j of `embedded` is read, the rest from `p_i r_ij = p_j r_ji`.
    /// Embedded None: `RateMatrix::max_entropy(p, L)`.
    pub fn from_geology(
        p: &[f64],
        mean_lengths: &[f64],
        embedded: Option<&Mat>,
        background: usize,
        symmetric: bool,
        direction: Direction,
    ) -> Result<Self> {
        todo!()
    }

    pub fn k(&self) -> usize {
        todo!()
    }

    /// Generator, or `P^n` for Fixed when `h = n step`.
    pub fn transition_at(&self, h: f64) -> Result<TransitionMatrix> {
        todo!()
    }

    /// Kemeny and Snell standard deviation of the proportions after `n_steps`.
    pub fn proportion_std(&self, n_steps: usize) -> Result<Vec<f64>> {
        todo!()
    }
}
