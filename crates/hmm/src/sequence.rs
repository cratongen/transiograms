/// `t x d` row-major; categorical codes stored as f64; NaN marks a missing channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Sequence {
    pub t: usize,
    pub d: usize,
    pub x: Vec<f64>,
}

/// `t x K` row-major; false forbids that state at that step.
#[derive(Clone, Debug, PartialEq)]
pub struct Clamp {
    pub k: usize,
    pub allowed: Vec<bool>,
}
