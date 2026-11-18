use varvite_core::{Result, invalid};

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

impl Sequence {
    pub(crate) fn row(&self, t: usize) -> &[f64] {
        &self.x[t * self.d..(t + 1) * self.d]
    }
}

impl Clamp {
    pub(crate) fn allows(&self, t: usize, j: usize) -> bool {
        self.allowed[t * self.k + j]
    }

    /// The single allowed state at step `t`, if exactly one.
    pub(crate) fn label(&self, t: usize) -> Option<usize> {
        let row = &self.allowed[t * self.k..(t + 1) * self.k];
        let mut it = (0..self.k).filter(|&j| row[j]);
        match (it.next(), it.next()) {
            (Some(j), None) => Some(j),
            _ => None,
        }
    }
}

pub(crate) fn check_sequence(
    seq: &Sequence,
    clamp: Option<&Clamp>,
    n: usize,
    d: usize,
    k: usize,
    symbols: Option<usize>,
) -> Result<()> {
    if seq.t == 0 {
        return invalid(format!("sequence {n} is empty"));
    }
    if seq.d != d {
        return invalid(format!(
            "sequence {n} has {} channels, the model has {d}",
            seq.d
        ));
    }
    if seq.x.len() != seq.t * seq.d {
        return invalid(format!(
            "sequence {n} holds {} values, not t x d = {}",
            seq.x.len(),
            seq.t * seq.d
        ));
    }
    for (i, &v) in seq.x.iter().enumerate() {
        if v.is_infinite() {
            return invalid(format!("sequence {n}, step {}: infinite value", i / d));
        }
        if let Some(m) = symbols
            && !v.is_nan()
            && !(v >= 0.0 && v.fract() == 0.0 && v < m as f64)
        {
            return invalid(format!(
                "sequence {n}, step {}: code {v} is not in 0..{m}",
                i / d
            ));
        }
    }
    if let Some(c) = clamp
        && (c.k != k || c.allowed.len() != seq.t * k)
    {
        return invalid(format!(
            "clamp of sequence {n} must be {} x {k}, got k = {} with {} entries",
            seq.t,
            c.k,
            c.allowed.len()
        ));
    }
    Ok(())
}

/// Per-channel population variance of the observed values, pooled over sequences in order.
pub fn pooled_moments(data: &[Sequence]) -> (Vec<f64>, Vec<f64>) {
    let d = data.first().map_or(0, |s| s.d);
    let (mut n, mut mean, mut m2) = (vec![0.0; d], vec![0.0; d], vec![0.0; d]);
    for seq in data {
        for t in 0..seq.t {
            for (c, &v) in seq.row(t).iter().enumerate() {
                if v.is_nan() {
                    continue;
                }
                n[c] += 1.0;
                let delta = v - mean[c];
                mean[c] += delta / n[c];
                m2[c] += delta * (v - mean[c]);
            }
        }
    }
    let var = (0..d)
        .map(|c| if n[c] > 0.0 { m2[c] / n[c] } else { 0.0 })
        .collect();
    (mean, var)
}
