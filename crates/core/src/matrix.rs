//! Validated transition (stochastic) and rate (generator) matrices.

use serde::{Deserialize, Serialize};

use crate::error::{Result, convergence, invalid};
use crate::numerics::{expm_rate, gth_stationary, logm_stochastic, quasi_independence};
use crate::{Mat, ROW_TOL};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmptyRow {
    Raise,
    Uniform,
}

fn check_square(m: &Mat, what: &str) -> Result<()> {
    if m.nrows() == 0 || m.nrows() != m.ncols() {
        return invalid(format!(
            "{what} must be a non-empty square matrix, got {}x{}",
            m.nrows(),
            m.ncols()
        ));
    }
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            if !m[(i, j)].is_finite() {
                return invalid(format!(
                    "{what} entry ({i}, {j}) = {} is not finite",
                    m[(i, j)]
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionMatrix {
    #[serde(with = "crate::serde_mat")]
    p: Mat,
}

impl TransitionMatrix {
    /// Entries in [0, 1] and every row sum within `ROW_TOL` of 1.
    pub fn new(p: Mat) -> Result<Self> {
        check_square(&p, "transition matrix")?;
        for i in 0..p.nrows() {
            for j in 0..p.ncols() {
                let v = p[(i, j)];
                if !(0.0..=1.0).contains(&v) {
                    return invalid(format!(
                        "transition matrix entry ({i}, {j}) = {v} is outside [0, 1]"
                    ));
                }
            }
            let sum: f64 = p.row(i).sum();
            if (sum - 1.0).abs() > ROW_TOL {
                return invalid(format!("transition matrix row {i} sums to {sum}, not 1"));
            }
        }
        Ok(Self { p })
    }

    /// `(n_ij + a) / (n_i. + a * allowed_i)`; allowed cells exclude the diagonal when `embedded`.
    /// A row with no count and `a = 0` follows `on_empty` (Uniform: uniform on allowed cells).
    pub fn from_counts(
        counts: &Mat,
        pseudocount: f64,
        embedded: bool,
        on_empty: EmptyRow,
    ) -> Result<Self> {
        check_square(counts, "counts")?;
        if !(pseudocount >= 0.0 && pseudocount.is_finite()) {
            return invalid(format!(
                "pseudocount must be finite and >= 0, got {pseudocount}"
            ));
        }
        let k = counts.nrows();
        if embedded && k < 2 {
            return invalid("an embedded chain needs at least 2 categories");
        }
        let mut p = Mat::zeros(k, k);
        for i in 0..k {
            let allowed = |j: usize| !(embedded && i == j);
            for j in 0..k {
                let c = counts[(i, j)];
                if c < 0.0 {
                    return invalid(format!("count ({i}, {j}) = {c} is negative"));
                }
                if !allowed(j) && c != 0.0 {
                    return invalid(format!(
                        "embedded counts must have a zero diagonal: ({i}, {i}) = {c}"
                    ));
                }
            }
            let total: f64 = (0..k)
                .filter(|&j| allowed(j))
                .map(|j| counts[(i, j)] + pseudocount)
                .sum();
            if total == 0.0 {
                match on_empty {
                    EmptyRow::Raise => {
                        return invalid(format!(
                            "category {i} has no transitions; pass on_empty_row=\"uniform\" or a pseudocount"
                        ));
                    }
                    EmptyRow::Uniform => {
                        let n = (0..k).filter(|&j| allowed(j)).count() as f64;
                        (0..k)
                            .filter(|&j| allowed(j))
                            .for_each(|j| p[(i, j)] = 1.0 / n);
                    }
                }
            } else {
                (0..k)
                    .filter(|&j| allowed(j))
                    .for_each(|j| p[(i, j)] = (counts[(i, j)] + pseudocount) / total);
            }
        }
        Self::new(p)
    }

    pub fn k(&self) -> usize {
        self.p.nrows()
    }

    pub fn matrix(&self) -> &Mat {
        &self.p
    }

    /// Every diagonal entry exactly 0.
    pub fn is_embedded(&self) -> bool {
        (0..self.k()).all(|i| self.p[(i, i)] == 0.0)
    }

    /// GTH on `P - I`; `InvalidInput` when reducible. Periodic chains are fine.
    pub fn stationary(&self) -> Result<Vec<f64>> {
        gth_stationary(&self.p)
    }

    /// `P^n` by repeated squaring.
    pub fn power(&self, n: u64) -> Mat {
        let k = self.k();
        let mut result = Mat::identity(k, k);
        let mut base = self.p.clone();
        let mut n = n;
        while n > 0 {
            if n & 1 == 1 {
                result = &result * &base;
            }
            n >>= 1;
            if n > 0 {
                base = &base * &base;
            }
        }
        result
    }

    /// Time reversal `P*_ij = pi_j P_ji / pi_i` for a stationary law `pi`.
    pub fn reversed(&self, pi: &[f64]) -> Result<TransitionMatrix> {
        let k = self.k();
        if pi.len() != k {
            return invalid(format!(
                "give {k} stationary probabilities, not {}",
                pi.len()
            ));
        }
        if let Some(i) = pi.iter().position(|&v| !(v > 0.0 && v.is_finite())) {
            return invalid(format!(
                "stationary probability {i} = {} must be positive",
                pi[i]
            ));
        }
        let mut r = Mat::from_fn(k, k, |i, j| pi[j] * self.p[(j, i)] / pi[i]);
        for i in 0..k {
            let sum: f64 = r.row(i).sum();
            if (sum - 1.0).abs() > 1e-9 {
                return invalid(format!(
                    "pi is not stationary for P: reversed row {i} sums to {sum}"
                ));
            }
            r.row_mut(i)
                .iter_mut()
                .for_each(|v| *v = (*v / sum).min(1.0));
        }
        Self::new(r)
    }
}

/// How far the matrix logarithm moved to become a generator: max |change| of its entries.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Projection {
    pub distance: f64,
    pub exact: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RateMatrix {
    #[serde(with = "crate::serde_mat")]
    q: Mat,
}

fn max_abs(m: &Mat) -> f64 {
    m.iter().fold(0.0, |a, v| a.max(v.abs()))
}

fn check_probabilities(p: &[f64], k: usize) -> Result<()> {
    if p.len() != k {
        return invalid(format!("give {k} proportions, not {}", p.len()));
    }
    if let Some(i) = p.iter().position(|&v| !(v > 0.0 && v.is_finite())) {
        return invalid(format!("proportion {i} = {} must be positive", p[i]));
    }
    let sum: f64 = p.iter().sum();
    if (sum - 1.0).abs() > 1e-12 {
        return invalid(format!("proportions sum to {sum}, not 1"));
    }
    Ok(())
}

fn check_lengths(l: &[f64], k: usize) -> Result<()> {
    if l.len() != k {
        return invalid(format!("give {k} mean lengths, not {}", l.len()));
    }
    if let Some(i) = l.iter().position(|&v| !(v > 0.0 && v.is_finite())) {
        return invalid(format!(
            "mean length {i} = {} must be positive and finite",
            l[i]
        ));
    }
    Ok(())
}

impl RateMatrix {
    /// Off-diagonals >= 0, diagonals < 0, rows summing to 0 within `ROW_TOL * max|q|`.
    /// A single category has the generator `[[0]]`.
    pub fn new(q: Mat) -> Result<Self> {
        check_square(&q, "rate matrix")?;
        let k = q.nrows();
        if k == 1 {
            if q[(0, 0)] != 0.0 {
                return invalid(format!(
                    "a one-category rate matrix is [[0]], got [[{}]]",
                    q[(0, 0)]
                ));
            }
            return Ok(Self { q });
        }
        let scale = max_abs(&q);
        for i in 0..k {
            for j in 0..k {
                let v = q[(i, j)];
                if i != j && v < 0.0 {
                    return invalid(format!("rate matrix entry ({i}, {j}) = {v} is negative"));
                }
                if i == j && v >= 0.0 {
                    return invalid(format!(
                        "rate matrix diagonal ({i}, {i}) = {v} must be negative"
                    ));
                }
            }
            let sum: f64 = q.row(i).sum();
            if sum.abs() > ROW_TOL * scale {
                return invalid(format!("rate matrix row {i} sums to {sum}, not 0"));
            }
        }
        Ok(Self { q })
    }

    /// `Q = diag(1 / L) (Pi - I)` for an embedded `Pi` and mean lengths `L`.
    pub fn from_embedded(pi: &TransitionMatrix, mean_lengths: &[f64]) -> Result<Self> {
        let k = pi.k();
        if !pi.is_embedded() {
            return invalid("from_embedded needs an embedded transition matrix (zero diagonal)");
        }
        check_lengths(mean_lengths, k)?;
        let p = pi.matrix();
        let q = Mat::from_fn(k, k, |i, j| {
            if i == j {
                -1.0 / mean_lengths[i]
            } else {
                p[(i, j)] / mean_lengths[i]
            }
        });
        Self::new(q)
    }

    /// `logm(P) / step` projected to the nearest generator (Israel, Rosenthal and Wei).
    pub fn from_transition(p: &TransitionMatrix, step: f64) -> Result<(Self, Projection)> {
        if !(step > 0.0 && step.is_finite()) {
            return invalid(format!("step must be positive and finite, got {step}"));
        }
        let q0 = logm_stochastic(p.matrix())? / step;
        let k = q0.nrows();
        let mut q = q0.map(|v| v.max(0.0));
        for i in 0..k {
            q[(i, i)] = 0.0;
            let off: f64 = q.row(i).sum();
            q[(i, i)] = -off;
        }
        if k > 1 && (0..k).any(|i| q[(i, i)] == 0.0) {
            return convergence(
                "the logarithm has a row with no positive rate: a category never leaves",
            );
        }
        let distance = max_abs(&(&q - &q0));
        let exact = distance <= 1e-12 * max_abs(&q0);
        Ok((Self::new(q)?, Projection { distance, exact }))
    }

    /// Carle background fill: rates given for all i, j != `background` (diagonals included);
    /// row and column `background` filled from row sums and `p^T Q = 0`.
    pub fn from_partial(partial: &Mat, proportions: &[f64], background: usize) -> Result<Self> {
        check_square(partial, "partial rate matrix")?;
        let k = partial.nrows();
        check_probabilities(proportions, k)?;
        if background >= k {
            return invalid(format!("background {background} is not a category of {k}"));
        }
        let b = background;
        let mut r = partial.clone();
        let scale = (0..k)
            .flat_map(|i| (0..k).map(move |j| (i, j)))
            .filter(|&(i, j)| i != b && j != b)
            .fold(0.0, |a: f64, (i, j)| a.max(partial[(i, j)].abs()));
        let fix = |v: f64, i: usize, j: usize| -> Result<f64> {
            if v < -1e-12 * scale {
                return invalid(format!(
                    "rates incompatible with proportions: entry ({i}, {j}) = {v}"
                ));
            }
            Ok(v.max(0.0))
        };
        for i in (0..k).filter(|&i| i != b) {
            let v = -(0..k).filter(|&j| j != b).map(|j| r[(i, j)]).sum::<f64>();
            r[(i, b)] = fix(v, i, b)?;
        }
        for j in (0..k).filter(|&j| j != b) {
            let v = -(0..k)
                .filter(|&i| i != b)
                .map(|i| proportions[i] * r[(i, j)])
                .sum::<f64>()
                / proportions[b];
            r[(b, j)] = fix(v, b, j)?;
        }
        r[(b, b)] = -(0..k).filter(|&j| j != b).map(|j| r[(b, j)]).sum::<f64>();
        Self::new(r)
    }

    /// Maximum-entropy (quasi-independent) embedded chain with proportions `p` and mean lengths `L`.
    pub fn max_entropy(proportions: &[f64], mean_lengths: &[f64]) -> Result<Self> {
        let k = proportions.len();
        check_probabilities(proportions, k)?;
        check_lengths(mean_lengths, k)?;
        if k == 1 {
            return Self::new(Mat::zeros(1, 1));
        }
        let phi: Vec<f64> = (0..k).map(|i| proportions[i] / mean_lengths[i]).collect();
        let total: f64 = phi.iter().sum();
        for i in 0..k {
            if phi[i] > (total - phi[i]) * (1.0 + 1e-12) {
                return invalid(format!(
                    "no embedded chain has these proportions and lengths: category {i} has p/L = {} above the other categories' sum {}",
                    phi[i],
                    total - phi[i]
                ));
            }
        }
        let e = if k == 2 {
            Mat::from_row_slice(2, 2, &[0.0, phi[0], phi[1], 0.0])
        } else {
            quasi_independence(&phi, &phi, 1e-13, 10_000)?
        };
        let q = Mat::from_fn(k, k, |i, j| {
            if i == j {
                -1.0 / mean_lengths[i]
            } else {
                e[(i, j)] / phi[i] / mean_lengths[i]
            }
        });
        let mut q = q;
        for i in 0..k {
            let off: f64 = (0..k).filter(|&j| j != i).map(|j| q[(i, j)]).sum();
            let f = 1.0 / mean_lengths[i] / off;
            (0..k).filter(|&j| j != i).for_each(|j| q[(i, j)] *= f);
        }
        let scale = max_abs(&q);
        for j in 0..k {
            let s: f64 = (0..k).map(|i| proportions[i] * q[(i, j)]).sum();
            if s.abs() > 1e-12 * scale {
                return convergence(format!(
                    "maximum-entropy rates miss p^T R = 0 by {s} in column {j}"
                ));
            }
        }
        Self::new(q)
    }

    pub fn k(&self) -> usize {
        self.q.nrows()
    }

    pub fn matrix(&self) -> &Mat {
        &self.q
    }

    /// `-1 / q_jj`.
    pub fn mean_lengths(&self) -> Vec<f64> {
        (0..self.k()).map(|j| -1.0 / self.q[(j, j)]).collect()
    }

    /// `q_jk / -q_jj` with a zero diagonal.
    pub fn embedded(&self) -> TransitionMatrix {
        let k = self.k();
        let p = Mat::from_fn(k, k, |i, j| {
            if i == j {
                0.0
            } else {
                self.q[(i, j)] / -self.q[(i, i)]
            }
        });
        TransitionMatrix { p }
    }

    /// GTH on Q.
    pub fn stationary(&self) -> Result<Vec<f64>> {
        gth_stationary(&self.q)
    }

    /// `T(h) = exp(Q h)` for `h >= 0` by uniformization.
    pub fn transition(&self, h: f64) -> TransitionMatrix {
        assert!(h >= 0.0, "transition(h) needs h >= 0, got {h}");
        TransitionMatrix {
            p: expm_rate(&self.q, h),
        }
    }

    /// Time reversal `D^-1 Q^T D`, `D = diag(p)`.
    pub fn reversed(&self, p: &[f64]) -> RateMatrix {
        let k = self.k();
        let mut r = Mat::from_fn(k, k, |i, j| {
            if i == j {
                self.q[(i, i)]
            } else {
                p[j] * self.q[(j, i)] / p[i]
            }
        });
        for i in 0..k {
            let off: f64 = (0..k).filter(|&j| j != i).map(|j| r[(i, j)]).sum();
            if off > 0.0 {
                let f = -r[(i, i)] / off;
                (0..k).filter(|&j| j != i).for_each(|j| r[(i, j)] *= f);
            }
        }
        RateMatrix { q: r }
    }

    /// `max |p_i q_ij - p_j q_ji| <= tol * max |q|`.
    pub fn is_reversible(&self, p: &[f64], tol: f64) -> bool {
        let k = self.k();
        let scale = max_abs(&self.q);
        (0..k).all(|i| {
            (0..k).all(|j| (p[i] * self.q[(i, j)] - p[j] * self.q[(j, i)]).abs() <= tol * scale)
        })
    }
}
