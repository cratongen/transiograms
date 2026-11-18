use serde::{Deserialize, Serialize};
use varvite_core::{Mat, RateMatrix, Result, invalid};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Structure {
    Exponential,
    Spherical,
    Gaussian,
}

impl Structure {
    pub(crate) fn shape(self, r: f64) -> f64 {
        match self {
            Structure::Exponential => 1.0 - (-3.0 * r).exp(),
            Structure::Spherical if r < 1.0 => 1.5 * r - 0.5 * r.powi(3),
            Structure::Spherical => 1.0,
            Structure::Gaussian => 1.0 - (-3.0 * r * r).exp(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub structure: Structure,
    pub range: f64,
    pub nugget: f64,
}

impl Entry {
    /// Entry `(i, j)` of row `i` at lag `h > 0`, before the complement is applied.
    pub(crate) fn value(&self, i: usize, j: usize, p: &[f64], h: f64) -> f64 {
        let g = self.nugget + (1.0 - self.nugget) * self.structure.shape(h / self.range);
        if i == j {
            1.0 - (1.0 - p[i]) * g
        } else {
            g * p[j]
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Model1D {
    Markov {
        proportions: Vec<f64>,
        rates: RateMatrix,
    },
    /// `entries`: k * k, None at the complement entry of each row.
    Entrywise {
        proportions: Vec<f64>,
        entries: Vec<Option<Entry>>,
        complement: Vec<usize>,
    },
}

pub(crate) fn check_proportions(p: &[f64], k: usize) -> Result<()> {
    if k == 0 || p.len() != k {
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

/// `T(-h) = D^-1 T(h)^T D`, rows renormalised.
pub(crate) fn reverse(t: &Mat, p: &[f64]) -> Mat {
    let k = t.nrows();
    let mut r = Mat::from_fn(k, k, |i, j| p[j] * t[(j, i)] / p[i]);
    normalise_rows(&mut r);
    r
}

fn normalise_rows(t: &mut Mat) {
    for mut row in t.row_iter_mut() {
        let sum: f64 = row.iter().sum();
        row /= sum;
    }
}

fn entrywise_raw(p: &[f64], entries: &[Option<Entry>], complement: &[usize], h: f64) -> Mat {
    let k = p.len();
    let mut t = Mat::zeros(k, k);
    for i in 0..k {
        let mut sum = 0.0;
        for j in (0..k).filter(|&j| j != complement[i]) {
            let v = entries[i * k + j]
                .expect("validated entries")
                .value(i, j, p, h);
            t[(i, j)] = v;
            sum += v;
        }
        t[(i, complement[i])] = 1.0 - sum;
    }
    t
}

impl Model1D {
    /// Proportions = stationary law of `rates`.
    pub fn markov(rates: RateMatrix) -> Result<Self> {
        let proportions = rates.stationary()?;
        Ok(Model1D::Markov { proportions, rates })
    }

    /// Validated on a lag grid up to `max_lag`.
    pub fn entrywise(
        proportions: Vec<f64>,
        entries: Vec<Option<Entry>>,
        complement: Vec<usize>,
        max_lag: f64,
    ) -> Result<Self> {
        let k = proportions.len();
        check_proportions(&proportions, k)?;
        if entries.len() != k * k {
            return invalid(format!(
                "give k * k = {} entries, not {}",
                k * k,
                entries.len()
            ));
        }
        if complement.len() != k {
            return invalid(format!(
                "give {k} complement columns, not {}",
                complement.len()
            ));
        }
        if let Some(i) = complement.iter().position(|&c| c >= k) {
            return invalid(format!(
                "complement column {} of row {i} is not a category of {k}",
                complement[i]
            ));
        }
        if !(max_lag > 0.0 && max_lag.is_finite()) {
            return invalid(format!(
                "max_lag must be positive and finite, got {max_lag}"
            ));
        }
        for i in 0..k {
            for j in 0..k {
                match (&entries[i * k + j], j == complement[i]) {
                    (Some(_), true) => {
                        return invalid(format!(
                            "entry ({i}, {j}) is the complement of row {i} and must be None"
                        ));
                    }
                    (None, false) => return invalid(format!("entry ({i}, {j}) is missing")),
                    (Some(e), false) => {
                        if !(e.range > 0.0 && e.range.is_finite()) {
                            return invalid(format!(
                                "entry ({i}, {j}) range {} must be positive and finite",
                                e.range
                            ));
                        }
                        if !(0.0..1.0).contains(&e.nugget) {
                            return invalid(format!(
                                "entry ({i}, {j}) nugget {} must be in [0, 1)",
                                e.nugget
                            ));
                        }
                    }
                    (None, true) => {}
                }
            }
        }
        let lags = (1..=1000)
            .map(|l| max_lag * l as f64 / 1000.0)
            .chain([1e4 * max_lag]);
        for h in lags {
            let t = entrywise_raw(&proportions, &entries, &complement, h);
            for i in 0..k {
                for j in 0..k {
                    if t[(i, j)] < -1e-15 {
                        return invalid(format!(
                            "entrywise model is negative at entry ({i}, {j}), h = {h}: {}",
                            t[(i, j)]
                        ));
                    }
                }
            }
        }
        Ok(Model1D::Entrywise {
            proportions,
            entries,
            complement,
        })
    }

    /// `h < 0` by the reversal identity.
    pub fn transition(&self, h: f64) -> Mat {
        let k = self.proportions().len();
        if h == 0.0 {
            return Mat::identity(k, k);
        }
        if h < 0.0 {
            return reverse(&self.transition(-h), self.proportions());
        }
        match self {
            Model1D::Markov { rates, .. } => varvite_core::numerics::expm_rate(rates.matrix(), h),
            Model1D::Entrywise {
                proportions,
                entries,
                complement,
            } => {
                let mut t = entrywise_raw(proportions, entries, complement, h);
                for mut row in t.row_iter_mut() {
                    if row.iter().any(|&v| v < 0.0) {
                        row.iter_mut().for_each(|v| *v = v.max(0.0));
                        let sum: f64 = row.iter().sum();
                        row /= sum;
                    }
                }
                t
            }
        }
    }

    pub fn proportions(&self) -> &[f64] {
        match self {
            Model1D::Markov { proportions, .. } | Model1D::Entrywise { proportions, .. } => {
                proportions
            }
        }
    }
}
