use serde::{Deserialize, Serialize};
use varvite_core::numerics::{golden_section, log_add, logsumexp};
use varvite_core::special::ln_gamma;
use varvite_core::{Result, Stream, invalid};

/// Shifted laws on u = 1, 2, ...; truncated and renormalised on `1..=max_duration`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "law", rename_all = "snake_case")]
pub enum Duration {
    /// `p (1 - p)^(u - 1)`, `0 < p <= 1`.
    Geometric { p: f64 },
    /// `e^-rate rate^(u-1) / (u-1)!`, `rate >= 0`.
    Poisson { rate: f64 },
    /// `C(u + r - 2, u - 1) p^r (1 - p)^(u - 1)`.
    NegativeBinomial { r: f64, p: f64 },
    /// `pmf[u - 1]`; support `1..=pmf.len()`.
    Empirical { pmf: Vec<f64> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DurationDist {
    pub law: Duration,
    pub frozen: bool,
    pub pseudocount: f64,
}

fn ln_mul(a: f64, b: f64) -> f64 {
    if a == 0.0 { 0.0 } else { a * b }
}

impl Duration {
    /// Untruncated log pmf at `u >= 1`.
    fn raw_log_pmf(&self, u: usize) -> f64 {
        let k = (u - 1) as f64;
        match self {
            Duration::Geometric { p } => p.ln() + ln_mul(k, (-p).ln_1p()),
            Duration::Poisson { rate } => -rate + ln_mul(k, rate.ln()) - ln_gamma(u as f64),
            Duration::NegativeBinomial { r, p } => {
                if *p >= 1.0 {
                    return if u == 1 { 0.0 } else { f64::NEG_INFINITY };
                }
                ln_gamma(k + r) - ln_gamma(*r) - ln_gamma(u as f64)
                    + r * p.ln()
                    + ln_mul(k, (-p).ln_1p())
            }
            Duration::Empirical { pmf } => match pmf.get(u - 1) {
                Some(&v) if v > 0.0 => v.ln(),
                _ => f64::NEG_INFINITY,
            },
        }
    }

    pub(crate) fn validate(&self, m: usize) -> Result<()> {
        let ok = match self {
            Duration::Geometric { p } => *p > 0.0 && *p <= 1.0,
            Duration::Poisson { rate } => *rate >= 0.0 && rate.is_finite(),
            Duration::NegativeBinomial { r, p } => {
                *r > 0.0 && r.is_finite() && *p > 0.0 && *p <= 1.0
            }
            Duration::Empirical { pmf } => {
                !pmf.is_empty() && pmf.iter().all(|v| *v >= 0.0 && v.is_finite())
            }
        };
        if !ok {
            return invalid(format!("invalid duration law {self:?}"));
        }
        if m == 0 {
            return invalid("max_duration must be at least 1");
        }
        let total = logsumexp(&(1..=m).map(|u| self.raw_log_pmf(u)).collect::<Vec<_>>());
        if !total.is_finite() {
            return invalid(format!(
                "duration law {self:?} has no mass on 1..={m}; raise max_duration"
            ));
        }
        Ok(())
    }

    /// Index `u - 1`, renormalised on `1..=m`.
    pub fn log_pmf(&self, m: usize) -> Vec<f64> {
        let raw: Vec<f64> = (1..=m).map(|u| self.raw_log_pmf(u)).collect();
        let total = logsumexp(&raw);
        raw.into_iter().map(|v| v - total).collect()
    }

    /// `log D(u) = log sum_{v >= u} d(v)`.
    pub fn log_survivor(&self, m: usize) -> Vec<f64> {
        let lp = self.log_pmf(m);
        let mut out = vec![f64::NEG_INFINITY; m];
        let mut acc = f64::NEG_INFINITY;
        for u in (0..m).rev() {
            acc = log_add(acc, lp[u]);
            out[u] = acc;
        }
        out
    }

    pub fn mean(&self, m: usize) -> f64 {
        self.log_pmf(m)
            .iter()
            .enumerate()
            .map(|(i, l)| (i + 1) as f64 * l.exp())
            .sum()
    }

    /// 1, 1, 2, m - 1.
    pub fn n_params(&self, m: usize) -> usize {
        match self {
            Duration::Geometric { .. } | Duration::Poisson { .. } => 1,
            Duration::NegativeBinomial { .. } => 2,
            Duration::Empirical { .. } => m.saturating_sub(1),
        }
    }

    /// Weighted MLE from `counts[u - 1]`.
    pub fn fit(&self, counts: &[f64], pseudocount: f64) -> Result<Duration> {
        if counts.iter().any(|c| !c.is_finite() || *c < 0.0) {
            return invalid("duration counts must be finite and non-negative");
        }
        let total: f64 = counts.iter().sum();
        let first: f64 = counts
            .iter()
            .enumerate()
            .map(|(i, c)| (i + 1) as f64 * c)
            .sum();
        if let Duration::Empirical { .. } = self {
            let denom = total + pseudocount * counts.len() as f64;
            if denom.is_nan() || denom <= 0.0 {
                return invalid("empirical duration fit needs positive counts or pseudocount");
            }
            return Ok(Duration::Empirical {
                pmf: counts.iter().map(|c| (c + pseudocount) / denom).collect(),
            });
        }
        if total.is_nan() || total <= 0.0 {
            return invalid("duration fit needs positive total count");
        }
        let mbar = first / total;
        Ok(match self {
            Duration::Geometric { .. } => Duration::Geometric {
                p: (total / first).min(1.0),
            },
            Duration::Poisson { .. } => Duration::Poisson {
                rate: (mbar - 1.0).max(0.0),
            },
            Duration::NegativeBinomial { .. } => {
                if mbar <= 1.0 + 1e-12 {
                    return Ok(Duration::NegativeBinomial { r: 1.0, p: 1.0 });
                }
                let m = counts.len();
                let law = |lr: f64| {
                    let r = lr.exp();
                    Duration::NegativeBinomial {
                        r,
                        p: r / (r + mbar - 1.0),
                    }
                };
                let neg_score = |lr: f64| -law(lr).weighted_log_lik(counts, m);
                let (lr, _) = golden_section(neg_score, 1e-3f64.ln(), 1e4f64.ln(), 1e-10);
                law(lr)
            }
            Duration::Empirical { .. } => unreachable!(),
        })
    }

    /// `sum_u counts[u - 1] ln d(u)` with `d` truncated on `1..=m`, `0 ln 0 = 0`.
    pub(crate) fn weighted_log_lik(&self, counts: &[f64], m: usize) -> f64 {
        self.log_pmf(m)
            .iter()
            .zip(counts)
            .map(|(l, c)| ln_mul(*c, *l))
            .sum()
    }

    /// Untruncated law's mass above `m`.
    pub(crate) fn mass_above(&self, m: usize) -> f64 {
        match self {
            Duration::Geometric { p } => ((-p).ln_1p() * m as f64).exp(),
            Duration::Empirical { pmf } => {
                let total: f64 = pmf.iter().sum();
                pmf.iter().skip(m).sum::<f64>() / total
            }
            _ => {
                let below: f64 = (1..=m).map(|u| self.raw_log_pmf(u).exp()).sum();
                (1.0 - below).max(0.0)
            }
        }
    }

    /// Inverse CDF draw on `1..=m`.
    pub(crate) fn draw(log_pmf: &[f64], stream: &mut Stream) -> usize {
        let w: Vec<f64> = log_pmf.iter().map(|l| l.exp()).collect();
        stream.categorical(&w) + 1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DurationFamily {
    Geometric,
    Poisson,
    NegativeBinomial,
    Empirical,
}

impl DurationFamily {
    pub(crate) fn template(self, m: usize) -> Duration {
        match self {
            DurationFamily::Geometric => Duration::Geometric { p: 0.5 },
            DurationFamily::Poisson => Duration::Poisson { rate: 1.0 },
            DurationFamily::NegativeBinomial => Duration::NegativeBinomial { r: 1.0, p: 0.5 },
            DurationFamily::Empirical => Duration::Empirical {
                pmf: vec![1.0 / m as f64; m],
            },
        }
    }
}
