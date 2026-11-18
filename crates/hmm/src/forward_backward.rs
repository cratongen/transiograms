use varvite_core::numerics::logsumexp;
use varvite_core::rng::tag;
use varvite_core::{Error, Mat, Result, Stream, invalid, memory};

use crate::emission::Density;
use crate::model::{Hmm, Implementation};
use crate::sequence::{Clamp, Sequence};

#[derive(Clone, Debug, PartialEq)]
pub struct Posterior {
    pub log_likelihood: f64,
    /// `t x K`.
    pub gamma: Mat,
    /// `K x K`, sum over t of `xi_t`.
    pub xi_sum: Mat,
    /// `H(S | X)` of the whole path.
    pub entropy: f64,
}

pub(crate) struct Pass {
    pub ll: f64,
    pub gamma: Mat,
    pub xi_sum: Mat,
}

pub(crate) fn impossible(n: usize, t: usize) -> Error {
    Error::InvalidInput(format!(
        "observation impossible under the model: sequence {n}, step {t}"
    ))
}

impl Hmm {
    /// `t x K` emission log densities, `-inf` where the clamp forbids a state.
    pub(crate) fn emission_logs(&self, seq: &Sequence, clamp: Option<&Clamp>) -> Mat {
        let k = self.k();
        let mut dens: Vec<Density> = self.emissions.iter().map(Density::new).collect();
        let mut lb = Mat::zeros(seq.t, k);
        for t in 0..seq.t {
            let row = seq.row(t);
            for (j, dj) in dens.iter_mut().enumerate() {
                lb[(t, j)] = if clamp.is_some_and(|c| !c.allows(t, j)) {
                    f64::NEG_INFINITY
                } else {
                    dj.log_density(row)
                };
            }
        }
        lb
    }

    pub(crate) fn log_start(&self) -> Vec<f64> {
        self.start.iter().map(|p| p.ln()).collect()
    }

    pub(crate) fn log_transition(&self) -> Mat {
        self.transition.map(f64::ln)
    }

    /// `lb - m_t` with `m_t = max_j lb_j(x_t)` (0 for an all `-inf` row), and the `m_t`.
    fn shifted(lb: &Mat) -> (Mat, Vec<f64>) {
        let mut out = lb.clone();
        let mut m = vec![0.0; lb.nrows()];
        for t in 0..lb.nrows() {
            let mt = lb.row(t).iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if mt > f64::NEG_INFINITY {
                m[t] = mt;
                out.row_mut(t).add_scalar_mut(-mt);
            }
        }
        (out, m)
    }

    /// Log forward pass returning `ln alpha_t - ln P(x_1..x_t)` (log of the normalised
    /// forward vector), the shifted emissions and `ln L`.
    pub(crate) fn forward_log(&self, lb: &Mat, n: usize) -> Result<(Mat, Mat, f64)> {
        // Every row is renormalised and emissions are shifted by their row max, so all
        // stored values stay O(1) and long sequences lose no absolute precision; ln L is
        // the sum of the per-step shifts and normalisers.
        let (tn, k) = lb.shape();
        let (ls, m) = Self::shifted(lb);
        let (lpi, lat) = (self.log_start(), self.log_transition());
        let mut la = Mat::zeros(tn, k);
        let mut tmp = vec![0.0; k];
        let mut row = vec![0.0; k];
        let mut ll = 0.0;
        for t in 0..tn {
            for j in 0..k {
                row[j] = ls[(t, j)]
                    + if t == 0 {
                        lpi[j]
                    } else {
                        for i in 0..k {
                            tmp[i] = la[(t - 1, i)] + lat[(i, j)];
                        }
                        logsumexp(&tmp)
                    };
            }
            let s = logsumexp(&row);
            if s == f64::NEG_INFINITY {
                return Err(impossible(n, t));
            }
            for j in 0..k {
                la[(t, j)] = row[j] - s;
            }
            ll += s + m[t];
        }
        Ok((la, ls, ll))
    }

    fn fb_log(&self, lb: &Mat, n: usize) -> Result<Pass> {
        let (tn, k) = lb.shape();
        let (la, ls, ll) = self.forward_log(lb, n)?;
        let lat = self.log_transition();
        let mut gamma = Mat::zeros(tn, k);
        let mut xi_sum = Mat::zeros(k, k);
        let mut beta_next = vec![0.0; k];
        let mut beta = vec![0.0; k];
        let mut tmp = vec![0.0; k];
        let mut e = vec![0.0; k];
        let set_gamma = |gamma: &mut Mat, t: usize, beta: &[f64], tmp: &mut [f64]| {
            for i in 0..k {
                tmp[i] = la[(t, i)] + beta[i];
            }
            let z = logsumexp(tmp);
            for i in 0..k {
                gamma[(t, i)] = (tmp[i] - z).exp();
            }
            z
        };
        set_gamma(&mut gamma, tn - 1, &beta_next, &mut tmp);
        for t in (0..tn - 1).rev() {
            for j in 0..k {
                e[j] = ls[(t + 1, j)] + beta_next[j];
            }
            for i in 0..k {
                for j in 0..k {
                    tmp[j] = lat[(i, j)] + e[j];
                }
                beta[i] = logsumexp(&tmp);
            }
            let z = set_gamma(&mut gamma, t, &beta, &mut tmp);
            for i in 0..k {
                for j in 0..k {
                    xi_sum[(i, j)] += (la[(t, i)] + lat[(i, j)] + e[j] - z).exp();
                }
            }
            let shift = beta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            beta.iter_mut().for_each(|b| *b -= shift);
            std::mem::swap(&mut beta, &mut beta_next);
        }
        Ok(Pass { ll, gamma, xi_sum })
    }

    /// Scaled forward pass on `b~ = exp(lb - m_t)`: normalised alphas, `b~`, `c_t` and `ln L`.
    /// `None` when a filter entry that is positive in exact arithmetic falls below the smallest
    /// normal f64: linear space has then lost a path that later evidence may need.
    #[allow(clippy::type_complexity)]
    fn forward_scaled(&self, lb: &Mat, n: usize) -> Result<Option<(Mat, Mat, Vec<f64>, f64)>> {
        let (tn, k) = lb.shape();
        let mut bt = Mat::zeros(tn, k);
        let mut alpha = Mat::zeros(tn, k);
        let mut c = vec![0.0; tn];
        let mut ll = 0.0;
        for t in 0..tn {
            let m = lb.row(t).iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if m == f64::NEG_INFINITY {
                return Err(impossible(n, t));
            }
            for j in 0..k {
                bt[(t, j)] = (lb[(t, j)] - m).exp();
                let (prior, reachable) = if t == 0 {
                    (self.start[j], self.start[j] > 0.0)
                } else {
                    let a = |i: usize| (alpha[(t - 1, i)], self.transition[(i, j)]);
                    (
                        (0..k).map(|i| a(i).0 * a(i).1).sum(),
                        (0..k).any(|i| a(i).0 > 0.0 && a(i).1 > 0.0),
                    )
                };
                alpha[(t, j)] = prior * bt[(t, j)];
                if reachable && lb[(t, j)] > f64::NEG_INFINITY && alpha[(t, j)] < f64::MIN_POSITIVE
                {
                    return Ok(None);
                }
            }
            c[t] = alpha.row(t).sum();
            if c[t] <= 0.0 {
                return Err(impossible(n, t));
            }
            for j in 0..k {
                alpha[(t, j)] /= c[t];
            }
            ll += c[t].ln() + m;
        }
        Ok(Some((alpha, bt, c, ll)))
    }

    /// Falls back to the log pass when linear space cannot hold the exact answer: a lost
    /// filter entry, or a backward value that overflows.
    fn fb_scaled(&self, lb: &Mat, n: usize) -> Result<Pass> {
        let Some((alpha, bt, c, ll)) = self.forward_scaled(lb, n)? else {
            return self.fb_log(lb, n);
        };
        let (tn, k) = lb.shape();
        let a = &self.transition;
        let mut gamma = Mat::zeros(tn, k);
        let mut xi_sum = Mat::zeros(k, k);
        let mut beta_next = vec![1.0; k];
        let mut beta = vec![0.0; k];
        let mut e = vec![0.0; k];
        for i in 0..k {
            gamma[(tn - 1, i)] = alpha[(tn - 1, i)];
        }
        for t in (0..tn - 1).rev() {
            for j in 0..k {
                e[j] = bt[(t + 1, j)] * beta_next[j] / c[t + 1];
            }
            // Structural zeros are skipped, not multiplied: `0 * inf` would be NaN.
            for i in 0..k {
                beta[i] = (0..k)
                    .filter(|&j| a[(i, j)] > 0.0)
                    .map(|j| a[(i, j)] * e[j])
                    .sum();
                if alpha[(t, i)] > 0.0 {
                    gamma[(t, i)] = alpha[(t, i)] * beta[i];
                    for j in 0..k {
                        if a[(i, j)] > 0.0 {
                            xi_sum[(i, j)] += alpha[(t, i)] * a[(i, j)] * e[j];
                        }
                    }
                }
            }
            std::mem::swap(&mut beta, &mut beta_next);
        }
        if !(gamma.iter().chain(xi_sum.iter())).all(|v| v.is_finite()) {
            return self.fb_log(lb, n);
        }
        Ok(Pass { ll, gamma, xi_sum })
    }

    pub(crate) fn forward_backward(&self, lb: &Mat, imp: Implementation, n: usize) -> Result<Pass> {
        match imp {
            Implementation::Log => self.fb_log(lb, n),
            Implementation::Scaled => self.fb_scaled(lb, n),
        }
    }

    pub(crate) fn path_entropy(&self, lb: &Mat, pass: &Pass) -> f64 {
        let (tn, k) = lb.shape();
        let mut e = 0.0;
        for j in 0..k {
            if pass.gamma[(0, j)] > 0.0 {
                e += pass.gamma[(0, j)] * self.start[j].ln();
            }
            for i in 0..k {
                if pass.xi_sum[(i, j)] > 0.0 {
                    e += pass.xi_sum[(i, j)] * self.transition[(i, j)].ln();
                }
            }
            for t in 0..tn {
                if pass.gamma[(t, j)] > 0.0 {
                    e += pass.gamma[(t, j)] * lb[(t, j)];
                }
            }
        }
        pass.ll - e
    }

    pub fn log_likelihood(
        &self,
        seq: &Sequence,
        clamp: Option<&Clamp>,
        imp: Implementation,
    ) -> Result<f64> {
        self.check(seq, clamp, 0)?;
        let lb = self.emission_logs(seq, clamp);
        match imp {
            Implementation::Log => Ok(self.forward_log(&lb, 0)?.2),
            Implementation::Scaled => match self.forward_scaled(&lb, 0)? {
                Some(pass) => Ok(pass.3),
                None => Ok(self.forward_log(&lb, 0)?.2),
            },
        }
    }

    pub fn posterior(
        &self,
        seq: &Sequence,
        clamp: Option<&Clamp>,
        imp: Implementation,
    ) -> Result<Posterior> {
        self.check(seq, clamp, 0)?;
        let lb = self.emission_logs(seq, clamp);
        let pass = self.forward_backward(&lb, imp, 0)?;
        let entropy = self.path_entropy(&lb, &pass);
        Ok(Posterior {
            log_likelihood: pass.ll,
            gamma: pass.gamma,
            xi_sum: pass.xi_sum,
            entropy,
        })
    }

    /// Stream `(seed, [SAMPLE, index])`, `n` draws sequential.
    pub fn sample_posterior(
        &self,
        seq: &Sequence,
        clamp: Option<&Clamp>,
        n: usize,
        seed: u64,
        index: u64,
    ) -> Result<Vec<Vec<u32>>> {
        self.check(seq, clamp, 0)?;
        let bytes = seq
            .t
            .checked_mul(size_of::<u32>())
            .and_then(|b| b.checked_add(size_of::<Vec<u32>>()))
            .and_then(|b| b.checked_mul(n));
        match bytes {
            Some(b) => memory::check(b as u64, None, &format!("{n} posterior paths"))?,
            None => {
                return invalid(format!(
                    "{n} posterior paths of {} steps overflow usize",
                    seq.t
                ));
            }
        }
        let lb = self.emission_logs(seq, clamp);
        let (la, _, _) = self.forward_log(&lb, 0)?;
        let lat = self.log_transition();
        let (tn, k) = la.shape();
        let mut stream = Stream::new(seed, &[tag::SAMPLE, index]);
        let mut w = vec![0.0; k];
        let draw = |stream: &mut Stream, w: &mut [f64]| {
            let m = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            w.iter_mut().for_each(|v| *v = (*v - m).exp());
            stream.categorical(w)
        };
        Ok((0..n)
            .map(|_| {
                let mut path = vec![0u32; tn];
                for t in (0..tn).rev() {
                    for i in 0..k {
                        w[i] = la[(t, i)]
                            + if t + 1 < tn {
                                lat[(i, path[t + 1] as usize)]
                            } else {
                                0.0
                            };
                    }
                    path[t] = draw(&mut stream, &mut w) as u32;
                }
                path
            })
            .collect())
    }
}
