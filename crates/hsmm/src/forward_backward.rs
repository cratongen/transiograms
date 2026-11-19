use transiograms_core::numerics::logsumexp;
use transiograms_core::rng::tag;
use transiograms_core::{Mat, Result, Stream, invalid, memory};
use transiograms_hmm::{Clamp, Emission, Sequence};

use crate::model::{Hsmm, Tables};

#[derive(Clone, Debug, PartialEq)]
pub struct SegmentPosterior {
    pub log_likelihood: f64,
    pub gamma: Mat,
    pub xi_sum: Mat,
    pub start: Vec<f64>,
    pub durations: Vec<Vec<f64>>,
    pub entropy: f64,
}

/// `(t + 1) x K` matrices alive in one pass: two emission tables, F, S, G, H and gamma.
const PASS_MATRICES: usize = 7;

/// `(unclamped, clamped)` emission log densities, `t x K`.
pub(crate) fn log_b(
    emissions: &[Emission],
    seq: &Sequence,
    clamp: Option<&Clamp>,
) -> Result<(Mat, Mat)> {
    let k = emissions.len();
    let (t, d) = (seq.t, seq.d);
    if t.checked_mul(d) != Some(seq.x.len()) {
        return invalid(format!(
            "sequence holds {} values, expected t x d = {t} x {d}",
            seq.x.len()
        ));
    }
    if emissions[0].dim() != d {
        return invalid(format!(
            "sequence has {d} channels, the emissions have {}",
            emissions[0].dim()
        ));
    }
    let bytes = t
        .checked_add(1)
        .and_then(|r| r.checked_mul(k))
        .and_then(|c| c.checked_mul(PASS_MATRICES * size_of::<f64>()));
    match bytes {
        Some(b) => memory::check(b as u64, None, "HSMM pass over the sequence")?,
        None => return invalid(format!("a sequence of {t} steps overflows usize")),
    }
    let lb = Mat::from_fn(t, k, |s, j| {
        emissions[j].log_density(&seq.x[s * d..(s + 1) * d])
    });
    let mut clamped = lb.clone();
    if let Some(c) = clamp {
        if c.k != k || c.allowed.len() != t * k {
            return invalid(format!(
                "clamp must be {t} x {k}, not {} values for {} states",
                c.allowed.len(),
                c.k
            ));
        }
        for s in 0..t {
            for j in 0..k {
                if !c.allowed[s * k + j] {
                    clamped[(s, j)] = f64::NEG_INFINITY;
                }
            }
        }
    }
    Ok((lb, clamped))
}

/// Right censoring: the segment ending at the last step weighs `D_j(u)` instead of `d_j(u)`.
fn log_w(tab: &Tables, j: usize, u: usize, end: usize, t_len: usize) -> f64 {
    if end == t_len {
        tab.log_dd[j][u - 1]
    } else {
        tab.log_d[j][u - 1]
    }
}

pub(crate) struct Forward {
    /// Rows `0..=T`; row 0 unused.
    pub f: Mat,
    /// Rows `0..T`; `S_0 = ln pi`.
    pub s: Mat,
    pub log_likelihood: f64,
    /// First 0-based step no path can explain.
    pub dead: Option<usize>,
}

pub(crate) fn forward(tab: &Tables, lb: &Mat) -> Forward {
    let (t_len, k) = (lb.nrows(), tab.k);
    let mut f = Mat::from_element(t_len + 1, k, f64::NEG_INFINITY);
    let mut s = Mat::from_element(t_len.max(1), k, f64::NEG_INFINITY);
    for j in 0..k {
        s[(0, j)] = tab.log_pi[j];
    }
    let mut dead = None;
    let mut terms = Vec::new();
    for t in 1..=t_len {
        let mut alive = false;
        for j in 0..k {
            terms.clear();
            let mut e = 0.0;
            for u in 1..=t.min(tab.m(j)) {
                e += lb[(t - u, j)];
                if e == f64::NEG_INFINITY {
                    break;
                }
                let prev = s[(t - u, j)];
                if prev == f64::NEG_INFINITY {
                    continue;
                }
                alive = true;
                terms.push(prev + log_w(tab, j, u, t, t_len) + e);
            }
            f[(t, j)] = logsumexp(&terms);
        }
        if !alive && dead.is_none() {
            dead = Some(t - 1);
        }
        if t < t_len {
            for j in 0..k {
                terms.clear();
                terms.extend(
                    (0..k)
                        .filter(|&i| i != j)
                        .map(|i| f[(t, i)] + tab.log_p[(i, j)]),
                );
                s[(t, j)] = logsumexp(&terms);
            }
        }
    }
    let log_likelihood = if t_len == 0 {
        0.0
    } else {
        logsumexp(&f.row(t_len).iter().copied().collect::<Vec<_>>())
    };
    if log_likelihood == f64::NEG_INFINITY && dead.is_none() {
        dead = Some(t_len.saturating_sub(1));
    }
    Forward {
        f,
        s,
        log_likelihood,
        dead,
    }
}

/// `(G, H)`: `G` rows `0..=T`, `H` rows `0..T`.
pub(crate) fn backward(tab: &Tables, lb: &Mat) -> (Mat, Mat) {
    let (t_len, k) = (lb.nrows(), tab.k);
    let mut g = Mat::from_element(t_len + 1, k, f64::NEG_INFINITY);
    let mut h = Mat::from_element(t_len.max(1), k, f64::NEG_INFINITY);
    for j in 0..k {
        g[(t_len, j)] = 0.0;
    }
    let mut terms = Vec::new();
    for t in (0..t_len).rev() {
        for j in 0..k {
            terms.clear();
            let mut e = 0.0;
            for u in 1..=(t_len - t).min(tab.m(j)) {
                e += lb[(t + u - 1, j)];
                if e == f64::NEG_INFINITY {
                    break;
                }
                let next = g[(t + u, j)];
                if next == f64::NEG_INFINITY {
                    continue;
                }
                terms.push(log_w(tab, j, u, t + u, t_len) + e + next);
            }
            h[(t, j)] = logsumexp(&terms);
        }
        if t >= 1 {
            for j in 0..k {
                terms.clear();
                terms.extend(
                    (0..k)
                        .filter(|&i| i != j)
                        .map(|i| tab.log_p[(j, i)] + h[(t, i)]),
                );
                g[(t, j)] = logsumexp(&terms);
            }
        }
    }
    (g, h)
}

fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * y }
}

pub(crate) fn impossible<T>(seq_index: Option<usize>, step: usize) -> Result<T> {
    match seq_index {
        Some(n) => invalid(format!(
            "observation impossible under the model: sequence {n}, step {step}"
        )),
        None => invalid(format!(
            "observation impossible under the model: step {step}"
        )),
    }
}

/// Posterior from clamped `lb`; `lb_free` (unclamped) feeds the entropy's emission term.
pub(crate) fn posterior_core(
    tab: &Tables,
    lb: &Mat,
    lb_free: &Mat,
    seq_index: Option<usize>,
) -> Result<SegmentPosterior> {
    let (t_len, k) = (lb.nrows(), tab.k);
    let fw = forward(tab, lb);
    let lnl = fw.log_likelihood;
    if lnl == f64::NEG_INFINITY {
        return impossible(seq_index, fw.dead.unwrap_or(0));
    }
    let mut durations: Vec<Vec<f64>> = (0..k).map(|j| vec![0.0; tab.m(j)]).collect();
    if t_len == 0 {
        return Ok(SegmentPosterior {
            log_likelihood: 0.0,
            gamma: Mat::zeros(0, k),
            xi_sum: Mat::zeros(k, k),
            start: vec![0.0; k],
            durations,
            entropy: 0.0,
        });
    }
    let (g, h) = backward(tab, lb);
    let start: Vec<f64> = (0..k)
        .map(|j| (tab.log_pi[j] + h[(0, j)] - lnl).exp())
        .collect();
    let mut joint = 0.0;
    for j in 0..k {
        joint += xlogy(start[j], tab.log_pi[j]);
    }
    for t in 1..=t_len {
        for j in 0..k {
            let mut e = 0.0;
            for u in 1..=t.min(tab.m(j)) {
                e += lb[(t - u, j)];
                if e == f64::NEG_INFINITY {
                    break;
                }
                let prev = fw.s[(t - u, j)];
                let lw = log_w(tab, j, u, t, t_len);
                let eta = (prev + lw + e + g[(t, j)] - lnl).exp();
                if eta.is_nan() || eta <= 0.0 {
                    continue;
                }
                joint += eta * lw;
                if t < t_len {
                    durations[j][u - 1] += eta;
                } else {
                    // Guedon's E-step: a censored run of observed length u has length v >= u
                    // with probability d(v) / D(u).
                    let base = tab.log_dd[j][u - 1];
                    for v in u..=tab.m(j) {
                        durations[j][v - 1] += eta * (tab.log_d[j][v - 1] - base).exp();
                    }
                }
            }
        }
    }
    let mut xi_sum = Mat::zeros(k, k);
    for t in 1..t_len {
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    xi_sum[(i, j)] += (fw.f[(t, i)] + tab.log_p[(i, j)] + h[(t, j)] - lnl).exp();
                }
            }
        }
    }
    for i in 0..k {
        for j in 0..k {
            joint += xlogy(xi_sum[(i, j)], tab.log_p[(i, j)]);
        }
    }
    // gamma_t = gamma_{t-1} + Start_t - End_{t-1}: occupancy from segment starts and ends.
    let mut gamma = Mat::zeros(t_len, k);
    let mut running = vec![0.0; k];
    for t in 1..=t_len {
        for j in 0..k {
            let st = (fw.s[(t - 1, j)] + h[(t - 1, j)] - lnl).exp();
            let en = if t >= 2 {
                (fw.f[(t - 1, j)] + g[(t - 1, j)] - lnl).exp()
            } else {
                0.0
            };
            running[j] += st - en;
            gamma[(t - 1, j)] = running[j].clamp(0.0, 1.0);
            if gamma[(t - 1, j)] > 0.0 && lb_free[(t - 1, j)] > f64::NEG_INFINITY {
                joint += gamma[(t - 1, j)] * lb_free[(t - 1, j)];
            }
        }
    }
    Ok(SegmentPosterior {
        log_likelihood: lnl,
        gamma,
        xi_sum,
        start,
        durations,
        entropy: lnl - joint,
    })
}

fn draw_log(weights: &[f64], stream: &mut Stream) -> usize {
    let m = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = weights.iter().map(|x| (x - m).exp()).collect();
    stream.categorical(&w)
}

pub(crate) fn sample_posterior_core(
    tab: &Tables,
    lb: &Mat,
    n: usize,
    stream: &mut Stream,
) -> Result<Vec<Vec<u32>>> {
    let (t_len, k) = (lb.nrows(), tab.k);
    let fw = forward(tab, lb);
    if fw.log_likelihood == f64::NEG_INFINITY {
        return impossible(None, fw.dead.unwrap_or(0));
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut path = vec![0u32; t_len];
        if t_len == 0 {
            out.push(path);
            continue;
        }
        let mut t = t_len;
        let mut j = draw_log(&fw.f.row(t_len).iter().copied().collect::<Vec<_>>(), stream);
        loop {
            let mut weights = Vec::with_capacity(t.min(tab.m(j)));
            let mut e = 0.0;
            for u in 1..=t.min(tab.m(j)) {
                e += lb[(t - u, j)];
                weights.push(fw.s[(t - u, j)] + log_w(tab, j, u, t, t_len) + e);
            }
            let u = draw_log(&weights, stream) + 1;
            for p in &mut path[t - u..t] {
                *p = j as u32;
            }
            t -= u;
            if t == 0 {
                break;
            }
            let prev: Vec<f64> = (0..k).map(|i| fw.f[(t, i)] + tab.log_p[(i, j)]).collect();
            j = draw_log(&prev, stream);
        }
        out.push(path);
    }
    Ok(out)
}

impl Hsmm {
    pub fn log_likelihood(&self, seq: &Sequence, clamp: Option<&Clamp>) -> Result<f64> {
        self.validate()?;
        let lnl = self.log_likelihood_or_neg_inf(seq, clamp)?;
        if lnl == f64::NEG_INFINITY {
            let (_, lb) = log_b(&self.emissions, seq, clamp)?;
            return impossible(None, forward(&self.tables(), &lb).dead.unwrap_or(0));
        }
        Ok(lnl)
    }

    /// As `log_likelihood`, but an impossible sequence scores `-inf` instead of raising.
    pub(crate) fn log_likelihood_or_neg_inf(
        &self,
        seq: &Sequence,
        clamp: Option<&Clamp>,
    ) -> Result<f64> {
        let (_, lb) = log_b(&self.emissions, seq, clamp)?;
        Ok(forward(&self.tables(), &lb).log_likelihood)
    }

    pub fn posterior(&self, seq: &Sequence, clamp: Option<&Clamp>) -> Result<SegmentPosterior> {
        self.validate()?;
        let (free, lb) = log_b(&self.emissions, seq, clamp)?;
        posterior_core(&self.tables(), &lb, &free, None)
    }

    pub fn sample_posterior(
        &self,
        seq: &Sequence,
        clamp: Option<&Clamp>,
        n: usize,
        seed: u64,
        index: u64,
    ) -> Result<Vec<Vec<u32>>> {
        self.validate()?;
        let (_, lb) = log_b(&self.emissions, seq, clamp)?;
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
        let mut stream = Stream::new(seed, &[tag::SAMPLE, index]);
        sample_posterior_core(&self.tables(), &lb, n, &mut stream)
    }
}
