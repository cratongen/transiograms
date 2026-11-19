use transiograms_core::{Mat, Result};

use crate::forward_backward::impossible;
use crate::model::{Decoder, Hmm, Implementation};
use crate::sequence::{Clamp, Sequence};

/// Max-product path; on ties the lowest lexicographic optimal path wins.
/// `Err(t)` names the first step where every state scores `-inf`.
pub(crate) fn viterbi_path(
    log_start: &[f64],
    log_a: &Mat,
    scores: &Mat,
) -> std::result::Result<(Vec<u32>, f64), usize> {
    let (tn, k) = scores.shape();
    let mut delta: Vec<f64> = (0..k).map(|j| log_start[j] + scores[(0, j)]).collect();
    let mut next = vec![0.0; k];
    let dead = |v: &[f64]| v.iter().all(|x| *x == f64::NEG_INFINITY);
    if dead(&delta) {
        return Err(0);
    }
    for t in 1..tn {
        for j in 0..k {
            let best = (0..k)
                .map(|i| delta[i] + log_a[(i, j)])
                .fold(f64::NEG_INFINITY, f64::max);
            next[j] = best + scores[(t, j)];
        }
        std::mem::swap(&mut delta, &mut next);
        if dead(&delta) {
            return Err(t);
        }
    }
    let best = delta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // v[t][i]: best log score of x_t..x_T over paths starting in state i at t. Choosing each
    // state front to back as the lowest index that attains the optimum yields the lowest
    // lexicographic optimal path (hmm.md section 10).
    let mut v = Mat::zeros(tn, k);
    for i in 0..k {
        v[(tn - 1, i)] = scores[(tn - 1, i)];
    }
    for t in (0..tn - 1).rev() {
        for i in 0..k {
            let tail = (0..k)
                .map(|j| log_a[(i, j)] + v[(t + 1, j)])
                .fold(f64::NEG_INFINITY, f64::max);
            v[(t, i)] = scores[(t, i)] + tail;
        }
    }
    let first_max = |f: &dyn Fn(usize) -> f64| {
        let (mut b, mut arg) = (f64::NEG_INFINITY, 0);
        for j in 0..k {
            let x = f(j);
            if x > b {
                b = x;
                arg = j;
            }
        }
        arg
    };
    let mut path = vec![0u32; tn];
    path[0] = first_max(&|j| log_start[j] + v[(0, j)]) as u32;
    for t in 1..tn {
        let i = path[t - 1] as usize;
        path[t] = first_max(&|j| log_a[(i, j)] + v[(t, j)]) as u32;
    }
    Ok((path, best))
}

impl Hmm {
    pub fn viterbi(&self, seq: &Sequence, clamp: Option<&Clamp>) -> Result<(Vec<u32>, f64)> {
        self.check(seq, clamp, 0)?;
        let lb = self.emission_logs(seq, clamp);
        viterbi_path(&self.log_start(), &self.log_transition(), &lb).map_err(|t| impossible(0, t))
    }

    pub fn decode(
        &self,
        seq: &Sequence,
        clamp: Option<&Clamp>,
        decoder: Decoder,
    ) -> Result<Vec<u32>> {
        match decoder {
            Decoder::Viterbi => Ok(self.viterbi(seq, clamp)?.0),
            Decoder::Map => {
                let g = self.posterior(seq, clamp, Implementation::Log)?.gamma;
                Ok((0..g.nrows())
                    .map(|t| {
                        let mut arg = 0;
                        for j in 1..g.ncols() {
                            if g[(t, j)] > g[(t, arg)] {
                                arg = j;
                            }
                        }
                        arg as u32
                    })
                    .collect())
            }
            Decoder::PosteriorViterbi => {
                let g = self.posterior(seq, clamp, Implementation::Log)?.gamma;
                let zero_or_dead = |p: f64| if p > 0.0 { 0.0 } else { f64::NEG_INFINITY };
                let start: Vec<f64> = self.start.iter().map(|&p| zero_or_dead(p)).collect();
                let a = self.transition.map(zero_or_dead);
                viterbi_path(&start, &a, &g.map(f64::ln))
                    .map(|r| r.0)
                    .map_err(|t| impossible(0, t))
            }
        }
    }
}
