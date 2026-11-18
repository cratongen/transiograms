use varvite_core::{Mat, Result};

use crate::forward_backward::impossible;
use crate::model::{Decoder, Hmm, Implementation};
use crate::sequence::{Clamp, Sequence};

/// Max-product path; predecessors scan upward with strict `>` so the lowest index wins ties.
/// `Err(t)` names the first step where every state scores `-inf`.
pub(crate) fn viterbi_path(
    log_start: &[f64],
    log_a: &Mat,
    scores: &Mat,
) -> std::result::Result<(Vec<u32>, f64), usize> {
    let (tn, k) = scores.shape();
    let mut delta: Vec<f64> = (0..k).map(|j| log_start[j] + scores[(0, j)]).collect();
    let mut next = vec![0.0; k];
    let mut psi = vec![0u32; tn * k];
    let dead = |v: &[f64]| v.iter().all(|x| *x == f64::NEG_INFINITY);
    if dead(&delta) {
        return Err(0);
    }
    for t in 1..tn {
        for j in 0..k {
            let (mut best, mut arg) = (f64::NEG_INFINITY, 0);
            for i in 0..k {
                let v = delta[i] + log_a[(i, j)];
                if v > best {
                    best = v;
                    arg = i;
                }
            }
            next[j] = best + scores[(t, j)];
            psi[t * k + j] = arg as u32;
        }
        std::mem::swap(&mut delta, &mut next);
        if dead(&delta) {
            return Err(t);
        }
    }
    let (mut best, mut arg) = (f64::NEG_INFINITY, 0);
    for (j, &v) in delta.iter().enumerate() {
        if v > best {
            best = v;
            arg = j;
        }
    }
    let mut path = vec![0u32; tn];
    path[tn - 1] = arg as u32;
    for t in (1..tn).rev() {
        path[t - 1] = psi[t * k + path[t] as usize];
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
