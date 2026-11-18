use varvite_core::{Mat, Result};
use varvite_hmm::{Clamp, Sequence};

use crate::forward_backward::{impossible, log_b};
use crate::model::{Hsmm, Tables};

/// Best segmentation: `(path, ln P(x, s*))`; smallest `u` and lowest state index on ties.
pub(crate) fn viterbi_core(tab: &Tables, lb: &Mat) -> Result<(Vec<u32>, f64)> {
    let (t_len, k) = (lb.nrows(), tab.k);
    if t_len == 0 {
        return Ok((Vec::new(), 0.0));
    }
    let ninf = f64::NEG_INFINITY;
    let mut v = Mat::from_element(t_len + 1, k, ninf);
    let mut w = Mat::from_element(t_len, k, ninf);
    let mut best_u = vec![0usize; (t_len + 1) * k];
    let mut best_i = vec![0usize; t_len * k];
    for j in 0..k {
        w[(0, j)] = tab.log_pi[j];
    }
    for t in 1..=t_len {
        for j in 0..k {
            let mut e = 0.0;
            for u in 1..=t.min(tab.m(j)) {
                e += lb[(t - u, j)];
                if e == ninf {
                    break;
                }
                let lw = if t == t_len {
                    tab.log_dd[j][u - 1]
                } else {
                    tab.log_d[j][u - 1]
                };
                let score = w[(t - u, j)] + lw + e;
                if score > v[(t, j)] {
                    v[(t, j)] = score;
                    best_u[t * k + j] = u;
                }
            }
        }
        if t < t_len {
            for j in 0..k {
                for i in (0..k).filter(|&i| i != j) {
                    let score = v[(t, i)] + tab.log_p[(i, j)];
                    if score > w[(t, j)] {
                        w[(t, j)] = score;
                        best_i[t * k + j] = i;
                    }
                }
            }
        }
    }
    let mut j = 0;
    for i in 1..k {
        if v[(t_len, i)] > v[(t_len, j)] {
            j = i;
        }
    }
    let score = v[(t_len, j)];
    if score == ninf {
        return impossible(None, t_len - 1);
    }
    let mut path = vec![0u32; t_len];
    let mut t = t_len;
    loop {
        let u = best_u[t * k + j];
        for p in &mut path[t - u..t] {
            *p = j as u32;
        }
        t -= u;
        if t == 0 {
            break;
        }
        j = best_i[t * k + j];
    }
    Ok((path, score))
}

impl Hsmm {
    pub fn viterbi(&self, seq: &Sequence, clamp: Option<&Clamp>) -> Result<(Vec<u32>, f64)> {
        self.validate()?;
        let (_, lb) = log_b(&self.emissions, seq, clamp)?;
        viterbi_core(&self.tables(), &lb)
    }
}
