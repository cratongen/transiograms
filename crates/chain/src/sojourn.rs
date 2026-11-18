use rayon::prelude::*;
use varvite_core::rng::tag;
use varvite_core::{Code, Direction, Result, SequenceSet, Stream, invalid};

use crate::tests::TestResult;

#[derive(Clone, Debug, PartialEq)]
pub struct Sojourn {
    pub n_beds: Vec<usize>,
    pub n_open: Vec<usize>,
    pub total: Vec<f64>,
    pub mean_length: Vec<f64>,
    pub thickness: Vec<Vec<f64>>,
    pub open: Vec<Vec<bool>>,
}

/// Merged beds per category. `mean_length = total / exits` (NaN with no exit), the MLE of an
/// exponential thickness with the open-ended beds censored. For sampled logs a run of `s`
/// samples spans between `(s - 1) step` and `(s + 1) step`, so short beds carry a
/// discretisation bias.
pub fn sojourn(seqs: &SequenceSet, direction: Direction, gap: f64) -> Sojourn {
    let m = seqs.n_categories;
    let mut s = Sojourn {
        n_beds: vec![0; m],
        n_open: vec![0; m],
        total: vec![0.0; m],
        mean_length: vec![f64::NAN; m],
        thickness: vec![vec![]; m],
        open: vec![vec![]; m],
    };
    let mut exits = vec![0usize; m];
    for bed in seqs.beds(direction, gap, true).iter().flatten() {
        let c = bed.code as usize;
        let open = bed.open_start || bed.open_end;
        s.n_beds[c] += 1;
        s.n_open[c] += open as usize;
        s.total[c] += bed.thickness;
        exits[c] += !bed.open_end as usize;
        s.thickness[c].push(bed.thickness);
        s.open[c].push(open);
    }
    for c in 0..m {
        if exits[c] > 0 {
            s.mean_length[c] = s.total[c] / exits[c] as f64;
        }
    }
    s
}

fn ks_exponential(x: &mut [f64]) -> f64 {
    x.sort_by(f64::total_cmp);
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let f = 1.0 - (-v / mean).exp();
            ((i + 1) as f64 / n - f).max(f - i as f64 / n)
        })
        .fold(0.0, f64::max)
}

impl Sojourn {
    /// Kaplan-Meier, open beds censored: distinct event thicknesses and the survival just after
    /// each. An unknown code gives empty vectors.
    pub fn survival(&self, code: Code) -> (Vec<f64>, Vec<f64>) {
        let c = code as usize;
        if c >= self.thickness.len() {
            return (vec![], vec![]);
        }
        let mut beds: Vec<(f64, bool)> = self.thickness[c]
            .iter()
            .copied()
            .zip(self.open[c].iter().copied())
            .collect();
        beds.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut times, mut surv) = (vec![], vec![]);
        let mut s = 1.0;
        let mut i = 0;
        while i < beds.len() {
            let t = beds[i].0;
            let at_risk = beds.len() - i;
            let mut events = 0;
            while i < beds.len() && beds[i].0 == t {
                events += !beds[i].1 as usize;
                i += 1;
            }
            if events > 0 {
                s *= 1.0 - events as f64 / at_risk as f64;
                times.push(t);
                surv.push(s);
            }
        }
        (times, surv)
    }

    /// Kolmogorov-Smirnov between the closed beds and the exponential with their mean, p value
    /// by parametric bootstrap (replicate `b` draws from `[BOOTSTRAP, b]`).
    pub fn exponential_test(
        &self,
        code: Code,
        n_bootstrap: usize,
        seed: u64,
    ) -> Result<TestResult> {
        let c = code as usize;
        if c >= self.thickness.len() {
            return invalid(format!(
                "code {code} is not one of {} categories",
                self.thickness.len()
            ));
        }
        let mut closed: Vec<f64> = self.thickness[c]
            .iter()
            .zip(&self.open[c])
            .filter(|(_, o)| !**o)
            .map(|(t, _)| *t)
            .collect();
        if closed.len() < 2 {
            return invalid(format!(
                "category {code} has {} closed beds; the exponential test needs at least 2",
                closed.len()
            ));
        }
        let n = closed.len();
        let mean = closed.iter().sum::<f64>() / n as f64;
        let d = ks_exponential(&mut closed);
        let exceed = (0..n_bootstrap)
            .into_par_iter()
            .map(|b| {
                let mut st = Stream::new(seed, &[tag::BOOTSTRAP, b as u64]);
                let mut x: Vec<f64> = (0..n).map(|_| mean * st.exponential()).collect();
                (ks_exponential(&mut x) >= d) as usize
            })
            .collect::<Vec<_>>()
            .iter()
            .sum::<usize>();
        Ok(TestResult {
            name: "exponential".into(),
            method: "ks".into(),
            statistic: d,
            df: 0,
            p_value: (1 + exceed) as f64 / (1 + n_bootstrap) as f64,
            expected: None,
            small_expected: 0.0,
        })
    }
}
