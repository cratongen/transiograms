use rayon::prelude::*;
use varvite_core::special::ln_gamma;
use varvite_core::{Code, Mat, RateMatrix, Result, Stream, TransitionMatrix, invalid, memory};

use crate::fit::{Kind, MarkovChain, stochastic};

/// Hard intervals as positions along the direction.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Hard {
    pub from: Vec<f64>,
    pub to: Vec<f64>,
    pub codes: Vec<Code>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    Stationary,
    Code(Code),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    Steps(usize),
    Beds(usize),
    Length(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Simulated {
    Codes(Vec<Vec<Code>>),
    Beds(Vec<(Vec<f64>, Vec<f64>, Vec<Code>)>),
}

fn row(m: &Mat, i: usize) -> Vec<f64> {
    m.row(i).iter().copied().collect()
}

fn check_hard(hard: &Hard, k: usize, lo: f64, hi: f64) -> Result<()> {
    let n = hard.codes.len();
    if hard.from.len() != n || hard.to.len() != n {
        return invalid(format!(
            "hard from, to and codes must have the same length, got {}, {} and {n}",
            hard.from.len(),
            hard.to.len()
        ));
    }
    for i in 0..n {
        let (a, b, c) = (hard.from[i], hard.to[i], hard.codes[i]);
        if c as usize >= k {
            return invalid(format!(
                "hard code {c} at row {i} is not one of {k} categories"
            ));
        }
        if !(a.is_finite() && b.is_finite() && a <= b && a >= lo - 1e-9 && b <= hi + 1e-9) {
            return invalid(format!(
                "hard interval {i} = [{a}, {b}] must satisfy {lo} <= from <= to <= {hi}"
            ));
        }
        if i > 0 && hard.from[i] < hard.to[i - 1] {
            return invalid(format!(
                "hard intervals must be sorted and non-overlapping: row {i} starts at {a} before row {} ends at {}",
                i - 1,
                hard.to[i - 1]
            ));
        }
    }
    Ok(())
}

/// Sorted `(index, code)` conditions on positions `0..len`.
fn frozen(hard: &Hard, start: Start, k: usize, len: usize) -> Result<Vec<(usize, usize)>> {
    check_hard(hard, k, 0.0, (len - 1) as f64)?;
    let mut out: Vec<(usize, usize)> = Vec::new();
    for i in 0..hard.codes.len() {
        let a = (hard.from[i] - 1e-9).ceil().max(0.0) as usize;
        let b = ((hard.to[i] + 1e-9).floor() as usize).min(len - 1);
        if a > b {
            return invalid(format!(
                "hard interval {i} = [{}, {}] covers no position",
                hard.from[i], hard.to[i]
            ));
        }
        for t in a..=b {
            match out.last() {
                Some(&(u, c)) if u == t && c != hard.codes[i] as usize => {
                    return invalid(format!(
                        "hard intervals give position {t} two codes, {c} and {}",
                        hard.codes[i]
                    ));
                }
                Some(&(u, _)) if u == t => {}
                _ => out.push((t, hard.codes[i] as usize)),
            }
        }
    }
    if let Start::Code(c) = start {
        if c as usize >= k {
            return invalid(format!("start code {c} is not one of {k} categories"));
        }
        match out.first() {
            Some(&(0, d)) if d != c as usize => {
                return invalid(format!(
                    "start code {c} disagrees with the hard code {d} at position 0"
                ));
            }
            Some(&(0, _)) => {}
            _ => out.insert(0, (0, c as usize)),
        }
    }
    Ok(out)
}

/// Columns `P^r e_b` for `r = 0..=n`.
pub(crate) fn bridge_columns(p: &Mat, b: usize, n: usize) -> Vec<Vec<f64>> {
    let k = p.nrows();
    let mut cols = Vec::with_capacity(n + 1);
    let mut v = vec![0.0; k];
    v[b] = 1.0;
    cols.push(v);
    for _ in 0..n {
        let last = cols.last().unwrap();
        let next: Vec<f64> = (0..k)
            .map(|i| (0..k).map(|j| p[(i, j)] * last[j]).sum())
            .collect();
        cols.push(next);
    }
    cols
}

/// `P(x_{t+1} = j | x_t = i, x_n = b) = P_ij (P^(n-t-1))_jb / (P^(n-t))_ib` (unnormalised).
pub(crate) fn bridge_weights(p: &Mat, cols: &[Vec<f64>], i: usize, remaining: usize) -> Vec<f64> {
    (0..p.nrows())
        .map(|j| p[(i, j)] * cols[remaining - 1][j])
        .collect()
}

struct Discrete {
    p: Mat,
    pi: Vec<f64>,
    reversed: Option<Mat>,
    frozen: Vec<(usize, usize)>,
    bridges: Vec<Vec<Vec<f64>>>,
    len: usize,
}

impl Discrete {
    fn new(
        p: &TransitionMatrix,
        pi: Vec<f64>,
        frozen: Vec<(usize, usize)>,
        len: usize,
    ) -> Result<Self> {
        let k = p.k();
        let span: usize = frozen.windows(2).map(|w| w[1].0 - w[0].0 + 1).sum();
        memory::check((span * k * 8) as u64, None, "bridge probabilities")?;
        let mut bridges = Vec::new();
        for w in frozen.windows(2) {
            let ((t0, a), (t1, b)) = (w[0], w[1]);
            let cols = bridge_columns(p.matrix(), b, t1 - t0);
            if cols[t1 - t0][a] <= 0.0 {
                return invalid(format!(
                    "hard data at positions {t0} and {t1} are impossible under the chain"
                ));
            }
            bridges.push(cols);
        }
        let reversed = match frozen.first() {
            Some(&(t, _)) if t > 0 => Some(p.reversed(&pi)?.matrix().clone()),
            _ => None,
        };
        Ok(Self {
            p: p.matrix().clone(),
            pi,
            reversed,
            frozen,
            bridges,
            len,
        })
    }

    fn draw(&self, start: Start, st: &mut Stream) -> Vec<Code> {
        let mut x = vec![0usize; self.len];
        let Some(&(f0, c0)) = self.frozen.first() else {
            x[0] = match start {
                Start::Code(c) => c as usize,
                Start::Stationary => st.categorical(&self.pi),
            };
            for t in 1..self.len {
                x[t] = st.categorical(&row(&self.p, x[t - 1]));
            }
            return x.into_iter().map(|c| c as Code).collect();
        };
        x[f0] = c0;
        if let Some(r) = &self.reversed {
            for t in (0..f0).rev() {
                x[t] = st.categorical(&row(r, x[t + 1]));
            }
        }
        for (w, cols) in self.frozen.windows(2).zip(&self.bridges) {
            let (t0, t1) = (w[0].0, w[1].0);
            for t in t0..t1 {
                x[t + 1] = st.categorical(&bridge_weights(&self.p, cols, x[t], t1 - t));
            }
        }
        let last = self.frozen.last().unwrap().0;
        for t in (last + 1)..self.len {
            x[t] = st.categorical(&row(&self.p, x[t - 1]));
        }
        x.into_iter().map(|c| c as Code).collect()
    }
}

type Beds = Vec<(f64, f64, usize)>;

fn push(beds: &mut Beds, a: f64, b: f64, c: usize) {
    if b <= a {
        return;
    }
    match beds.last_mut() {
        Some(last) if last.2 == c => last.1 = b,
        _ => beds.push((a, b, c)),
    }
}

/// Gillespie from `x` over `[0, duration)`: `(start, end, code)` relative to 0.
fn gillespie(q: &Mat, pi: &Mat, mut x: usize, duration: f64, st: &mut Stream) -> Beds {
    let mut out = Vec::new();
    let mut t = 0.0;
    loop {
        let end = (t + st.exponential() / -q[(x, x)]).min(duration);
        out.push((t, end, x));
        if end >= duration {
            return out;
        }
        t = end;
        x = st.categorical(&row(pi, x));
    }
}

struct Bridge {
    a: usize,
    b: usize,
    from: f64,
    to: f64,
    mu: f64,
    r: Mat,
    tab: f64,
}

impl Bridge {
    fn new(q: &RateMatrix, a: usize, b: usize, from: f64, to: f64) -> Result<Self> {
        let k = q.k();
        let tab = q.transition(to - from).matrix()[(a, b)];
        if tab <= 0.0 {
            return invalid(format!(
                "hard data at positions {from} and {to} are impossible under the chain"
            ));
        }
        let mu = (0..k).map(|j| -q.matrix()[(j, j)]).fold(0.0, f64::max);
        let r = Mat::identity(k, k) + q.matrix() / mu;
        Ok(Self {
            a,
            b,
            from,
            to,
            mu,
            r,
            tab,
        })
    }

    /// Hobolth and Stone (2009) uniformization sampler.
    fn draw(&self, beds: &mut Beds, st: &mut Stream) {
        let k = self.r.nrows();
        let length = self.to - self.from;
        let mt = self.mu * length;
        let u = st.uniform();
        let mut v = vec![0.0; k];
        v[self.a] = 1.0;
        let mut cum = 0.0;
        let mut n = 0usize;
        let guard = (mt + 20.0 * mt.sqrt() + 100.0) as usize;
        loop {
            let lw = -mt + n as f64 * mt.ln() - ln_gamma(n as f64 + 1.0);
            cum += lw.exp() * v[self.b] / self.tab;
            if cum > u || n >= guard {
                break;
            }
            v = (0..k)
                .map(|j| (0..k).map(|i| v[i] * self.r[(i, j)]).sum())
                .collect();
            n += 1;
        }
        let mut times: Vec<f64> = (0..n).map(|_| self.from + st.uniform() * length).collect();
        times.sort_by(f64::total_cmp);
        let cols = bridge_columns(&self.r, self.b, n);
        let mut x = self.a;
        let mut left = self.from;
        for (idx, &t) in times.iter().enumerate() {
            let next = st.categorical(&bridge_weights(&self.r, &cols, x, n - idx));
            if next != x {
                push(beds, left, t, x);
                left = t;
                x = next;
            }
        }
        push(beds, left, self.to, x);
    }
}

impl MarkovChain {
    /// Realization `r` draws from `Stream::new(seed, &[r])`.
    pub fn simulate(
        &self,
        target: Target,
        hard: &Hard,
        start: Start,
        n: usize,
        seed: u64,
    ) -> Result<Simulated> {
        let k = self.k();
        match target {
            Target::Steps(len) | Target::Beds(len) => {
                if len == 0 {
                    return invalid("give at least 1 step or bed to simulate");
                }
                let (p, pi) = match target {
                    Target::Steps(_) => {
                        if !matches!(self.kind, Kind::Fixed { .. }) {
                            return invalid("simulating steps needs a fixed chain");
                        }
                        (self.transition.clone(), self.proportions.clone())
                    }
                    _ => {
                        if !self.transition.is_embedded() {
                            return invalid(
                                "simulating beds needs an embedded transition matrix (zero diagonal)",
                            );
                        }
                        (self.transition.clone(), self.bed_frequencies.clone())
                    }
                };
                memory::check((n * len * 4) as u64, None, "simulated codes")?;
                let sim = Discrete::new(&p, pi, frozen(hard, start, k, len)?, len)?;
                let out = (0..n)
                    .into_par_iter()
                    .map(|r| sim.draw(start, &mut Stream::new(seed, &[r as u64])))
                    .collect();
                Ok(Simulated::Codes(out))
            }
            Target::Length(length) => self.simulate_length(length, hard, start, n, seed),
        }
    }

    fn simulate_length(
        &self,
        length: f64,
        hard: &Hard,
        start: Start,
        n: usize,
        seed: u64,
    ) -> Result<Simulated> {
        let Some(q) = &self.generator else {
            return invalid("simulating a length needs a chain with a generator");
        };
        let k = self.k();
        if !(length > 0.0 && length.is_finite()) {
            return invalid(format!("length must be positive and finite, got {length}"));
        }
        check_hard(hard, k, 0.0, length)?;
        let mut cond: Vec<(f64, f64, usize)> = (0..hard.codes.len())
            .map(|i| {
                (
                    hard.from[i].max(0.0),
                    hard.to[i].min(length),
                    hard.codes[i] as usize,
                )
            })
            .collect();
        if let Start::Code(c) = start {
            if c as usize >= k {
                return invalid(format!("start code {c} is not one of {k} categories"));
            }
            match cond.first() {
                Some(&(f, _, d)) if f <= 0.0 && d != c as usize => {
                    return invalid(format!(
                        "start code {c} disagrees with the hard code {d} at position 0"
                    ));
                }
                Some(&(f, _, _)) if f <= 0.0 => {}
                _ => cond.insert(0, (0.0, 0.0, c as usize)),
            }
        }
        let qm = q.matrix().clone();
        let pi = stochastic(q.embedded().matrix().clone())?.matrix().clone();
        let p = q.stationary()?;
        let reversed = q.reversed(&p);
        let rq = reversed.matrix().clone();
        let rpi = stochastic(reversed.embedded().matrix().clone())?
            .matrix()
            .clone();
        let bridges: Vec<Bridge> = cond
            .windows(2)
            .filter(|w| w[1].0 > w[0].1)
            .map(|w| Bridge::new(q, w[0].2, w[1].2, w[0].1, w[1].0))
            .collect::<Result<_>>()?;
        let out = (0..n)
            .into_par_iter()
            .map(|r| {
                let mut st = Stream::new(seed, &[r as u64]);
                let mut beds: Beds = Vec::new();
                let Some(&(f0, _, c0)) = cond.first() else {
                    let x0 = st.categorical(&p);
                    for (a, b, c) in gillespie(&qm, &pi, x0, length, &mut st) {
                        push(&mut beds, a, b, c);
                    }
                    return unzip(beds);
                };
                if f0 > 0.0 {
                    for (a, b, c) in gillespie(&rq, &rpi, c0, f0, &mut st).into_iter().rev() {
                        push(&mut beds, f0 - b, f0 - a, c);
                    }
                }
                let mut bridge = bridges.iter();
                for (i, &(a, b, c)) in cond.iter().enumerate() {
                    push(&mut beds, a, b, c);
                    if let Some(&(next, _, _)) = cond.get(i + 1)
                        && next > b
                    {
                        bridge.next().unwrap().draw(&mut beds, &mut st);
                    }
                }
                let (_, last, cl) = *cond.last().unwrap();
                if last < length {
                    for (a, b, c) in gillespie(&qm, &pi, cl, length - last, &mut st) {
                        push(&mut beds, last + a, last + b, c);
                    }
                }
                unzip(beds)
            })
            .collect();
        Ok(Simulated::Beds(out))
    }
}

fn unzip(beds: Beds) -> (Vec<f64>, Vec<f64>, Vec<Code>) {
    let mut out = (vec![], vec![], vec![]);
    for (a, b, c) in beds {
        out.0.push(a);
        out.1.push(b);
        out.2.push(c as Code);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_path_probabilities_match_exhaustive_enumeration() {
        let p = Mat::from_row_slice(3, 3, &[0.2, 0.5, 0.3, 0.6, 0.1, 0.3, 0.25, 0.25, 0.5]);
        let (a, b, n) = (0usize, 2usize, 6usize);
        let cols = bridge_columns(&p, b, n);
        let pn = TransitionMatrix::new(p.clone()).unwrap().power(n as u64)[(a, b)];
        assert!((cols[n][a] - pn).abs() < 1e-15);
        let mut total = 0.0;
        for idx in 0..3usize.pow(5) {
            let mut x = vec![a];
            x.extend((0..5).rev().map(|d| idx / 3usize.pow(d) % 3));
            x.push(b);
            let exact: f64 = x.windows(2).map(|w| p[(w[0], w[1])]).product::<f64>() / pn;
            let sampler: f64 = (0..n)
                .map(|t| {
                    let w = bridge_weights(&p, &cols, x[t], n - t);
                    w[x[t + 1]] / w.iter().sum::<f64>()
                })
                .product();
            assert!((sampler - exact).abs() < 1e-12, "path {x:?}");
            total += sampler;
        }
        assert!((total - 1.0).abs() < 1e-12);
    }
}
