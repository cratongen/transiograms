use serde::{Deserialize, Serialize};
use varvite_core::{
    Code, Direction, EmptyRow, Mat, Projection, RateMatrix, Result, SequenceSet, TransitionMatrix,
    invalid,
    numerics::{components, gth_stationary},
};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Kind {
    Embedded,
    Fixed {
        step: f64,
    },
    /// Only for chains built from rates.
    Continuous,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitOptions {
    pub kind: Kind,
    pub direction: Direction,
    pub gap: f64,
    /// Embedded only.
    pub self_transitions: bool,
    pub pseudocount: f64,
    pub on_empty_row: EmptyRow,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            kind: Kind::Embedded,
            direction: Direction::Up,
            gap: 1e-6,
            self_transitions: false,
            pseudocount: 0.0,
            on_empty_row: EmptyRow::Raise,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarkovChain {
    pub kind: Kind,
    pub direction: Direction,
    /// Embedded: bed transitions; Fixed: step transitions.
    #[serde(with = "varvite_core::serde_mat::option")]
    pub counts: Option<Mat>,
    /// Total thickness `T_j` of each category (Embedded).
    pub exposure: Option<Vec<f64>>,
    /// Pi (Embedded, Continuous) or P(step) (Fixed).
    pub transition: TransitionMatrix,
    pub generator: Option<RateMatrix>,
    pub proportions: Vec<f64>,
    pub mean_lengths: Option<Vec<f64>>,
    pub bed_frequencies: Vec<f64>,
    /// Fixed: how far `logm(P) / step` moved to be a generator.
    pub projection: Option<Projection>,
    /// Embedded fitted with `self_transitions`: the diagonal is observed, not structural.
    #[serde(default)]
    pub self_transitions: bool,
}

/// Code sequences whose consecutive entries are the transitions the options count: bed codes for
/// Embedded (merged unless `self_transitions`), sampled codes for Fixed.
pub(crate) fn sequences(seqs: &SequenceSet, opts: &FitOptions) -> Result<Vec<Vec<Code>>> {
    check_gap(opts.gap)?;
    match opts.kind {
        Kind::Embedded => Ok(seqs
            .beds(opts.direction, opts.gap, !opts.self_transitions)
            .into_iter()
            .map(|beds| beds.iter().map(|b| b.code).collect())
            .collect()),
        Kind::Fixed { step } => seqs.regular(opts.direction, step, opts.gap),
        Kind::Continuous => {
            invalid("kind Continuous is only for chains built from rates; fit Embedded or Fixed")
        }
    }
}

pub(crate) fn tally(sequences: &[Vec<Code>], m: usize) -> Mat {
    let mut c = Mat::zeros(m, m);
    for s in sequences {
        for w in s.windows(2) {
            c[(w[0] as usize, w[1] as usize)] += 1.0;
        }
    }
    c
}

pub(crate) fn embedded_fit(opts: &FitOptions) -> bool {
    matches!(opts.kind, Kind::Embedded) && !opts.self_transitions
}

fn check_gap(gap: f64) -> Result<()> {
    if !(gap >= 0.0 && gap.is_finite()) {
        return invalid(format!("gap must be finite and >= 0, got {gap}"));
    }
    Ok(())
}

fn check_k(k: usize) -> Result<()> {
    if k < 2 {
        return invalid(format!(
            "a Markov chain needs at least 2 categories, got {k}"
        ));
    }
    Ok(())
}

fn check_step(step: f64) -> Result<()> {
    if !(step > 0.0 && step.is_finite()) {
        return invalid(format!("step must be positive and finite, got {step}"));
    }
    Ok(())
}

/// Rows renormalised so they sum to 1 within rounding before validation.
pub(crate) fn stochastic(mut p: Mat) -> Result<TransitionMatrix> {
    for i in 0..p.nrows() {
        p.row_mut(i).iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
        let s: f64 = p.row(i).sum();
        if s > 0.0 {
            p.row_mut(i).iter_mut().for_each(|v| *v /= s);
        }
    }
    TransitionMatrix::new(p)
}

/// Stationary law of `p` when it has one closed class; transient categories get 0. Several
/// closed classes raise as `TransitionMatrix::stationary`.
pub(crate) fn stationary(p: &TransitionMatrix) -> Result<Vec<f64>> {
    let (m, k) = (p.matrix(), p.k());
    let classes = components(m);
    let closed: Vec<&Vec<usize>> = classes
        .iter()
        .filter(|c| {
            c.iter()
                .all(|&i| (0..k).all(|j| m[(i, j)] <= 0.0 || c.contains(&j)))
        })
        .collect();
    let [c] = closed.as_slice() else {
        return p.stationary();
    };
    if c.len() == k {
        return p.stationary();
    }
    let x = gth_stationary(&Mat::from_fn(c.len(), c.len(), |a, b| m[(c[a], c[b])]))?;
    let mut out = vec![0.0; k];
    for (a, &i) in c.iter().enumerate() {
        out[i] = x[a];
    }
    Ok(out)
}

/// `p_ij / (1 - p_ii)` with a zero diagonal; a row with `p_ii = 1` raises.
fn embedded_of(p: &TransitionMatrix) -> Result<TransitionMatrix> {
    let m = p.matrix();
    let k = p.k();
    for i in 0..k {
        if m[(i, i)] >= 1.0 {
            return invalid(format!(
                "category {i} never leaves (p_ii = 1): the chain is reducible"
            ));
        }
    }
    stochastic(Mat::from_fn(k, k, |i, j| {
        if i == j {
            0.0
        } else {
            m[(i, j)] / (1.0 - m[(i, i)])
        }
    }))
}

impl MarkovChain {
    pub fn fit(seqs: &SequenceSet, opts: &FitOptions) -> Result<Self> {
        let m = seqs.n_categories;
        check_k(m)?;
        if !(opts.pseudocount >= 0.0 && opts.pseudocount.is_finite()) {
            return invalid(format!(
                "pseudocount must be finite and >= 0, got {}",
                opts.pseudocount
            ));
        }
        let counts = tally(&sequences(seqs, opts)?, m);
        match opts.kind {
            Kind::Embedded => {
                let transition = TransitionMatrix::from_counts(
                    &counts,
                    opts.pseudocount,
                    !opts.self_transitions,
                    opts.on_empty_row,
                )?;
                let merged = seqs.beds(opts.direction, opts.gap, true);
                let mut exposure = vec![0.0; m];
                for bed in merged.iter().flatten() {
                    exposure[bed.code as usize] += bed.thickness;
                }
                if let Some(j) = exposure.iter().position(|&t| t <= 0.0) {
                    return invalid(format!(
                        "category {j} is absent from the data; give data for every category or fewer categories"
                    ));
                }
                let contacts: Vec<Vec<Code>> = merged
                    .iter()
                    .map(|beds| beds.iter().map(|b| b.code).collect())
                    .collect();
                let n = tally(&contacts, m);
                let a = opts.pseudocount;
                let mut q = Mat::zeros(m, m);
                for j in 0..m {
                    let exits: f64 = n.row(j).sum();
                    if exits == 0.0 && a == 0.0 {
                        match opts.on_empty_row {
                            EmptyRow::Raise => {
                                return invalid(format!(
                                    "category {j} has no transitions; pass on_empty_row=\"uniform\" or a pseudocount"
                                ));
                            }
                            EmptyRow::Uniform => {
                                (0..m)
                                    .filter(|&k| k != j)
                                    .for_each(|k| q[(j, k)] = 1.0 / ((m - 1) as f64 * exposure[j]));
                            }
                        }
                    } else {
                        (0..m)
                            .filter(|&k| k != j)
                            .for_each(|k| q[(j, k)] = (n[(j, k)] + a) / exposure[j]);
                    }
                    q[(j, j)] = -(0..m).filter(|&k| k != j).map(|k| q[(j, k)]).sum::<f64>();
                }
                let generator = RateMatrix::new(q)?;
                Ok(Self {
                    kind: Kind::Embedded,
                    direction: opts.direction,
                    proportions: generator.stationary()?,
                    mean_lengths: Some(generator.mean_lengths()),
                    bed_frequencies: transition.stationary()?,
                    counts: Some(counts),
                    exposure: Some(exposure),
                    transition,
                    generator: Some(generator),
                    projection: None,
                    self_transitions: opts.self_transitions,
                })
            }
            Kind::Fixed { step } => {
                check_step(step)?;
                let transition = TransitionMatrix::from_counts(
                    &counts,
                    opts.pseudocount,
                    false,
                    opts.on_empty_row,
                )?;
                let mut chain = Self::fixed(transition, step, opts.direction)?;
                chain.counts = Some(counts);
                Ok(chain)
            }
            Kind::Continuous => unreachable!("sequences rejects Continuous"),
        }
    }

    fn fixed(p: TransitionMatrix, step: f64, direction: Direction) -> Result<Self> {
        let proportions = stationary(&p)?;
        let bed_frequencies = stationary(&embedded_of(&p)?)?;
        let mean_lengths = (0..p.k())
            .map(|i| step / (1.0 - p.matrix()[(i, i)]))
            .collect();
        let (generator, projection) = match RateMatrix::from_transition(&p, step) {
            Ok((q, pr)) => (Some(q), Some(pr)),
            Err(_) => (None, None),
        };
        Ok(Self {
            kind: Kind::Fixed { step },
            direction,
            counts: None,
            exposure: None,
            transition: p,
            generator,
            proportions,
            mean_lengths: Some(mean_lengths),
            bed_frequencies,
            projection,
            self_transitions: false,
        })
    }

    /// Embedded without mean lengths: no generator, and `proportions` equal the bed frequencies
    /// (every bed counted with the same thickness). Fixed: `mean_lengths` must be `None` (they
    /// follow from `P`). Continuous: needs an embedded `p` and mean lengths, as `from_rates`.
    pub fn from_transition(
        p: TransitionMatrix,
        kind: Kind,
        mean_lengths: Option<Vec<f64>>,
        direction: Direction,
    ) -> Result<Self> {
        check_k(p.k())?;
        match kind {
            Kind::Fixed { step } => {
                check_step(step)?;
                if mean_lengths.is_some() {
                    return invalid(
                        "a fixed chain takes no mean_lengths: they follow from the diagonal of P",
                    );
                }
                Self::fixed(p, step, direction)
            }
            Kind::Embedded | Kind::Continuous => {
                if !p.is_embedded() {
                    return invalid(
                        "an embedded chain needs a transition matrix with a zero diagonal",
                    );
                }
                match mean_lengths {
                    Some(l) => {
                        let q = RateMatrix::from_embedded(&p, &l)?;
                        let mut chain = Self::from_rates(q, direction)?;
                        chain.kind = kind;
                        chain.transition = p;
                        chain.mean_lengths = Some(l);
                        Ok(chain)
                    }
                    None if kind == Kind::Continuous => {
                        invalid("a continuous chain needs mean_lengths")
                    }
                    None => {
                        let nu = p.stationary()?;
                        Ok(Self {
                            kind,
                            direction,
                            counts: None,
                            exposure: None,
                            transition: p,
                            generator: None,
                            proportions: nu.clone(),
                            mean_lengths: None,
                            bed_frequencies: nu,
                            projection: None,
                            self_transitions: false,
                        })
                    }
                }
            }
        }
    }

    /// Kind Continuous.
    pub fn from_rates(q: RateMatrix, direction: Direction) -> Result<Self> {
        check_k(q.k())?;
        let transition = stochastic(q.embedded().matrix().clone())?;
        Ok(Self {
            kind: Kind::Continuous,
            direction,
            counts: None,
            exposure: None,
            proportions: q.stationary()?,
            mean_lengths: Some(q.mean_lengths()),
            bed_frequencies: transition.stationary()?,
            transition,
            generator: Some(q),
            projection: None,
            self_transitions: false,
        })
    }

    /// Embedded given: `RateMatrix::from_partial` with `partial_ij = pi_ij / L_i` (i != q);
    /// symmetric: only i < j of `embedded` is read, the rest from `p_i r_ij = p_j r_ji`.
    /// Embedded None: `RateMatrix::max_entropy(p, L)`.
    /// The background's own mean length is read only when `embedded` is None.
    pub fn from_geology(
        p: &[f64],
        mean_lengths: &[f64],
        embedded: Option<&Mat>,
        background: usize,
        symmetric: bool,
        direction: Direction,
    ) -> Result<Self> {
        let k = p.len();
        check_k(k)?;
        if mean_lengths.len() != k {
            return invalid(format!(
                "give {k} mean lengths, one per proportion, not {}",
                mean_lengths.len()
            ));
        }
        let q = match embedded {
            None => RateMatrix::max_entropy(p, mean_lengths)?,
            Some(pi) => {
                if pi.nrows() != k || pi.ncols() != k {
                    return invalid(format!(
                        "embedded must be {k}x{k}, got {}x{}",
                        pi.nrows(),
                        pi.ncols()
                    ));
                }
                if background >= k {
                    return invalid(format!("background {background} is not a category of {k}"));
                }
                if let Some(i) = p.iter().position(|&v| !(v > 0.0 && v.is_finite())) {
                    return invalid(format!("proportion {i} = {} must be positive", p[i]));
                }
                let mut partial = Mat::zeros(k, k);
                for i in (0..k).filter(|&i| i != background) {
                    let l = mean_lengths[i];
                    if !(l > 0.0 && l.is_finite()) {
                        return invalid(format!(
                            "mean length {i} = {l} must be positive and finite"
                        ));
                    }
                    partial[(i, i)] = -1.0 / l;
                }
                for i in (0..k).filter(|&i| i != background) {
                    for j in (0..k).filter(|&j| j != background && j != i) {
                        let v = pi[(i, j)];
                        if !(0.0..=1.0).contains(&v) {
                            return invalid(format!(
                                "embedded entry ({i}, {j}) = {v} is outside [0, 1]"
                            ));
                        }
                        if !symmetric || i < j {
                            partial[(i, j)] = v / mean_lengths[i];
                        }
                    }
                }
                if symmetric {
                    for i in (0..k).filter(|&i| i != background) {
                        for j in (0..i).filter(|&j| j != background) {
                            partial[(i, j)] = p[j] * partial[(j, i)] / p[i];
                        }
                    }
                }
                RateMatrix::from_partial(&partial, p, background)?
            }
        };
        Self::from_rates(q, direction)
    }

    pub fn k(&self) -> usize {
        self.transition.k()
    }

    /// Fixed chains return `P^n` when `h = n step` (within `1e-9 step`); otherwise the generator's
    /// `exp(Q h)`.
    pub fn transition_at(&self, h: f64) -> Result<TransitionMatrix> {
        if !(h >= 0.0 && h.is_finite()) {
            return invalid(format!("h must be finite and >= 0, got {h}"));
        }
        if let Kind::Fixed { step } = self.kind {
            let n = (h / step).round();
            if (h - n * step).abs() <= 1e-9 * step {
                return stochastic(self.transition.power(n as u64));
            }
        }
        match &self.generator {
            Some(q) => Ok(q.transition(h)),
            None => match self.kind {
                Kind::Fixed { step } => invalid(format!(
                    "this fixed chain has no generator: h must be a multiple of the step {step}, got {h}"
                )),
                _ => invalid("this chain has no generator: build it with mean lengths"),
            },
        }
    }

    /// Kemeny and Snell standard deviation of the proportions after `n_steps`: of the
    /// proportions for Fixed, of the bed frequencies (steps are beds) otherwise.
    pub fn proportion_std(&self, n_steps: usize) -> Result<Vec<f64>> {
        if n_steps == 0 {
            return invalid("n_steps must be at least 1");
        }
        let pi = stationary(&self.transition)?;
        let z = varvite_core::numerics::fundamental_matrix(self.transition.matrix(), &pi)?;
        Ok((0..self.k())
            .map(|k| {
                (pi[k] * (2.0 * z[(k, k)] - 1.0 - pi[k]) / n_steps as f64)
                    .max(0.0)
                    .sqrt()
            })
            .collect())
    }
}
