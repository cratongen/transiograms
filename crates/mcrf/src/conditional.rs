use transiograms_core::{Code, Mat};
use transiograms_transiogram::TransitionModel;

/// Eq. 14 into `out` from `(T(h_g), c_g)` pairs, each `T` a column-major K x K slice; false
/// when every candidate has zero likelihood.
pub(crate) fn eq14<'a>(
    proportions: &[f64],
    terms: impl Iterator<Item = (&'a [f64], Code)> + Clone,
    out: &mut [f64],
) -> bool {
    let n = out.len();
    for (k, s) in out.iter_mut().enumerate() {
        *s = proportions[k].ln()
            + terms
                .clone()
                .map(|(t, c)| t[c as usize * n + k].ln())
                .sum::<f64>();
    }
    normalise_log(out)
}

/// `exp(s - logsumexp(s))` in place; false when every `s_k = -inf`.
fn normalise_log(s: &mut [f64]) -> bool {
    let max = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY || max.is_nan() {
        return false;
    }
    for v in s.iter_mut() {
        *v = (*v - max).exp();
    }
    let total: f64 = s.iter().sum();
    for v in s.iter_mut() {
        *v /= total;
    }
    true
}

/// Zero-likelihood policy: drop the last neighbour (lowest closeness, ties to the last found)
/// and retry; with none left the result is `p`.
pub(crate) fn resolve<'a>(
    proportions: &[f64],
    found: &[(u32, Code)],
    transition: impl Fn(u32) -> &'a [f64] + Copy,
    out: &mut [f64],
) {
    for m in (0..=found.len()).rev() {
        if eq14(
            proportions,
            found[..m].iter().map(|&(t, c)| (transition(t), c)),
            out,
        ) {
            return;
        }
    }
    unreachable!("proportions are positive, so the empty neighbourhood always succeeds");
}

fn check_codes(model: &TransitionModel, neighbours: &[([f64; 3], Code)]) {
    let k = model.k();
    assert!(
        neighbours.iter().all(|&(_, c)| (c as usize) < k),
        "neighbour codes must lie in 0..{k}"
    );
}

/// Eq. 14: `p_k prod_g t_{k, c_g}(h_g)` normalised, `h_g = x_g - x_0`, in log space.
/// None when every candidate has zero likelihood.
pub fn conditional(model: &TransitionModel, neighbours: &[([f64; 3], Code)]) -> Option<Vec<f64>> {
    check_codes(model, neighbours);
    let mats: Vec<Mat> = neighbours
        .iter()
        .map(|&(h, _)| model.transition(h))
        .collect();
    let mut out = vec![0.0; model.k()];
    eq14(
        &model.proportions,
        mats.iter()
            .map(|m| m.as_slice())
            .zip(neighbours.iter().map(|&(_, c)| c)),
        &mut out,
    )
    .then_some(out)
}

/// Li and Zhang Eq. 13 with `first` the index of the neighbour the chain comes from; test oracle.
pub fn conditional_eq13(
    model: &TransitionModel,
    neighbours: &[([f64; 3], Code)],
    first: usize,
) -> Option<Vec<f64>> {
    check_codes(model, neighbours);
    assert!(first < neighbours.len(), "first must index a neighbour");
    let (h1, c1) = neighbours[first];
    let from = model.transition([-h1[0], -h1[1], -h1[2]]);
    let mats: Vec<Option<Mat>> = neighbours
        .iter()
        .enumerate()
        .map(|(g, &(h, _))| (g != first).then(|| model.transition(h)))
        .collect();
    let mut s: Vec<f64> = (0..model.k())
        .map(|k| {
            from[(c1 as usize, k)].ln()
                + mats
                    .iter()
                    .zip(neighbours)
                    .filter_map(|(t, &(_, c))| t.as_ref().map(|t| t[(k, c as usize)].ln()))
                    .sum::<f64>()
        })
        .collect();
    normalise_log(&mut s).then_some(s)
}
