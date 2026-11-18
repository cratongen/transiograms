use varvite_core::numerics::logsumexp;
use varvite_core::{Error, Mat, Stream};
use varvite_hmm::{
    Categorical, Clamp, CovarianceType, Decoder, Emission, Family, FitOptions, Gaussian,
    GaussianMixture, Hmm, Implementation, Ordering, Params, Poisson, Priors, Sequence, kmeans,
    n_observed,
};

fn simplex(s: &mut Stream, k: usize) -> Vec<f64> {
    let w: Vec<f64> = (0..k).map(|_| s.gamma(1.0) + 0.05).collect();
    let t: f64 = w.iter().sum();
    w.iter().map(|x| x / t).collect()
}

fn transition(s: &mut Stream, k: usize) -> Mat {
    let mut a = Mat::zeros(k, k);
    for i in 0..k {
        let r = simplex(s, k);
        for j in 0..k {
            a[(i, j)] = r[j];
        }
    }
    a
}

fn full_gaussian(s: &mut Stream, d: usize, spread: f64) -> Gaussian {
    let b = Mat::from_fn(d, d, |_, _| 0.5 * s.normal());
    Gaussian {
        mean: (0..d).map(|_| spread * s.normal()).collect(),
        covariance: &b * b.transpose() + Mat::identity(d, d) * 0.5,
        covariance_type: CovarianceType::Full,
        frozen: false,
    }
}

fn gaussian_model(k: usize, d: usize, seed: u64) -> Hmm {
    let mut s = Stream::new(seed, &[]);
    let start = simplex(&mut s, k);
    let a = transition(&mut s, k);
    let e = (0..k)
        .map(|_| Emission::Gaussian(full_gaussian(&mut s, d, 2.0)))
        .collect();
    Hmm::new(start, a, e).unwrap()
}

fn categorical_model(k: usize, m: usize, seed: u64) -> Hmm {
    let mut s = Stream::new(seed, &[]);
    let start = simplex(&mut s, k);
    let a = transition(&mut s, k);
    let e = (0..k)
        .map(|_| {
            Emission::Categorical(Categorical {
                probabilities: simplex(&mut s, m),
                pseudocount: 0.0,
                frozen: false,
            })
        })
        .collect();
    Hmm::new(start, a, e).unwrap()
}

fn diag(mean: &[f64], var: &[f64]) -> Emission {
    Emission::Gaussian(Gaussian {
        mean: mean.to_vec(),
        covariance: Mat::from_diagonal(&varvite_core::nalgebra::DVector::from_vec(var.to_vec())),
        covariance_type: CovarianceType::Diag,
        frozen: false,
    })
}

fn lb_matrix(h: &Hmm, seq: &Sequence, clamp: Option<&Clamp>) -> Vec<Vec<f64>> {
    (0..seq.t)
        .map(|t| {
            (0..h.k())
                .map(|j| {
                    if clamp.is_some_and(|c| !c.allowed[t * h.k() + j]) {
                        f64::NEG_INFINITY
                    } else {
                        h.emissions[j].log_density(&seq.x[t * seq.d..(t + 1) * seq.d])
                    }
                })
                .collect()
        })
        .collect()
}

struct Brute {
    ll: f64,
    path: Vec<u32>,
    best: f64,
    gamma: Mat,
    xi: Mat,
    entropy: f64,
}

/// Exhaustive enumeration of all K^T paths, first step most significant (lexicographic order).
fn brute(h: &Hmm, seq: &Sequence, clamp: Option<&Clamp>) -> Brute {
    let (k, tn) = (h.k(), seq.t);
    let lb = lb_matrix(h, seq, clamp);
    let n = k.pow(tn as u32);
    let decode = |mut idx: usize| -> Vec<usize> {
        let mut p = vec![0; tn];
        for t in (0..tn).rev() {
            p[t] = idx % k;
            idx /= k;
        }
        p
    };
    let lp: Vec<f64> = (0..n)
        .map(|idx| {
            let p = decode(idx);
            let mut v = h.start[p[0]].ln() + lb[0][p[0]];
            for t in 1..tn {
                v += h.transition[(p[t - 1], p[t])].ln() + lb[t][p[t]];
            }
            v
        })
        .collect();
    let ll = logsumexp(&lp);
    let (mut best, mut arg) = (f64::NEG_INFINITY, 0);
    for (i, &v) in lp.iter().enumerate() {
        if v > best {
            best = v;
            arg = i;
        }
    }
    let mut gamma = Mat::zeros(tn, k);
    let mut xi = Mat::zeros(k, k);
    let mut entropy = 0.0;
    for (idx, &v) in lp.iter().enumerate() {
        let pr = (v - ll).exp();
        if pr == 0.0 {
            continue;
        }
        let p = decode(idx);
        entropy -= pr * (v - ll);
        for t in 0..tn {
            gamma[(t, p[t])] += pr;
            if t + 1 < tn {
                xi[(p[t], p[t + 1])] += pr;
            }
        }
    }
    Brute {
        ll,
        path: decode(arg).iter().map(|&s| s as u32).collect(),
        best,
        gamma,
        xi,
        entropy,
    }
}

fn max_abs_diff(a: &Mat, b: &Mat) -> f64 {
    (a - b).amax()
}

fn check_against_brute(h: &Hmm, seq: &Sequence, clamp: Option<&Clamp>) {
    let b = brute(h, seq, clamp);
    let (path, lp) = h.viterbi(seq, clamp).unwrap();
    assert_eq!(path, b.path);
    assert!((lp - b.best).abs() < 1e-12, "viterbi {lp} vs {}", b.best);
    for imp in [Implementation::Log, Implementation::Scaled] {
        let p = h.posterior(seq, clamp, imp).unwrap();
        assert!((p.log_likelihood - b.ll).abs() < 1e-12, "{imp:?} ll");
        let ll = h.log_likelihood(seq, clamp, imp).unwrap();
        assert!((ll - b.ll).abs() < 1e-12, "{imp:?} forward ll");
        assert!(max_abs_diff(&p.gamma, &b.gamma) < 1e-12, "{imp:?} gamma");
        assert!(max_abs_diff(&p.xi_sum, &b.xi) < 1e-12, "{imp:?} xi");
        assert!((p.entropy - b.entropy).abs() < 1e-10, "{imp:?} entropy");
        for t in 0..seq.t {
            assert!((p.gamma.row(t).sum() - 1.0).abs() < 1e-12);
        }
        for i in 0..h.k() {
            let g: f64 = (0..seq.t - 1).map(|t| p.gamma[(t, i)]).sum();
            assert!((p.xi_sum.row(i).sum() - g).abs() < 1e-12);
        }
    }
}

#[test]
fn m2_gaussian_k4_t8_matches_brute_force() {
    for seed in 0..3 {
        let h = gaussian_model(4, 2, seed);
        let (_, mut seq) = h.sample(&[8], seed).remove(0);
        seq.x[2 * 2 + 1] = f64::NAN;
        seq.x[5 * 2] = f64::NAN;
        seq.x[5 * 2 + 1] = f64::NAN;
        check_against_brute(&h, &seq, None);
        let mut allowed = vec![true; 8 * 4];
        allowed[3 * 4] = false;
        allowed[3 * 4 + 2] = false;
        allowed[7 * 4 + 1] = false;
        check_against_brute(&h, &seq, Some(&Clamp { k: 4, allowed }));
    }
}

#[test]
fn m2_categorical_k4_t8_matches_brute_force() {
    for seed in 10..13 {
        let h = categorical_model(4, 3, seed);
        let (_, mut seq) = h.sample(&[8], seed).remove(0);
        seq.x[4] = f64::NAN;
        check_against_brute(&h, &seq, None);
    }
}

#[test]
fn viterbi_ties_take_the_lowest_path() {
    let e = Emission::Categorical(Categorical {
        probabilities: vec![0.5, 0.5],
        pseudocount: 0.0,
        frozen: false,
    });
    let h = Hmm::new(
        vec![0.5, 0.5],
        Mat::from_element(2, 2, 0.5),
        vec![e.clone(), e],
    )
    .unwrap();
    let seq = Sequence {
        t: 6,
        d: 1,
        x: vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    };
    let (path, _) = h.viterbi(&seq, None).unwrap();
    assert_eq!(path, vec![0; 6]);
    assert_eq!(brute(&h, &seq, None).path, path);
}

#[test]
fn forward_and_backward_agree() {
    let h = gaussian_model(3, 2, 7);
    let (_, seq) = h.sample(&[200], 1).remove(0);
    let lb = lb_matrix(&h, &seq, None);
    let k = h.k();
    let mut beta = vec![0.0; k];
    for t in (0..seq.t - 1).rev() {
        beta = (0..k)
            .map(|i| {
                let v: Vec<f64> = (0..k)
                    .map(|j| h.transition[(i, j)].ln() + lb[t + 1][j] + beta[j])
                    .collect();
                logsumexp(&v)
            })
            .collect();
    }
    let back: Vec<f64> = (0..k)
        .map(|j| h.start[j].ln() + lb[0][j] + beta[j])
        .collect();
    let fwd = h.log_likelihood(&seq, None, Implementation::Log).unwrap();
    assert!((logsumexp(&back) - fwd).abs() < 1e-12 * fwd.abs().max(1.0));
}

#[test]
fn scaled_and_log_agree_on_long_sequences_without_underflow() {
    let mut s = Stream::new(3, &[]);
    let e: Vec<Emission> = (0..3)
        .map(|j| {
            let mut g = full_gaussian(&mut s, 4, 3.0);
            g.mean[0] += j as f64 * 4.0;
            Emission::Gaussian(g)
        })
        .collect();
    let h = Hmm::new(vec![0.2, 0.3, 0.5], transition(&mut s, 3), e).unwrap();
    let (_, mut seq) = h.sample(&[20_000], 4).remove(0);
    for c in 0..4 {
        seq.x[1000 * 4 + c] = 400.0;
    }
    let lb = lb_matrix(&h, &seq, None);
    assert!(lb[1000].iter().all(|v| v.exp() == 0.0 && v.is_finite()));
    let a = h.posterior(&seq, None, Implementation::Log).unwrap();
    let b = h.posterior(&seq, None, Implementation::Scaled).unwrap();
    assert!(a.log_likelihood.is_finite() && a.log_likelihood < -1e4);
    assert!((a.log_likelihood - b.log_likelihood).abs() < 1e-10 * a.log_likelihood.abs());
    let gd = max_abs_diff(&a.gamma, &b.gamma);
    assert!(gd < 1e-12, "gamma diff {gd}");
    assert!(max_abs_diff(&a.xi_sum, &b.xi_sum) < 1e-8 * a.xi_sum.amax());
    for t in 0..seq.t {
        assert!((a.gamma.row(t).sum() - 1.0).abs() < 1e-12);
        assert!((b.gamma.row(t).sum() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn missing_row_equals_two_step_transition() {
    let h = gaussian_model(3, 2, 21);
    let (_, mut seq) = h.sample(&[9], 2).remove(0);
    let gap = 4;
    seq.x[gap * 2] = f64::NAN;
    seq.x[gap * 2 + 1] = f64::NAN;
    let lb = lb_matrix(&h, &seq, None);
    let a2 = &h.transition * &h.transition;
    let k = h.k();
    let mut la: Vec<f64> = (0..k).map(|j| h.start[j].ln() + lb[0][j]).collect();
    for t in 1..seq.t {
        if t == gap {
            continue;
        }
        let a = if t == gap + 1 { &a2 } else { &h.transition };
        la = (0..k)
            .map(|j| {
                let v: Vec<f64> = (0..k).map(|i| la[i] + a[(i, j)].ln()).collect();
                lb[t][j] + logsumexp(&v)
            })
            .collect();
    }
    let oracle = logsumexp(&la);
    for imp in [Implementation::Log, Implementation::Scaled] {
        let ll = h.log_likelihood(&seq, None, imp).unwrap();
        assert!((ll - oracle).abs() < 1e-12, "{imp:?}: {ll} vs {oracle}");
    }
}

#[test]
fn partially_missing_full_covariance_equals_quadrature() {
    let mut s = Stream::new(5, &[]);
    let g = full_gaussian(&mut s, 2, 1.0);
    let e = Emission::Gaussian(g.clone());
    let x1 = g.mean[0] + 0.7;
    let sd = g.covariance[(1, 1)].sqrt();
    let (lo, hi, n) = (g.mean[1] - 14.0 * sd, g.mean[1] + 14.0 * sd, 20_000);
    let step = (hi - lo) / n as f64;
    let f = |y: f64| e.log_density(&[x1, y]).exp();
    let mut integral = f(lo) + f(hi);
    for i in 1..n {
        integral += f(lo + i as f64 * step) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    integral *= step / 3.0;
    let marginal = e.log_density(&[x1, f64::NAN]);
    assert!(
        (marginal - integral.ln()).abs() < 1e-8,
        "{marginal} vs {}",
        integral.ln()
    );
    assert_eq!(e.log_density(&[f64::NAN, f64::NAN]), 0.0);
}

#[test]
fn separate_holes_score_as_sum_and_concatenation_differs() {
    let h = gaussian_model(3, 2, 30);
    let s = h.sample(&[50, 70], 3);
    let (a, b) = (&s[0].1, &s[1].1);
    let la = h.log_likelihood(a, None, Implementation::Log).unwrap();
    let lb = h.log_likelihood(b, None, Implementation::Log).unwrap();
    let mut x = a.x.clone();
    x.extend(&b.x);
    let cat = Sequence { t: 120, d: 2, x };
    let lc = h.log_likelihood(&cat, None, Implementation::Log).unwrap();
    assert!((lc - (la + lb)).abs() > 1e-6);
}

#[test]
fn state_permutation_leaves_likelihood_unchanged() {
    let h = gaussian_model(4, 2, 40);
    let (_, seq) = h.sample(&[100], 4).remove(0);
    let perm = [2usize, 0, 3, 1];
    let p = Hmm::new(
        perm.iter().map(|&i| h.start[i]).collect(),
        Mat::from_fn(4, 4, |i, j| h.transition[(perm[i], perm[j])]),
        perm.iter().map(|&i| h.emissions[i].clone()).collect(),
    )
    .unwrap();
    let a = h.log_likelihood(&seq, None, Implementation::Log).unwrap();
    let b = p.log_likelihood(&seq, None, Implementation::Log).unwrap();
    assert!((a - b).abs() < 1e-10 * a.abs());
}

fn assert_monotone(history: &[f64]) {
    assert!(history.len() >= 2);
    for w in history.windows(2) {
        assert!(w[1] >= w[0] - 1e-9 * w[1].abs(), "{} -> {}", w[0], w[1]);
    }
}

fn assert_rows(h: &Hmm) {
    assert!((h.start.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    for i in 0..h.k() {
        assert!((h.transition.row(i).sum() - 1.0).abs() < 1e-12);
    }
}

fn opts(seed: u64) -> FitOptions {
    FitOptions {
        max_iter: 40,
        tol: 0.0,
        n_init: 2,
        seed,
        ..FitOptions::default()
    }
}

fn punch_holes(data: &mut [Sequence], every: usize) {
    for seq in data.iter_mut() {
        for t in (0..seq.t).step_by(every) {
            seq.x[t * seq.d] = f64::NAN;
        }
        if seq.t > 3 {
            for c in 0..seq.d {
                seq.x[3 * seq.d + c] = f64::NAN;
            }
        }
    }
}

#[test]
fn em_is_monotone_for_every_gaussian_covariance_type() {
    let truth = gaussian_model(3, 3, 50);
    let mut data: Vec<Sequence> = truth
        .sample(&[150, 1, 90, 200], 5)
        .into_iter()
        .map(|p| p.1)
        .collect();
    punch_holes(&mut data, 7);
    let mut allowed = vec![true; 150 * 3];
    for t in 10..20 {
        allowed[t * 3] = false;
        allowed[t * 3 + 1] = false;
    }
    let clamps = [
        Clamp { k: 3, allowed },
        Clamp {
            k: 3,
            allowed: vec![true; 3],
        },
        Clamp {
            k: 3,
            allowed: vec![true; 270],
        },
        Clamp {
            k: 3,
            allowed: vec![true; 600],
        },
    ];
    for ct in [
        CovarianceType::Full,
        CovarianceType::Diag,
        CovarianceType::Spherical,
    ] {
        for (clamp, imp) in [
            (None, Implementation::Log),
            (Some(&clamps[..]), Implementation::Scaled),
        ] {
            let fam = Family::Gaussian {
                covariance_type: ct,
            };
            let mut h = truth.clone();
            let mut o = opts(1);
            o.priors = Priors {
                start: 2.0,
                transition: 1.5,
                covariance: 1e-2,
            };
            o.implementation = imp;
            let rep = h.fit(&fam, &data, clamp, &o).unwrap();
            assert_monotone(&rep.history);
            assert_rows(&h);
            for e in &h.emissions {
                e.validate().unwrap();
            }
        }
    }
}

#[test]
fn em_is_monotone_for_categorical_poisson_and_mixture() {
    let truth = categorical_model(3, 4, 60);
    let data: Vec<Sequence> = truth
        .sample(&[300, 200], 6)
        .into_iter()
        .map(|p| p.1)
        .collect();
    let mut h = truth.clone();
    let mut o = opts(2);
    o.priors.start = 3.0;
    o.priors.transition = 2.0;
    let rep = h
        .fit(&Family::Categorical { n_symbols: 4 }, &data, None, &o)
        .unwrap();
    assert_monotone(&rep.history);
    assert_rows(&h);

    let mut h = truth.clone();
    for e in &mut h.emissions {
        if let Emission::Categorical(c) = e {
            c.pseudocount = 0.5;
        }
    }
    o.init_params = Params::parse("st").unwrap();
    let rep = h
        .fit(&Family::Categorical { n_symbols: 4 }, &data, None, &o)
        .unwrap();
    assert_monotone(&rep.history);

    let pois = Hmm::new(
        vec![0.5, 0.5],
        Mat::from_row_slice(2, 2, &[0.9, 0.1, 0.2, 0.8]),
        vec![
            Emission::Poisson(Poisson {
                rate: vec![1.0, 5.0],
                frozen: false,
            }),
            Emission::Poisson(Poisson {
                rate: vec![6.0, 0.5],
                frozen: false,
            }),
        ],
    )
    .unwrap();
    let mut data: Vec<Sequence> = pois
        .sample(&[400, 300], 7)
        .into_iter()
        .map(|p| p.1)
        .collect();
    punch_holes(&mut data, 11);
    let mut h = pois.clone();
    let rep = h.fit(&Family::Poisson, &data, None, &opts(3)).unwrap();
    assert_monotone(&rep.history);
    assert_rows(&h);

    let mut s = Stream::new(8, &[]);
    let mix = |s: &mut Stream, shift: f64, ct: CovarianceType| {
        let mut a = full_gaussian(s, 2, 1.0);
        let mut b = full_gaussian(s, 2, 1.0);
        a.mean[0] += shift;
        b.mean[0] += shift + 2.0;
        if ct == CovarianceType::Diag {
            for g in [&mut a, &mut b] {
                g.covariance = Mat::from_diagonal(&g.covariance.diagonal());
                g.covariance_type = ct;
            }
        }
        Emission::Mixture(GaussianMixture {
            weights: vec![0.4, 0.6],
            components: vec![a, b],
            frozen: false,
        })
    };
    for ct in [CovarianceType::Full, CovarianceType::Diag] {
        let truth = Hmm::new(
            vec![0.5, 0.5],
            Mat::from_row_slice(2, 2, &[0.95, 0.05, 0.1, 0.9]),
            vec![mix(&mut s, -4.0, ct), mix(&mut s, 4.0, ct)],
        )
        .unwrap();
        let mut data: Vec<Sequence> = truth
            .sample(&[500, 300], 9)
            .into_iter()
            .map(|p| p.1)
            .collect();
        punch_holes(&mut data, 13);
        let fam = Family::Mixture {
            n_mix: 2,
            covariance_type: ct,
        };
        let mut h = truth.clone();
        let rep = h.fit(&fam, &data, None, &opts(4)).unwrap();
        assert_monotone(&rep.history);
        assert_rows(&h);
    }
}

#[test]
fn fully_clamped_data_give_the_counting_mle() {
    let truth = categorical_model(3, 4, 70);
    let pieces = truth.sample(&[120, 80, 1], 10);
    let data: Vec<Sequence> = pieces.iter().map(|p| p.1.clone()).collect();
    let clamps: Vec<Clamp> = pieces
        .iter()
        .map(|(s, _)| Clamp {
            k: 3,
            allowed: s
                .iter()
                .flat_map(|&q| (0..3).map(move |j| j == q as usize))
                .collect(),
        })
        .collect();
    let (mut pi, mut a, mut b) = (vec![0.0; 3], Mat::zeros(3, 3), Mat::zeros(3, 4));
    for (s, seq) in &pieces {
        pi[s[0] as usize] += 1.0;
        for t in 0..s.len() {
            b[(s[t] as usize, seq.x[t] as usize)] += 1.0;
            if t + 1 < s.len() {
                a[(s[t] as usize, s[t + 1] as usize)] += 1.0;
            }
        }
    }
    let n: f64 = pi.iter().sum();
    pi.iter_mut().for_each(|p| *p /= n);
    for m in [&mut a, &mut b] {
        for i in 0..3 {
            let r = m.row(i).sum();
            m.row_mut(i).iter_mut().for_each(|v| *v /= r);
        }
    }
    let mut o = FitOptions {
        max_iter: 1,
        tol: 0.0,
        init_params: Params::parse("").unwrap(),
        ..FitOptions::default()
    };
    let fam = Family::Categorical { n_symbols: 4 };
    let mut h = truth.clone();
    h.fit(&fam, &data, Some(&clamps), &o).unwrap();
    let close = |x: &[f64], y: &[f64]| x.iter().zip(y).all(|(p, q)| (p - q).abs() < 1e-12);
    assert!(close(&h.start, &pi));
    assert!(max_abs_diff(&h.transition, &a) < 1e-12);
    for j in 0..3 {
        let Emission::Categorical(c) = &h.emissions[j] else {
            unreachable!()
        };
        let row: Vec<f64> = b.row(j).iter().copied().collect();
        assert!(close(&c.probabilities, &row));
    }
    let first = h.clone();
    o.max_iter = 1;
    h.fit(&fam, &data, Some(&clamps), &o).unwrap();
    assert!(close(&h.start, &first.start));
    assert!(max_abs_diff(&h.transition, &first.transition) < 1e-12);
}

#[test]
fn recovery_from_the_truth_within_three_binomial_sigma() {
    let truth = Hmm::new(
        vec![0.3, 0.3, 0.4],
        Mat::from_row_slice(
            3,
            3,
            &[0.90, 0.07, 0.03, 0.05, 0.85, 0.10, 0.04, 0.06, 0.90],
        ),
        vec![
            diag(&[0.0, 0.0], &[1.0, 0.5]),
            diag(&[3.0, 2.0], &[0.8, 1.0]),
            diag(&[-3.0, 4.0], &[1.2, 0.7]),
        ],
    )
    .unwrap();
    let pieces = truth.sample(&[5000; 20], 11);
    let data: Vec<Sequence> = pieces.iter().map(|p| p.1.clone()).collect();
    let mut visits = [0.0; 3];
    for (s, _) in &pieces {
        for &q in &s[..s.len() - 1] {
            visits[q as usize] += 1.0;
        }
    }
    let mut h = truth.clone();
    let o = FitOptions {
        init_params: Params::parse("").unwrap(),
        ..FitOptions::default()
    };
    let rep = h
        .fit(
            &Family::Gaussian {
                covariance_type: CovarianceType::Diag,
            },
            &data,
            None,
            &o,
        )
        .unwrap();
    assert!(rep.converged);
    for i in 0..3 {
        for j in 0..3 {
            let a = truth.transition[(i, j)];
            let sigma = (a * (1.0 - a) / visits[i]).sqrt();
            let diff = (h.transition[(i, j)] - a).abs();
            assert!(
                diff < 3.0 * sigma,
                "a[{i},{j}] off by {diff}, sigma {sigma}"
            );
        }
    }
}

#[test]
fn recovery_from_kmeans_up_to_permutation() {
    let truth = Hmm::new(
        vec![0.3, 0.3, 0.4],
        Mat::from_row_slice(
            3,
            3,
            &[0.90, 0.07, 0.03, 0.05, 0.85, 0.10, 0.04, 0.06, 0.90],
        ),
        vec![
            diag(&[3.0, 2.0], &[0.8, 1.0]),
            diag(&[-3.0, 4.0], &[1.2, 0.7]),
            diag(&[0.0, 0.0], &[1.0, 0.5]),
        ],
    )
    .unwrap();
    let data: Vec<Sequence> = truth
        .sample(&[2000; 10], 12)
        .into_iter()
        .map(|p| p.1)
        .collect();
    let mut h = truth.clone();
    let o = FitOptions {
        n_init: 3,
        seed: 5,
        ordering: Ordering::Mean,
        ..FitOptions::default()
    };
    let fam = Family::Gaussian {
        covariance_type: CovarianceType::Diag,
    };
    h.fit(&fam, &data, None, &o).unwrap();
    let perm = [1usize, 2, 0];
    for i in 0..3 {
        for j in 0..3 {
            let diff = (h.transition[(i, j)] - truth.transition[(perm[i], perm[j])]).abs();
            assert!(diff < 0.02, "a[{i},{j}] off by {diff}");
        }
        let (Emission::Gaussian(g), Emission::Gaussian(t)) =
            (&h.emissions[i], &truth.emissions[perm[i]])
        else {
            unreachable!()
        };
        for c in 0..2 {
            assert!((g.mean[c] - t.mean[c]).abs() < 0.05);
            assert!((g.covariance[(c, c)] / t.covariance[(c, c)] - 1.0).abs() < 0.05);
        }
    }
}

fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

#[test]
fn results_do_not_depend_on_thread_count() {
    let truth = gaussian_model(3, 2, 80);
    let run = || {
        let data: Vec<Sequence> = truth
            .sample(&[300, 200, 250, 50], 13)
            .into_iter()
            .map(|p| p.1)
            .collect();
        let mut h = truth.clone();
        let o = FitOptions {
            n_init: 3,
            max_iter: 30,
            ..FitOptions::default()
        };
        let rep = h
            .fit(
                &Family::Gaussian {
                    covariance_type: CovarianceType::Full,
                },
                &data,
                None,
                &o,
            )
            .unwrap();
        let draws = h.sample_posterior(&data[0], None, 20, 3, 0).unwrap();
        (h, rep, draws)
    };
    let one = in_pool(1, run);
    let four = in_pool(4, run);
    assert_eq!(one.0, four.0);
    assert_eq!(one.1, four.1);
    assert_eq!(one.2, four.2);
    assert_eq!(one, in_pool(1, run));
}

#[test]
fn sample_posterior_marginals_match_gamma() {
    let h = gaussian_model(3, 1, 90);
    let (_, seq) = h.sample(&[6], 14).remove(0);
    let n = 40_000;
    let draws = h.sample_posterior(&seq, None, n, 1, 0).unwrap();
    let g = h.posterior(&seq, None, Implementation::Log).unwrap().gamma;
    for t in 0..6 {
        for j in 0..3 {
            let f = draws.iter().filter(|p| p[t] as usize == j).count() as f64 / n as f64;
            let p = g[(t, j)];
            assert!((f - p).abs() <= 5.0 * (p * (1.0 - p) / n as f64).sqrt() + 1e-12);
        }
    }
}

#[test]
fn decoders_match_their_definitions() {
    let mut h = gaussian_model(3, 2, 91);
    h.transition[(0, 2)] = 0.0;
    let r = h.transition.row(0).sum();
    h.transition.row_mut(0).iter_mut().for_each(|v| *v /= r);
    let (_, seq) = h.sample(&[6], 15).remove(0);
    let g = h.posterior(&seq, None, Implementation::Log).unwrap().gamma;
    let map = h.decode(&seq, None, Decoder::Map).unwrap();
    for t in 0..6 {
        for j in 0..3 {
            assert!(g[(t, map[t] as usize)] >= g[(t, j)]);
        }
    }
    let pv = h.decode(&seq, None, Decoder::PosteriorViterbi).unwrap();
    let mut best = (f64::NEG_INFINITY, vec![]);
    for idx in 0..3usize.pow(6) {
        let p: Vec<usize> = (0..6).rev().map(|t| idx / 3usize.pow(t) % 3).collect();
        if (1..6).any(|t| h.transition[(p[t - 1], p[t])] == 0.0) {
            continue;
        }
        let v: f64 = (0..6).map(|t| g[(t, p[t])].ln()).sum();
        if v > best.0 {
            best = (v, p.iter().map(|&s| s as u32).collect());
        }
    }
    assert_eq!(pv, best.1);
    assert_eq!(
        h.decode(&seq, None, Decoder::Viterbi).unwrap(),
        h.viterbi(&seq, None).unwrap().0
    );
}

#[test]
fn emission_samples_have_the_right_moments() {
    let n = 100_000;
    let mut s = Stream::new(16, &[]);
    let p = Emission::Poisson(Poisson {
        rate: vec![0.3, 7.5],
        frozen: false,
    });
    let xs: Vec<Vec<f64>> = (0..n).map(|_| p.sample(&mut s)).collect();
    for (c, lam) in [0.3, 7.5].into_iter().enumerate() {
        let m = xs.iter().map(|x| x[c]).sum::<f64>() / n as f64;
        assert!(
            (m - lam).abs() < 5.0 * (lam / n as f64).sqrt(),
            "poisson mean {m}"
        );
    }
    let g = full_gaussian(&mut s, 2, 1.0);
    let e = Emission::Gaussian(g.clone());
    let xs: Vec<Vec<f64>> = (0..n).map(|_| e.sample(&mut s)).collect();
    for a in 0..2 {
        let m = xs.iter().map(|x| x[a]).sum::<f64>() / n as f64;
        assert!((m - g.mean[a]).abs() < 5.0 * (g.covariance[(a, a)] / n as f64).sqrt());
    }
    let cov01 = xs
        .iter()
        .map(|x| (x[0] - g.mean[0]) * (x[1] - g.mean[1]))
        .sum::<f64>()
        / n as f64;
    assert!((cov01 - g.covariance[(0, 1)]).abs() < 0.03);
}

#[test]
fn poisson_density_matches_closed_form() {
    let p = Emission::Poisson(Poisson {
        rate: vec![2.5, 0.0],
        frozen: false,
    });
    let expected = 3.0 * 2.5f64.ln() - 2.5 - 6f64.ln();
    assert!((p.log_density(&[3.0, 0.0]) - expected).abs() < 1e-12);
    let floored = expected + 1e-300f64.ln() - 1e-300;
    assert!((p.log_density(&[3.0, 1.0]) - floored).abs() < 1e-12);
    assert!((p.log_density(&[3.0, f64::NAN]) - expected).abs() < 1e-12);
}

#[test]
fn free_parameters_and_letters() {
    let h = gaussian_model(3, 2, 92);
    assert_eq!(
        h.n_params(Params::parse("ste").unwrap()),
        2 + 6 + 3 * (2 + 3)
    );
    assert_eq!(h.n_params(Params::parse("s").unwrap()), 2);
    assert!(Params::parse("stx").is_err());
    let mut c = categorical_model(2, 5, 93);
    if let Emission::Categorical(e) = &mut c.emissions[1] {
        e.frozen = true;
    }
    assert_eq!(c.n_params(Params::parse("e").unwrap()), 4);
    let json = serde_json::to_string(&h).unwrap();
    assert_eq!(serde_json::from_str::<Hmm>(&json).unwrap(), h);
}

#[test]
fn impossible_observations_name_the_step() {
    let e = |p: Vec<f64>| {
        Emission::Categorical(Categorical {
            probabilities: p,
            pseudocount: 0.0,
            frozen: false,
        })
    };
    let h = Hmm::new(
        vec![1.0, 0.0],
        Mat::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
        vec![e(vec![1.0, 0.0]), e(vec![0.0, 1.0])],
    )
    .unwrap();
    let seq = Sequence {
        t: 4,
        d: 1,
        x: vec![0.0, 0.0, 1.0, 0.0],
    };
    for imp in [Implementation::Log, Implementation::Scaled] {
        let err = h.log_likelihood(&seq, None, imp).unwrap_err();
        assert_eq!(
            err,
            Error::InvalidInput(
                "observation impossible under the model: sequence 0, step 2".into()
            )
        );
    }
    assert!(h.viterbi(&seq, None).is_err());
    let bad = Sequence {
        t: 1,
        d: 1,
        x: vec![2.0],
    };
    assert!(matches!(
        h.log_likelihood(&bad, None, Implementation::Log),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn kmeans_separates_clusters_and_is_reproducible() {
    let mut s = Stream::new(17, &[]);
    let pts = Mat::from_fn(300, 2, |i, _| (i / 100) as f64 * 10.0 + s.normal());
    let a = kmeans(&pts, 3, &mut Stream::new(1, &[]));
    assert_eq!(a, kmeans(&pts, 3, &mut Stream::new(1, &[])));
    for block in 0..3 {
        let first = a[block * 100];
        assert!(
            a[block * 100..(block + 1) * 100]
                .iter()
                .all(|&x| x == first)
        );
    }
}

#[test]
fn initialize_and_n_observed() {
    let truth = gaussian_model(3, 2, 94);
    let mut data: Vec<Sequence> = truth
        .sample(&[100, 50], 18)
        .into_iter()
        .map(|p| p.1)
        .collect();
    data[0].x[0] = f64::NAN;
    data[0].x[1] = f64::NAN;
    assert_eq!(n_observed(&data), 149);
    for fam in [
        Family::Gaussian {
            covariance_type: CovarianceType::Spherical,
        },
        Family::Mixture {
            n_mix: 2,
            covariance_type: CovarianceType::Full,
        },
    ] {
        let h = Hmm::initialize(
            3,
            &fam,
            &data,
            None,
            &Priors::default(),
            &mut Stream::new(0, &[4, 0]),
        )
        .unwrap();
        assert_eq!(h.k(), 3);
    }
    assert!(
        Hmm::initialize(
            3,
            &Family::Categorical { n_symbols: 2 },
            &data,
            None,
            &Priors::default(),
            &mut Stream::new(0, &[])
        )
        .is_err()
    );
}

#[test]
fn em_log_likelihood_never_decreases_without_priors() {
    let truth = gaussian_model(4, 2, 95);
    let data: Vec<Sequence> = truth
        .sample(&[400, 250], 19)
        .into_iter()
        .map(|p| p.1)
        .collect();
    let mut o = opts(6);
    o.priors = Priors {
        start: 1.0,
        transition: 1.0,
        covariance: 0.0,
    };
    let mut h = truth.clone();
    let rep = h
        .fit(
            &Family::Gaussian {
                covariance_type: CovarianceType::Full,
            },
            &data,
            None,
            &o,
        )
        .unwrap();
    assert_eq!(rep.history.len(), rep.n_iter + 1);
    assert_monotone(&rep.history);
    let ll: f64 = data
        .iter()
        .map(|s| h.log_likelihood(s, None, Implementation::Log).unwrap())
        .sum();
    assert!(ll >= *rep.history.last().unwrap() - 1e-9 * ll.abs());
}
