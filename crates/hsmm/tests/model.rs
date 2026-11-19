use std::panic::{AssertUnwindSafe, catch_unwind};

use transiograms_core::{Mat, Stream};
use transiograms_hmm::{
    CovarianceType, Emission, Family, FitOptions, Gaussian, Hmm, Implementation, Ordering, Params,
    Priors, Sequence,
};
use transiograms_hsmm::{
    Duration, DurationDist, DurationFamily, Hsmm, ModelKind, SelectionSpec, select_states,
};

fn gaussian(mean: &[f64], sd: f64) -> Emission {
    let d = mean.len();
    Emission::Gaussian(Gaussian {
        mean: mean.to_vec(),
        covariance: Mat::from_diagonal_element(d, d, sd * sd),
        covariance_type: CovarianceType::Diag,
        frozen: false,
    })
}

fn dist(law: Duration) -> DurationDist {
    DurationDist {
        law,
        frozen: false,
        pseudocount: 0.0,
    }
}

fn truth(m: usize) -> Hsmm {
    Hsmm::new(
        vec![0.5, 0.3, 0.2],
        Mat::from_row_slice(3, 3, &[0.0, 0.7, 0.3, 0.5, 0.0, 0.5, 0.6, 0.4, 0.0]),
        vec![
            dist(Duration::NegativeBinomial { r: 4.0, p: 0.4 }),
            dist(Duration::NegativeBinomial { r: 2.0, p: 0.25 }),
            dist(Duration::NegativeBinomial { r: 6.0, p: 0.5 }),
        ],
        vec![
            gaussian(&[0.0, 0.0], 0.6),
            gaussian(&[3.0, 2.0], 0.6),
            gaussian(&[6.0, -1.0], 0.6),
        ],
        vec![m; 3],
    )
    .unwrap()
}

fn sequences(model: &Hsmm, holes: usize, len: usize, seed: u64) -> Vec<Sequence> {
    model
        .sample(&vec![len; holes], seed)
        .into_iter()
        .map(|(_, s)| s)
        .collect()
}

fn diag() -> Family {
    Family::Gaussian {
        covariance_type: CovarianceType::Diag,
    }
}

fn sted() -> Params {
    Params::parse("sted").unwrap()
}

fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

#[test]
fn from_hmm_matches_hmm_likelihood_and_viterbi() {
    let hmm = Hmm::new(
        vec![0.6, 0.3, 0.1],
        Mat::from_row_slice(3, 3, &[0.9, 0.07, 0.03, 0.05, 0.92, 0.03, 0.1, 0.1, 0.8]),
        vec![
            gaussian(&[0.0], 1.0),
            gaussian(&[1.5], 0.8),
            gaussian(&[3.0], 1.2),
        ],
    )
    .unwrap();
    let m = (14.0 / -0.92f64.log10()).ceil() as usize + 1;
    assert!(0.92f64.powi(m as i32) < 1e-14);
    let hsmm = Hsmm::from_hmm(&hmm, m).unwrap();
    for (_, seq) in hmm.sample(&[200, 50, 1], 3) {
        let a = hmm.log_likelihood(&seq, None, Implementation::Log).unwrap();
        let b = hsmm.log_likelihood(&seq, None).unwrap();
        assert!((a - b).abs() <= 1e-10 * a.abs().max(1.0), "{a} {b}");
        assert_eq!(
            hmm.viterbi(&seq, None).unwrap().0,
            hsmm.viterbi(&seq, None).unwrap().0
        );
    }
    let absorbing = Hmm::new(
        vec![1.0, 0.0],
        Mat::from_row_slice(2, 2, &[1.0, 0.0, 0.5, 0.5]),
        vec![gaussian(&[0.0], 1.0), gaussian(&[1.0], 1.0)],
    )
    .unwrap();
    assert!(Hsmm::from_hmm(&absorbing, 10).is_err());
}

#[test]
fn new_rejects_bad_models() {
    let ok = truth(50);
    let mut bad = ok.clone();
    bad.transition[(0, 0)] = 0.1;
    assert!(
        Hsmm::new(
            bad.start,
            bad.transition,
            bad.durations,
            bad.emissions,
            bad.max_duration
        )
        .is_err()
    );
    let bad = ok.clone();
    assert!(
        Hsmm::new(
            bad.start,
            bad.transition,
            bad.durations,
            bad.emissions,
            vec![0, 5, 5]
        )
        .is_err()
    );
    let mut bad = ok.clone();
    bad.durations[0].law = Duration::Empirical {
        pmf: vec![0.0, 0.0, 1.0],
    };
    assert!(
        Hsmm::new(
            bad.start,
            bad.transition,
            bad.durations,
            bad.emissions,
            vec![2, 5, 5]
        )
        .is_err()
    );
    let json = serde_json::to_string(&ok).unwrap();
    assert_eq!(serde_json::from_str::<Hsmm>(&json).unwrap(), ok);
}

#[test]
fn single_state_is_one_censored_segment() {
    let m = Hsmm::new(
        vec![1.0],
        Mat::zeros(1, 1),
        vec![dist(Duration::Poisson { rate: 3.0 })],
        vec![gaussian(&[0.0], 1.0)],
        vec![40],
    )
    .unwrap();
    let (_, seq) = m.sample(&[12], 0).remove(0);
    let lb: f64 = (0..12)
        .map(|t| m.emissions[0].log_density(&seq.x[t..t + 1]))
        .sum();
    let survivor = Duration::Poisson { rate: 3.0 }.log_survivor(40)[11];
    let lnl = m.log_likelihood(&seq, None).unwrap();
    assert!((lnl - (lb + survivor)).abs() < 1e-12);
    let short = Hsmm {
        max_duration: vec![10],
        ..m.clone()
    };
    assert!(short.log_likelihood(&seq, None).is_err());
}

#[test]
fn n_params_counts_free_parameters() {
    let m = truth(30);
    let p = sted();
    assert_eq!(m.n_params(p), 2 + 3 + 3 * 2 + 3 * 4);
    let mut e = m.clone();
    for d in &mut e.durations {
        d.law = Duration::Empirical { pmf: vec![1.0; 30] };
    }
    assert_eq!(e.n_params(p), 2 + 3 + 3 * 29 + 3 * 4);
    assert_eq!(m.n_params(Params::parse("ste").unwrap()), 2 + 3 + 12);
    assert!(m.truncated_mass().iter().any(|v| *v > 1e-6));
    assert!(truth(200).truncated_mass().iter().all(|v| *v < 1e-6));
}

#[test]
fn sampling_is_seeded() {
    let m = truth(60);
    assert_eq!(m.sample(&[100, 80], 9), m.sample(&[100, 80], 9));
    assert_ne!(m.sample(&[100], 9)[0].0, m.sample(&[100], 10)[0].0);
    let draws = m
        .sample_posterior(&m.sample(&[40], 1)[0].1, None, 5, 3, 0)
        .unwrap();
    assert_eq!(
        draws,
        m.sample_posterior(&m.sample(&[40], 1)[0].1, None, 5, 3, 0)
            .unwrap()
    );
}

#[test]
fn em_is_monotone_and_recovers_durations() {
    let model = truth(120);
    let data = sequences(&model, 30, 300, 1);
    let opts = FitOptions {
        params: sted(),
        init_params: sted(),
        n_init: 2,
        ordering: Ordering::Mean,
        tol: 1e-8,
        max_iter: 200,
        ..FitOptions::default()
    };
    let mut fitted = model.clone();
    let report = fitted
        .fit(
            &diag(),
            DurationFamily::NegativeBinomial,
            &data,
            None,
            &opts,
        )
        .unwrap();
    for w in report.history.windows(2) {
        assert!(w[1] >= w[0] - 1e-9 * w[0].abs());
    }
    assert!(report.converged);
    for j in 0..3 {
        let a = model.durations[j].law.mean(120);
        let b = fitted.durations[j].law.mean(120);
        assert!(
            (a - b).abs() < 0.08 * a,
            "state {j}: true mean {a}, fitted {b}"
        );
    }
    for i in 0..3 {
        for j in 0..3 {
            assert!((model.transition[(i, j)] - fitted.transition[(i, j)]).abs() < 0.1);
        }
    }
}

#[test]
fn fit_does_not_depend_on_thread_count() {
    let data = sequences(&truth(80), 8, 150, 4);
    let opts = FitOptions {
        params: sted(),
        init_params: sted(),
        n_init: 3,
        max_iter: 15,
        ..FitOptions::default()
    };
    let run = || {
        let mut m = truth(80);
        let r = m
            .fit(&diag(), DurationFamily::Poisson, &data, None, &opts)
            .unwrap();
        (m, r)
    };
    let one = in_pool(1, run);
    let four = in_pool(4, run);
    assert_eq!(one, four);
}

#[test]
fn bic_selects_three_states_hsmm() {
    let data = sequences(&truth(60), 8, 200, 2);
    let spec = SelectionSpec {
        states: vec![1, 2, 3, 4, 5],
        family: diag(),
        kind: ModelKind::Hsmm {
            duration: DurationFamily::NegativeBinomial,
            max_duration: None,
        },
        fit: FitOptions {
            params: sted(),
            n_init: 2,
            max_iter: 60,
            ..FitOptions::default()
        },
        cv_folds: Some(0),
    };
    let table = select_states(&data, None, &spec).unwrap();
    let best = (0..5)
        .min_by(|&a, &b| table.bic[a].total_cmp(&table.bic[b]))
        .unwrap();
    assert_eq!(table.states[best], 3, "{table:?}");
    for i in 0..5 {
        assert!(
            table.icl[i] >= table.bic[i] - 1e-12 * table.bic[i].abs(),
            "{table:?}"
        );
    }
    assert!(table.cv_log_likelihood.iter().all(|v| v.is_nan()));
}

#[test]
fn cross_validation_does_not_depend_on_thread_count() {
    let data = sequences(&truth(40), 5, 80, 6);
    let spec = SelectionSpec {
        states: vec![1, 2, 3],
        family: diag(),
        kind: ModelKind::Hsmm {
            duration: DurationFamily::Geometric,
            max_duration: None,
        },
        fit: FitOptions {
            params: sted(),
            n_init: 2,
            max_iter: 10,
            ..FitOptions::default()
        },
        cv_folds: Some(2),
    };
    let one = in_pool(1, || select_states(&data, None, &spec).unwrap());
    let four = in_pool(4, || select_states(&data, None, &spec).unwrap());
    assert_eq!(format!("{one:?}"), format!("{four:?}"));
    assert!(one.cv_log_likelihood.iter().all(|v| v.is_finite()));
    let loo = SelectionSpec {
        cv_folds: None,
        ..spec
    };
    let t = select_states(&data, None, &loo).unwrap();
    assert!(t.cv_per_step.iter().all(|v| v.is_finite() && *v < 0.0));
}

/// M3: 20 holes x 300 steps, 3 well-separated Gaussian states, 2 channels, `n_init = 5`,
/// seed 0, K = 1..6, diag covariance: BIC argmin = 3.
#[test]
fn m3_bic_selects_three_states_hmm() {
    let hmm = Hmm::new(
        vec![0.4, 0.35, 0.25],
        Mat::from_row_slice(
            3,
            3,
            &[0.95, 0.03, 0.02, 0.02, 0.95, 0.03, 0.03, 0.02, 0.95],
        ),
        vec![
            gaussian(&[0.0, 0.0], 0.5),
            gaussian(&[3.0, 2.5], 0.5),
            gaussian(&[6.0, 0.0], 0.5),
        ],
    )
    .unwrap();
    let data: Vec<Sequence> = hmm
        .sample(&[300; 20], 0)
        .into_iter()
        .map(|(_, s)| s)
        .collect();
    let spec = SelectionSpec {
        states: (1..=6).collect(),
        family: diag(),
        kind: ModelKind::Hmm,
        fit: FitOptions {
            n_init: 5,
            seed: 0,
            ..FitOptions::default()
        },
        cv_folds: Some(0),
    };
    let table = select_states(&data, None, &spec).unwrap();
    let best = (0..6)
        .min_by(|&a, &b| table.bic[a].total_cmp(&table.bic[b]))
        .unwrap();
    assert_eq!(table.states[best], 3, "{table:?}");
}

#[test]
fn clamped_fit_and_impossible_clamp() {
    let model = truth(50);
    let sampled = model.sample(&[60, 60], 5);
    let data: Vec<Sequence> = sampled.iter().map(|(_, s)| s.clone()).collect();
    let clamps: Vec<transiograms_hmm::Clamp> = sampled
        .iter()
        .map(|(states, s)| transiograms_hmm::Clamp {
            k: 3,
            allowed: (0..s.t * 3)
                .map(|i| i % 3 == states[i / 3] as usize)
                .collect(),
        })
        .collect();
    let (path, _) = model.viterbi(&data[0], Some(&clamps[0])).unwrap();
    assert_eq!(path, sampled[0].0);
    let post = model.posterior(&data[0], Some(&clamps[0])).unwrap();
    assert!(post.entropy.abs() < 1e-9);
    let mut none = clamps[0].clone();
    for j in 0..3 {
        none.allowed[10 * 3 + j] = false;
    }
    let err = model.log_likelihood(&data[0], Some(&none)).unwrap_err();
    assert!(err.to_string().contains("step 10"), "{err}");
}

fn two_state(laws: Duration, m: usize) -> Hsmm {
    Hsmm::new(
        vec![0.5, 0.5],
        Mat::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]),
        vec![dist(laws.clone()), dist(laws)],
        vec![gaussian(&[-0.5], 1.0), gaussian(&[2.0], 1.0)],
        vec![m, m],
    )
    .unwrap()
}

/// `max_duration` far below the true bed thickness keeps truncation active; EM stays monotone.
#[test]
fn em_monotone_under_active_truncation() {
    let mut truth = two_state(Duration::NegativeBinomial { r: 8.0, p: 0.4 }, 80);
    truth.emissions = vec![gaussian(&[0.0], 1.0), gaussian(&[1.5], 1.0)];
    let data = sequences(&truth, 8, 150, 11);
    let mut failures = Vec::new();
    for m in [3usize, 5] {
        for law in [
            Duration::Geometric { p: 0.5 },
            Duration::Poisson { rate: 1.0 },
            Duration::NegativeBinomial { r: 1.0, p: 0.5 },
            Duration::Empirical {
                pmf: vec![1.0 / m as f64; m],
            },
        ] {
            let fam = match law {
                Duration::Geometric { .. } => DurationFamily::Geometric,
                Duration::Poisson { .. } => DurationFamily::Poisson,
                Duration::NegativeBinomial { .. } => DurationFamily::NegativeBinomial,
                Duration::Empirical { .. } => DurationFamily::Empirical,
            };
            let mut model = two_state(law, m);
            let opts = FitOptions {
                params: sted(),
                init_params: Params::parse("").unwrap(),
                n_init: 1,
                tol: 1e-12,
                max_iter: 60,
                ordering: Ordering::None,
                ..FitOptions::default()
            };
            match model.fit(&diag(), fam, &data, None, &opts) {
                Ok(report) => {
                    for w in report.history.windows(2) {
                        if w[1] < w[0] - 1e-12 * w[0].abs() {
                            failures.push(format!("{fam:?} m={m}: {} -> {}", w[0], w[1]));
                        }
                    }
                }
                Err(e) => failures.push(format!("{fam:?} m={m}: {e}")),
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The duration M-step maximises `sum_u N(u) ln d(u)` for `d` truncated on `1..=M`, so a law
/// renormalised from the untruncated closed form cannot beat it.
#[test]
fn duration_m_step_is_truncated_mle() {
    let counts = [1.0, 1.0, 8.0];
    let m = counts.len();
    let score =
        |law: &Duration| -> f64 { law.log_pmf(m).iter().zip(&counts).map(|(l, c)| c * l).sum() };
    let grid = |f: &dyn Fn(f64) -> Duration| -> f64 {
        (1..2000)
            .map(|i| score(&f(i as f64 / 2000.0)))
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let cases: [(Duration, f64); 3] = [
        (
            Duration::Geometric { p: 0.5 },
            grid(&|x| Duration::Geometric { p: x }),
        ),
        (
            Duration::Poisson { rate: 1.0 },
            grid(&|x| Duration::Poisson { rate: 100.0 * x }),
        ),
        (
            Duration::NegativeBinomial { r: 1.0, p: 0.5 },
            (1..200)
                .map(|i| {
                    grid(&|x| Duration::NegativeBinomial {
                        r: i as f64 / 10.0,
                        p: x,
                    })
                })
                .fold(f64::NEG_INFINITY, f64::max),
        ),
    ];
    for (template, best) in cases {
        let fitted = template.fit(&counts, 0.0).unwrap();
        assert!(
            score(&fitted) >= best - 1e-12,
            "{fitted:?} {} < {best}",
            score(&fitted)
        );
    }
    for (template, other) in [
        (
            Duration::Geometric { p: 0.5 },
            Duration::Geometric { p: 0.01 },
        ),
        (
            Duration::Poisson { rate: 1.0 },
            Duration::Poisson { rate: 20.0 },
        ),
    ] {
        let fitted = template.fit(&counts, 0.0).unwrap();
        assert!(score(&fitted) >= score(&other), "{fitted:?} vs {other:?}");
    }
}

fn small_data() -> Vec<Sequence> {
    sequences(&two_state(Duration::Geometric { p: 0.3 }, 30), 3, 40, 1)
}

/// `Ok(true)` when the call returned `Err`, `Ok(false)` when it returned `Ok`, `Err` on a panic.
fn outcome<T>(f: impl FnOnce() -> transiograms_core::Result<T>) -> Result<bool, String> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => Ok(r.is_err()),
        Err(p) => Err(p
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()),
    }
}

fn one_state(m: usize) -> transiograms_core::Result<Hsmm> {
    Hsmm::new(
        vec![1.0],
        Mat::zeros(1, 1),
        vec![dist(Duration::Geometric { p: 0.5 })],
        vec![gaussian(&[0.0], 1.0)],
        vec![m],
    )
}

#[test]
fn initialize_rejects_zero_max_duration() {
    let data = small_data();
    let got = outcome(|| {
        Hsmm::initialize(
            2,
            &diag(),
            DurationFamily::Geometric,
            &data,
            None,
            &Priors::default(),
            &[0, 0],
            &mut Stream::new(0, &[1]),
        )
    });
    assert_eq!(got, Ok(true), "initialize with max_duration 0");
}

fn selection_with(max_duration: usize) -> SelectionSpec {
    SelectionSpec {
        states: vec![2],
        family: diag(),
        kind: ModelKind::Hsmm {
            duration: DurationFamily::Geometric,
            max_duration: Some(max_duration),
        },
        fit: FitOptions {
            max_iter: 2,
            ..FitOptions::default()
        },
        cv_folds: Some(0),
    }
}

#[test]
fn select_states_rejects_zero_and_absurd_max_duration() {
    let data = small_data();
    for m in [0, 1usize << 40, usize::MAX] {
        let spec = selection_with(m);
        let got = outcome(|| select_states(&data, None, &spec));
        assert_eq!(got, Ok(true), "select_states with max_duration Some({m})");
    }
}

#[test]
fn new_rejects_absurd_max_duration_without_panicking() {
    assert_eq!(outcome(|| one_state(usize::MAX)), Ok(true));
    assert_eq!(outcome(|| one_state(usize::MAX / 2)), Ok(true));
}

/// 2^36 steps need 2 TiB of duration tables: rejected by the memory budget before allocating.
#[test]
fn new_rejects_huge_max_duration_before_allocating() {
    let e = one_state(1usize << 36).unwrap_err().to_string();
    assert!(e.contains("duration tables"), "{e}");
}

#[test]
fn entry_points_revalidate_public_fields() {
    let mut m = two_state(Duration::Geometric { p: 0.3 }, 5);
    m.max_duration = vec![1usize << 40, 5];
    let seq = Sequence {
        t: 3,
        d: 1,
        x: vec![0.0, 3.0, 0.0],
    };
    assert_eq!(outcome(|| m.log_likelihood(&seq, None)), Ok(true));
    assert_eq!(outcome(|| m.posterior(&seq, None)), Ok(true));
    assert_eq!(outcome(|| m.viterbi(&seq, None)), Ok(true));
    assert_eq!(
        outcome(|| m.sample_posterior(&seq, None, 2, 0, 0)),
        Ok(true)
    );
    let data = small_data();
    let opts = FitOptions {
        max_iter: 2,
        ..FitOptions::default()
    };
    assert_eq!(
        outcome(|| m.fit(&diag(), DurationFamily::Geometric, &data, None, &opts)),
        Ok(true)
    );
}

#[test]
fn log_likelihood_rejects_overflowing_sequence_shape() {
    let m = two_state(Duration::Geometric { p: 0.3 }, 5);
    for (t, d) in [(usize::MAX, 2), (1usize << 40, 1)] {
        let seq = Sequence {
            t,
            d,
            x: Vec::new(),
        };
        assert_eq!(outcome(|| m.log_likelihood(&seq, None)), Ok(true));
        assert_eq!(outcome(|| m.viterbi(&seq, None)), Ok(true));
        assert_eq!(outcome(|| m.posterior(&seq, None)), Ok(true));
    }
}

#[test]
fn sample_posterior_rejects_absurd_draw_count() {
    let m = two_state(Duration::Geometric { p: 0.3 }, 5);
    let seq = Sequence {
        t: 3,
        d: 1,
        x: vec![0.5, 2.0, 0.5],
    };
    for n in [usize::MAX, 1usize << 50] {
        let got = outcome(|| m.sample_posterior(&seq, None, n, 0, 0));
        assert_eq!(got, Ok(true), "sample_posterior with n = {n}");
    }
    assert_eq!(m.sample_posterior(&seq, None, 4, 0, 0).unwrap().len(), 4);
}

#[test]
fn duration_fit_rejects_bad_pseudocount() {
    for pc in [-1.0, f64::NAN, f64::INFINITY] {
        for law in [
            Duration::Empirical {
                pmf: vec![0.5, 0.5],
            },
            Duration::Geometric { p: 0.5 },
        ] {
            assert!(
                law.fit(&[0.0, 5.0], pc).is_err(),
                "{law:?} pseudocount {pc}"
            );
        }
    }
}

fn score(law: &Duration, counts: &[f64]) -> f64 {
    law.log_pmf(counts.len())
        .iter()
        .zip(counts)
        .map(|(l, c)| if *c == 0.0 { 0.0 } else { c * l })
        .sum()
}

/// Brute-force grid maximum of the truncated weighted log-likelihood; the fit must reach it.
#[test]
fn duration_fits_reach_the_grid_maximum() {
    let mut stream = Stream::new(42, &[7]);
    let mut failures = Vec::new();
    for case in 0..12 {
        let m = 2 + case % 7;
        let counts: Vec<f64> = (0..m)
            .map(|_| {
                let u = stream.categorical(&[1.0, 1.0, 1.0, 1.0]);
                (u as f64) * 1.5
            })
            .collect();
        if counts.iter().sum::<f64>() == 0.0 {
            continue;
        }
        let ps: Vec<f64> = (1..=2000).map(|i| i as f64 / 2000.0).collect();
        let geo_best = ps
            .iter()
            .map(|&p| score(&Duration::Geometric { p }, &counts))
            .fold(f64::NEG_INFINITY, f64::max);
        let rates: Vec<f64> = (0..=2000)
            .map(|i| 10f64.powf(-3.0 + 9.0 * i as f64 / 2000.0))
            .collect();
        let poi_best = rates
            .iter()
            .map(|&rate| score(&Duration::Poisson { rate }, &counts))
            .fold(f64::NEG_INFINITY, f64::max);
        let rs: Vec<f64> = (0..=120)
            .map(|i| 10f64.powf(-3.0 + 7.0 * i as f64 / 120.0))
            .collect();
        let mut nb_best = f64::NEG_INFINITY;
        for &r in &rs {
            for i in 1..=400 {
                let p = i as f64 / 400.0;
                nb_best = nb_best.max(score(&Duration::NegativeBinomial { r, p }, &counts));
            }
        }
        for (template, best) in [
            (Duration::Geometric { p: 0.5 }, geo_best),
            (Duration::Poisson { rate: 1.0 }, poi_best),
            (Duration::NegativeBinomial { r: 1.0, p: 0.5 }, nb_best),
        ] {
            let fitted = template.fit(&counts, 0.0).unwrap();
            let s = score(&fitted, &counts);
            if s < best - 1e-9 * (1.0 + best.abs()) {
                failures.push(format!("{counts:?}: {fitted:?} scores {s} < grid {best}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
