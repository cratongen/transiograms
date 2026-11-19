use std::sync::mpsc;
use std::time::Duration;

use transiograms_core::synthetic::{drillholes, layered_field, rate_holes};
use transiograms_core::{Direction, Error, Grid, MISSING, Mat, RateMatrix, SequenceSet, Stream};
use transiograms_transiogram::{
    Entry, Experimental, Model1D, PairFilter, PairSpec, Structure, TransitionModel, downhole,
    experimental, on_grid,
};

const P: [f64; 3] = [0.5, 0.3, 0.2];

fn carle() -> RateMatrix {
    let partial = Mat::from_row_slice(3, 3, &[-0.25, 0.15, 0.0, 0.35, -0.5, 0.0, 0.0, 0.0, 0.0]);
    RateMatrix::from_partial(&partial, &P, 2).unwrap()
}

fn model3d(rotation: [f64; 3]) -> TransitionModel {
    let embedded = Mat::from_row_slice(3, 3, &[0.0, 0.6, 0.4, 0.7, 0.0, 0.3, 0.0, 0.0, 0.0]);
    TransitionModel::from_geology(
        &P,
        [&[40.0, 20.0, 30.0], &[25.0, 15.0, 20.0], &[4.0, 2.0, 2.0]],
        [None, None, Some(&embedded)],
        2,
        rotation,
    )
    .unwrap()
}

fn entrywise() -> Model1D {
    let e = |structure, range| {
        Some(Entry {
            structure,
            range,
            nugget: 0.0,
        })
    };
    let entries = vec![
        None,
        e(Structure::Exponential, 6.0),
        e(Structure::Spherical, 8.0),
        e(Structure::Gaussian, 5.0),
        None,
        e(Structure::Exponential, 4.0),
        e(Structure::Spherical, 7.0),
        e(Structure::Exponential, 3.0),
        None,
    ];
    Model1D::entrywise(P.to_vec(), entries, vec![0, 1, 2], 100.0).unwrap()
}

fn max_diff(a: &Mat, b: &Mat) -> f64 {
    (a - b).iter().fold(0.0, |m: f64, v| m.max(v.abs()))
}

/// Spec section 7, M4: 200 lags from 0 to 20 max L plus 10^4 max L.
fn assert_valid(t: impl Fn(f64) -> Mat, p: &[f64], max_l: f64, what: &str) {
    let k = p.len();
    assert_eq!(t(0.0), Mat::identity(k, k), "{what}: T(0)");
    let lags = (0..200)
        .map(|i| 20.0 * max_l * i as f64 / 199.0)
        .chain([1e4 * max_l]);
    for h in lags {
        let m = t(h);
        for i in 0..k {
            let sum: f64 = m.row(i).sum();
            assert!(
                (sum - 1.0).abs() <= 1e-12,
                "{what}: row {i} at h={h} sums {sum}"
            );
            for j in 0..k {
                assert!(
                    m[(i, j)] >= 0.0,
                    "{what}: ({i},{j}) at h={h} = {}",
                    m[(i, j)]
                );
            }
        }
    }
    let far = t(1e4 * max_l);
    for i in 0..k {
        for j in 0..k {
            assert!(
                (far[(i, j)] - p[j]).abs() <= 1e-10,
                "{what}: sill ({i},{j}) = {} vs {}",
                far[(i, j)],
                p[j]
            );
        }
    }
}

fn directions() -> Vec<[f64; 3]> {
    let mut s = Stream::new(7, &[1]);
    let mut out = vec![
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.6, 0.8, 0.0],
    ];
    for _ in 0..40 {
        let v = [s.normal(), s.normal(), s.normal()];
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        out.push(v.map(|x| x / n));
    }
    out
}

#[test]
fn carle_golden_rates() {
    let q = carle();
    let golden = Mat::from_row_slice(
        3,
        3,
        &[-0.25, 0.15, 0.10, 0.35, -0.5, 0.15, 0.1, 0.375, -0.475],
    );
    assert!(max_diff(q.matrix(), &golden) < 1e-12);
    let m = model3d([0.0; 3]);
    assert!(max_diff(m.rates[2].matrix(), &golden) < 1e-12);
    assert!(m.rates[0].is_reversible(&P, 1e-10));
    assert!(m.rates[1].is_reversible(&P, 1e-10));
}

#[test]
fn m4_lag_grid_validity() {
    let markov = Model1D::markov(carle()).unwrap();
    for (a, b) in markov.proportions().iter().zip(P) {
        assert!((a - b).abs() < 1e-12);
    }
    assert_valid(|h| markov.transition(h), &P, 4.0, "markov");
    assert_valid(|h| markov.transition(-h), &P, 4.0, "markov reversed");
    let ew = entrywise();
    assert_valid(|h| ew.transition(h), &P, 8.0, "entrywise");
    for rotation in [[0.0; 3], [30.0, 20.0, 10.0]] {
        let m = model3d(rotation);
        for u in directions() {
            let max_l = m.mean_lengths(u).iter().fold(0.0, |a: f64, &v| a.max(v));
            assert_valid(|h| m.transition(u.map(|x| x * h)), &P, max_l, "3d");
        }
    }
}

#[test]
fn reversal_and_determinant() {
    let markov = Model1D::markov(carle()).unwrap();
    let m = model3d([30.0, 20.0, 10.0]);
    for h in [0.1, 1.0, 3.7, 12.0] {
        let (f, b) = (markov.transition(h), markov.transition(-h));
        let det = carle().matrix().trace() * h;
        assert!((f.determinant() - det.exp()).abs() < 1e-12);
        for i in 0..3 {
            for j in 0..3 {
                assert!((P[i] * f[(i, j)] - P[j] * b[(j, i)]).abs() < 1e-12);
            }
        }
        for u in directions() {
            let lag = u.map(|x| x * h);
            let (f, b) = (m.transition(lag), m.transition(lag.map(|x| -x)));
            for i in 0..3 {
                for j in 0..3 {
                    assert!((P[i] * f[(i, j)] - P[j] * b[(j, i)]).abs() < 1e-12);
                }
            }
            let trace = m.rates_along(lag).unwrap().matrix().trace();
            assert!((f.determinant() - (h * trace).exp()).abs() < 1e-12);
            assert!((m.closeness(lag) - f.determinant().powf(0.5)).abs() < 1e-12);
        }
    }
}

#[test]
fn reduces_to_directional_models_on_principal_axes() {
    let m = model3d([0.0; 3]);
    for v in 0..3 {
        let mut e = [0.0; 3];
        e[v] = 2.5;
        assert_eq!(m.rates_along(e).unwrap(), m.rates[v]);
        let along = Model1D::markov(m.rates[v].clone()).unwrap();
        assert!(max_diff(&m.transition(e), m.rates[v].transition(2.5).matrix()) < 1e-12);
        assert!(max_diff(&m.transition(e), &along.transition(2.5)) == 0.0);
        e[v] = -2.5;
        assert_eq!(m.rates_along(e).unwrap(), m.rates[v].reversed(&P));
        assert!(max_diff(&m.transition(e), &along.transition(-2.5)) < 1e-12);
    }
    let r = m.rates_along([0.0, 0.0, 2.0]).unwrap();
    assert!(max_diff(r.matrix(), &carle().matrix().clone()) < 1e-12);
    let rotated = model3d([30.0, 20.0, 10.0]);
    for v in 0..3 {
        let axis: [f64; 3] = std::array::from_fn(|a| {
            let mut x = [0.0; 3];
            x[a] = 1.0;
            rotated.to_principal(x)[v]
        });
        let r = rotated.rates_along(axis).unwrap();
        assert!(max_diff(r.matrix(), rotated.rates[v].matrix()) < 1e-12);
    }
}

#[test]
fn mean_lengths_on_the_ellipsoid() {
    let m = model3d([30.0, 20.0, 10.0]);
    let lv: Vec<Vec<f64>> = m.rates.iter().map(|r| r.mean_lengths()).collect();
    for u in directions() {
        let hp = m.to_principal(u);
        let l = m.mean_lengths(u);
        for i in 0..2 {
            let rhs: f64 = (0..3).map(|v| (hp[v].abs() / lv[v][i]).powi(2)).sum();
            assert!(
                ((1.0 / l[i].powi(2)) - rhs).abs() <= 1e-12 * rhs,
                "{u:?} {i}"
            );
        }
    }
    assert!(m.mean_lengths([0.0; 3]).iter().all(|v| v.is_nan()));
}

#[test]
fn rotating_model_and_lag_together() {
    let rotation = [30.0, 20.0, 10.0];
    let rotated = model3d(rotation);
    let plain = model3d([0.0; 3]);
    for u in directions() {
        let h = u.map(|x| 7.0 * x);
        let hp = rotated.to_principal(h);
        let n = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((n(hp) - n(h)).abs() < 1e-12);
        assert!(max_diff(&rotated.transition(h), &plain.transition(hp)) < 1e-12);
    }
    let east = model3d([90.0, 0.0, 0.0]);
    let p = east.to_principal([1.0, 0.0, 0.0]);
    assert!(p[0].abs() < 1e-15 && (p[1] - 1.0).abs() < 1e-15 && p[2] == 0.0);
    let dipping = model3d([0.0, 90.0, 0.0]);
    let p = dipping.to_principal([0.0, 1.0, 0.0]);
    assert!((p[2] - 1.0).abs() < 1e-15, "{p:?}");
}

#[test]
fn downhole_exact_overlaps() {
    let ids: Vec<String> = vec!["a".into(); 4];
    let from = [0.0, 2.0, 3.0, 5.0];
    let to = [2.0, 3.0, 5.0, 6.0];
    let seqs = SequenceSet::from_intervals(&ids, &from, &to, &[0, 1, 0, 1], 2).unwrap();
    let e = downhole(&seqs, Direction::Down, &[0.5, 2.5], 0.0).unwrap();
    let want = [[3.0, 1.0, 0.5, 1.0], [1.5, 1.0, 0.5, 0.5]];
    for (l, w) in want.iter().enumerate() {
        let got = [
            e.counts[l][(0, 0)],
            e.counts[l][(0, 1)],
            e.counts[l][(1, 0)],
            e.counts[l][(1, 1)],
        ];
        for (g, w) in got.iter().zip(w) {
            assert!((g - w).abs() < 1e-12, "lag {l}: {got:?}");
        }
        let total: f64 = e.tail_totals[l].iter().sum();
        assert!((total - (6.0 - e.lags[l])).abs() < 1e-12);
        assert!((e.counts[l].sum() - total).abs() < 1e-12);
    }
    assert!((e.probabilities[0][(1, 0)] - 1.0 / 3.0).abs() < 1e-12);
    let up = downhole(&seqs, Direction::Up, &[0.5], 0.0).unwrap();
    let t = &up.probabilities[0];
    assert!((t[(0, 0)] - 6.0 / 7.0).abs() < 1e-12 && (t[(1, 1)] - 0.5).abs() < 1e-12);
    let split =
        SequenceSet::from_intervals(&ids, &[0.0, 2.0, 3.5, 5.0], &to, &[0, 1, 0, 1], 2).unwrap();
    let gapped = downhole(&split, Direction::Down, &[0.5], 0.1).unwrap();
    let total: f64 = gapped.tail_totals[0].iter().sum();
    assert!((total - (2.5 + 2.0)).abs() < 1e-12, "two runs of 3 and 2.5");
}

#[test]
fn simulated_chain_matches_matrix_exponential() {
    let r = carle();
    let grid = Grid::new([0.0; 3], [1.0, 1.0, 0.5], [1, 1, 200_000]).unwrap();
    let field = layered_field(&grid, &r, 3).unwrap();
    let e = on_grid(&field, &grid, None, [0, 0, 1], 12, 3).unwrap();
    let iv = rate_holes(&r, &[50_000.0; 2], Direction::Up, 5).unwrap();
    let seqs = SequenceSet::from_intervals(&iv.holes, &iv.from, &iv.to, &iv.codes, 3).unwrap();
    let lags: Vec<f64> = (1..=12).map(|l| 0.5 * l as f64).collect();
    let d = downhole(&seqs, Direction::Up, &lags, 0.0).unwrap();
    for (name, exp) in [("grid", &e), ("downhole", &d)] {
        let mut worst: f64 = 0.0;
        for (l, &h) in exp.lags.iter().enumerate() {
            worst = worst.max(max_diff(&exp.probabilities[l], r.transition(h).matrix()));
        }
        assert!(worst < 0.02, "{name}: max |t_hat - exp(Rh)| = {worst}");
    }
}

fn random_field(grid: &Grid, seed: u64) -> Vec<u32> {
    let mut s = Stream::new(seed, &[2]);
    (0..grid.n_cells())
        .map(|_| {
            if s.uniform() < 0.05 {
                MISSING
            } else {
                s.categorical(&P) as u32
            }
        })
        .collect()
}

#[test]
fn points_on_vertical_holes_equal_grid_estimator() {
    let grid = Grid::new([100.0, 200.0, -50.0], [1.0, 1.0, 0.5], [4, 3, 60]).unwrap();
    let field = random_field(&grid, 11);
    let columns = [[0, 0], [1, 0], [3, 2]];
    let dh = drillholes(&grid, &field, &columns).unwrap();
    let holes: Vec<u32> = dh.holes.iter().map(|h| h.parse().unwrap()).collect();
    let mask: Vec<bool> = (0..grid.n_cells())
        .map(|c| {
            let [i, j, _] = grid.ijk(c);
            columns.contains(&[i, j])
        })
        .collect();
    let g = on_grid(&field, &grid, Some(&mask), [0, 0, 1], 6, 3).unwrap();
    let mut spec = PairSpec {
        direction: Some([0.0, -90.0]),
        lag: 0.5,
        n_lags: 6,
        lag_tolerance: 0.25,
        angle_tolerance: 22.5,
        bandwidth: f64::INFINITY,
        vertical_bandwidth: f64::INFINITY,
        directed: true,
        pairs: PairFilter::SameHole,
    };
    let p = experimental(&dh.xyz, &dh.codes, Some(&holes), 3, &spec).unwrap();
    assert_eq!(p.counts, g.counts);
    assert_eq!(p.tail_totals, g.tail_totals);
    for (a, b) in p.mean_lags.iter().zip(&g.mean_lags) {
        assert!((a - b).abs() < 1e-12);
    }
    spec.pairs = PairFilter::All;
    let all = experimental(&dh.xyz, &dh.codes, Some(&holes), 3, &spec).unwrap();
    spec.pairs = PairFilter::DifferentHole;
    let diff = experimental(&dh.xyz, &dh.codes, Some(&holes), 3, &spec).unwrap();
    for l in 0..6 {
        assert_eq!(all.counts[l], &p.counts[l] + &diff.counts[l]);
    }
    spec.pairs = PairFilter::All;
    spec.directed = false;
    let undirected = experimental(&dh.xyz, &dh.codes, None, 3, &spec).unwrap();
    assert!(!undirected.directed);
    for c in &undirected.counts {
        assert_eq!(c, &c.transpose());
    }
    spec.direction = None;
    let omni = experimental(&dh.xyz, &dh.codes, None, 3, &spec).unwrap();
    for (l, c) in omni.counts.iter().enumerate() {
        assert_eq!(c, &c.transpose());
        let rows: f64 = omni.probabilities[l]
            .row_iter()
            .filter(|r| r[0].is_finite())
            .map(|r| (r.sum() - 1.0).abs())
            .fold(0.0, f64::max);
        assert!(rows < 1e-12);
    }
    let pairs = brute_force_omni(&dh.xyz, &dh.codes, 0.5, 0.25, 6);
    assert_eq!(omni.counts, pairs);
}

fn brute_force_omni(xyz: &[[f64; 3]], codes: &[u32], lag: f64, tol: f64, n: usize) -> Vec<Mat> {
    let mut out = vec![Mat::zeros(3, 3); n];
    for a in 0..xyz.len() {
        for b in 0..xyz.len() {
            if a == b || codes[a] == MISSING || codes[b] == MISSING {
                continue;
            }
            let d = (0..3)
                .map(|i| (xyz[b][i] - xyz[a][i]).powi(2))
                .sum::<f64>()
                .sqrt();
            for l in 1..=n {
                if (d - l as f64 * lag).abs() <= tol {
                    out[l - 1][(codes[a] as usize, codes[b] as usize)] += 1.0;
                }
            }
        }
    }
    out
}

fn regular_logs(r: &RateMatrix, n: usize, length: f64, seed: u64) -> SequenceSet {
    let iv = rate_holes(r, &vec![length; n], Direction::Down, seed).unwrap();
    let seqs = SequenceSet::from_intervals(&iv.holes, &iv.from, &iv.to, &iv.codes, 3).unwrap();
    SequenceSet::from_regular(&seqs.regular(Direction::Down, 1.0, 0.0).unwrap(), 3).unwrap()
}

#[test]
fn fit_markov_recovers_rates() {
    let r = carle();
    let logs = regular_logs(&r, 10, 100_000.0, 21);
    let lags: Vec<f64> = (1..=6).map(f64::from).collect();
    let exp = downhole(&logs, Direction::Down, &lags, 0.0).unwrap();
    let fit = Model1D::fit_markov(&exp, None, 2, None).unwrap();
    let Model1D::Markov { rates, .. } = &fit else {
        panic!()
    };
    for i in 0..3 {
        for j in 0..3 {
            let (a, b) = (rates.matrix()[(i, j)], r.matrix()[(i, j)]);
            assert!(((a - b) / b).abs() < 0.05, "({i},{j}): {a} vs {b}");
        }
    }
}

fn exact_experimental(model: &Model1D, lags: &[f64]) -> Experimental {
    let t: Vec<Mat> = lags.iter().map(|&h| model.transition(h)).collect();
    Experimental {
        lags: lags.to_vec(),
        mean_lags: lags.to_vec(),
        counts: t.clone(),
        probabilities: t,
        tail_totals: vec![vec![1.0; 3]; lags.len()],
        directed: true,
    }
}

#[test]
fn fits_recover_exact_models() {
    let lags: Vec<f64> = (1..=10).map(|l| 0.7 * l as f64).collect();
    let truth = Model1D::markov(carle()).unwrap();
    let fit = Model1D::fit_markov(&exact_experimental(&truth, &lags), Some(&P), 2, None).unwrap();
    let (Model1D::Markov { rates: a, .. }, Model1D::Markov { rates: b, .. }) = (&fit, &truth)
    else {
        panic!()
    };
    assert!(max_diff(a.matrix(), b.matrix()) < 1e-5, "{a:?}");
    let truth = entrywise();
    let lags: Vec<f64> = (1..=20).map(f64::from).collect();
    let structures = [
        Structure::Exponential,
        Structure::Exponential,
        Structure::Spherical,
        Structure::Gaussian,
        Structure::Exponential,
        Structure::Exponential,
        Structure::Spherical,
        Structure::Exponential,
        Structure::Exponential,
    ];
    let fit = Model1D::fit_entrywise(
        &exact_experimental(&truth, &lags),
        &P,
        &structures,
        &[0, 1, 2],
    )
    .unwrap();
    let (Model1D::Entrywise { entries: a, .. }, Model1D::Entrywise { entries: b, .. }) =
        (&fit, &truth)
    else {
        panic!()
    };
    for (x, y) in a.iter().zip(b) {
        match (x, y) {
            (Some(x), Some(y)) => assert!((x.range - y.range).abs() < 1e-6 * y.range),
            (None, None) => {}
            _ => panic!("complement mismatch"),
        }
    }
}

#[test]
fn negative_entrywise_raises_naming_entry() {
    let e = |range| {
        Some(Entry {
            structure: Structure::Exponential,
            range,
            nugget: 0.0,
        })
    };
    let entries = vec![
        e(100.0),
        None,
        e(1.0),
        e(2.0),
        None,
        e(2.0),
        e(2.0),
        e(2.0),
        None,
    ];
    let err = Model1D::entrywise(P.to_vec(), entries, vec![1, 1, 2], 10.0).unwrap_err();
    let Error::InvalidInput(msg) = err else {
        panic!()
    };
    assert!(msg.contains("(0, 1)") && msg.contains("h = "), "{msg}");
}

#[test]
fn table_matches_brute_force() {
    let m = model3d([20.0, 0.0, 0.0]);
    let grid = Grid::new([0.0; 3], [10.0, 10.0, 1.0], [15, 12, 9]).unwrap();
    let limit = 0.05;
    let t = m.table(&grid, limit, None).unwrap();
    assert!(!t.offsets.contains(&[0, 0, 0]));
    let mut brute = Vec::new();
    for dz in -8i64..=8 {
        for dy in -11i64..=11 {
            for dx in -14i64..=14 {
                let d = [dx, dy, dz];
                if d != [0; 3] && m.closeness(grid.lag(d)) >= limit {
                    brute.push(d);
                }
            }
        }
    }
    let mut got = t.offsets.clone();
    got.sort();
    brute.sort();
    assert_eq!(got, brute);
    for w in 0..t.offsets.len() - 1 {
        let (a, b) = (t.offsets[w], t.offsets[w + 1]);
        let key = |d: [i64; 3]| [d[2], d[1], d[0]];
        assert!(
            t.closeness[w] > t.closeness[w + 1]
                || (t.closeness[w] == t.closeness[w + 1] && key(a) < key(b))
        );
    }
    for (i, &d) in t.offsets.iter().enumerate() {
        assert_eq!(t.get(d), Some(i));
        assert!(t.closeness[i] >= limit);
        let neg = t.get(d.map(|v| -v)).unwrap();
        assert_eq!(t.closeness[neg], t.closeness[i]);
        assert!(max_diff(&t.transition[i], &m.transition(grid.lag(d))) < 1e-12);
    }
    assert_eq!(t.get([100, 0, 0]), None);
    let small = m.table(&grid, limit, Some(64)).unwrap_err().to_string();
    assert!(small.contains("transition table"), "{small}");
}

#[test]
fn invalid_inputs_raise() {
    let bad = Mat::from_row_slice(3, 3, &[-1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, -1.0]);
    let r = RateMatrix::new(bad).unwrap();
    let e = TransitionModel::new(P.to_vec(), [r.clone(), carle(), carle()], [0.0; 3], 2);
    assert!(e.unwrap_err().to_string().contains("axis x'"));
    let spec = PairSpec {
        direction: None,
        lag: 1.0,
        n_lags: 2,
        lag_tolerance: 0.5,
        angle_tolerance: 22.5,
        bandwidth: f64::INFINITY,
        vertical_bandwidth: f64::INFINITY,
        directed: true,
        pairs: PairFilter::SameHole,
    };
    assert!(experimental(&[[0.0; 3]], &[0], None, 1, &spec).is_err());
    assert!(experimental(&[[0.0; 3]], &[5], Some(&[0]), 1, &spec).is_err());
    let g = Grid::new([0.0; 3], [1.0; 3], [2, 1, 1]).unwrap();
    assert!(on_grid(&[0, 1], &g, None, [0; 3], 1, 2).is_err());
    assert!(model3d([0.0; 3]).rates_along([0.0; 3]).is_err());
}

#[test]
fn json_round_trips() {
    let m = model3d([10.0, 5.0, 0.0]);
    let back: TransitionModel = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
    assert_eq!(back, m);
    let ew = entrywise();
    let back: Model1D = serde_json::from_str(&serde_json::to_string(&ew).unwrap()).unwrap();
    assert_eq!(back, ew);
    let seqs = SequenceSet::from_regular(&[vec![0, 1, 1, 0, 2]], 3).unwrap();
    let e = downhole(&seqs, Direction::Down, &[1.0, 2.0], 0.0).unwrap();
    let text = serde_json::to_string(&e).unwrap();
    let back: Experimental = serde_json::from_str(&text).unwrap();
    assert_eq!(back.counts, e.counts);
    assert!(e.probabilities[1].row(2).iter().all(|v| v.is_nan()));
    assert_eq!(serde_json::to_string(&back).unwrap(), text);
}

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

#[test]
fn results_do_not_depend_on_thread_count() {
    let grid = Grid::new([0.0; 3], [1.0, 1.0, 0.5], [12, 10, 30]).unwrap();
    let field = random_field(&grid, 4);
    let xyz: Vec<[f64; 3]> = (0..grid.n_cells()).map(|c| grid.centroid(c)).collect();
    let spec = PairSpec {
        direction: Some([30.0, 10.0]),
        lag: 1.0,
        n_lags: 5,
        lag_tolerance: 0.5,
        angle_tolerance: 30.0,
        bandwidth: 2.0,
        vertical_bandwidth: 1.0,
        directed: true,
        pairs: PairFilter::All,
    };
    let logs = regular_logs(&carle(), 6, 5_000.0, 2);
    let lags = [1.0, 2.0, 3.0];
    let m = model3d([20.0, 10.0, 5.0]);
    let run = || {
        let p = experimental(&xyz, &field, None, 3, &spec).unwrap();
        let g = on_grid(&field, &grid, None, [1, 1, 0], 4, 3).unwrap();
        let d = downhole(&logs, Direction::Up, &lags, 0.0).unwrap();
        let f = Model1D::fit_markov(&d, None, 2, None).unwrap();
        let t = m.table(&grid, 0.05, None).unwrap();
        let bits = |e: &Experimental| {
            e.counts
                .iter()
                .flat_map(|c| c.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                .chain(e.mean_lags.iter().map(|v| v.to_bits()))
                .collect::<Vec<u64>>()
        };
        (
            bits(&p),
            bits(&g),
            bits(&d),
            serde_json::to_string(&f).unwrap(),
            t,
        )
    };
    let one = pool(1).install(run);
    let four = pool(4).install(run);
    assert!(one.0.iter().any(|&b| b != 0));
    assert_eq!(one.0, four.0);
    assert_eq!(one.1, four.1);
    assert_eq!(one.2, four.2);
    assert_eq!(one.3, four.3);
    assert_eq!(one.4, four.4);
}

#[test]
fn markov_transition_at_infinite_lag_returns() {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let m = Model1D::markov(carle()).unwrap();
        let r = std::panic::catch_unwind(|| m.transition(f64::INFINITY));
        let _ = tx.send(r.map_err(|e| {
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default()
        }));
    });
    let got = rx
        .recv_timeout(Duration::from_secs(30))
        .expect("Model1D::transition(inf) neither returned nor panicked within 30 s");
    let t = got.unwrap_or_else(|msg| panic!("Model1D::transition(inf) panicked: {msg}"));
    for i in 0..3 {
        for j in 0..3 {
            assert!((t[(i, j)] - P[j]).abs() < 1e-10, "T(inf) = {t}");
        }
    }
}

fn gslib_accepts(s: [f64; 3], dir: [f64; 2], atol: f64, bw: f64, vbw: f64) -> bool {
    let (a, d) = (dir[0].to_radians(), dir[1].to_radians());
    let u = [a.sin() * d.cos(), a.cos() * d.cos(), -d.sin()];
    let len = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    let along = s[0] * u[0] + s[1] * u[1] + s[2] * u[2];
    if len == 0.0 || along <= 0.0 {
        return false;
    }
    let angle = (along / len).clamp(-1.0, 1.0).acos().to_degrees();
    if angle > atol + 1e-9 {
        return false;
    }
    let horiz_perp = [a.cos(), -a.sin(), 0.0];
    let vert_perp = [a.sin() * d.sin(), a.cos() * d.sin(), d.cos()];
    let dot = |v: [f64; 3]| (s[0] * v[0] + s[1] * v[1] + s[2] * v[2]).abs();
    dot(horiz_perp) <= bw && dot(vert_perp) <= vbw
}

#[test]
fn directional_point_estimator_matches_brute_force() {
    let mut s = Stream::new(3, &[9]);
    let n = 400;
    let xyz: Vec<[f64; 3]> = (0..n)
        .map(|_| [s.uniform() * 30.0, s.uniform() * 30.0, s.uniform() * 10.0])
        .collect();
    let codes: Vec<u32> = (0..n).map(|_| s.categorical(&P) as u32).collect();
    for dir in [[0.0, 0.0], [37.0, 20.0], [120.0, -60.0], [250.0, 5.0]] {
        for directed in [true, false] {
            let spec = PairSpec {
                direction: Some(dir),
                lag: 2.0,
                n_lags: 5,
                lag_tolerance: 1.3,
                angle_tolerance: 25.0,
                bandwidth: 3.0,
                vertical_bandwidth: 1.5,
                directed,
                pairs: PairFilter::All,
            };
            let got = experimental(&xyz, &codes, None, 3, &spec).unwrap();
            let mut want = vec![Mat::zeros(3, 3); 5];
            for t in 0..n {
                for h in 0..n {
                    if t == h {
                        continue;
                    }
                    let sv: [f64; 3] = std::array::from_fn(|a| xyz[h][a] - xyz[t][a]);
                    let ok = gslib_accepts(sv, dir, 25.0, 3.0, 1.5)
                        || (!directed && gslib_accepts(sv.map(|v| -v), dir, 25.0, 3.0, 1.5));
                    if !ok {
                        continue;
                    }
                    let d = (sv[0] * sv[0] + sv[1] * sv[1] + sv[2] * sv[2]).sqrt();
                    for l in 1..=5 {
                        if (d - l as f64 * 2.0).abs() <= 1.3 {
                            want[l - 1][(codes[t] as usize, codes[h] as usize)] += 1.0;
                        }
                    }
                }
            }
            assert_eq!(got.counts, want, "direction {dir:?} directed {directed}");
        }
    }
}

fn overlap(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.1.min(b.1) - a.0.max(b.0)).max(0.0)
}

#[test]
fn downhole_irregular_matches_pairwise_overlaps() {
    let mut s = Stream::new(5, &[1]);
    let mut ids = Vec::new();
    let (mut from, mut to, mut codes) = (Vec::new(), Vec::new(), Vec::new());
    for hole in 0..3 {
        let mut z = 0.0;
        for _ in 0..40 {
            z += if s.uniform() < 0.2 { 0.05 } else { 0.0 };
            let len = 0.1 + 2.0 * s.uniform();
            ids.push(format!("h{hole}"));
            from.push(z);
            to.push(z + len);
            codes.push(if s.uniform() < 0.05 {
                MISSING
            } else {
                s.categorical(&P) as u32
            });
            z += len;
        }
    }
    let seqs = SequenceSet::from_intervals(&ids, &from, &to, &codes, 3).unwrap();
    let lags = [0.03, 0.4, 1.7, 5.0, 11.3];
    for direction in [Direction::Down, Direction::Up] {
        let got = downhole(&seqs, direction, &lags, 0.1).unwrap();
        let runs = seqs.runs(direction, 0.1);
        for (l, &h) in lags.iter().enumerate() {
            let mut want = Mat::zeros(3, 3);
            for run in &runs {
                let n = run.codes.len();
                let (r0, r1) = (run.start[0], run.end[n - 1]);
                for a in 0..n {
                    for b in 0..n {
                        let ab = overlap(
                            (run.start[a], run.end[a]),
                            (run.start[b] - h, run.end[b] - h),
                        );
                        if ab <= 0.0 {
                            continue;
                        }
                        let lo = run.start[a].max(run.start[b] - h).max(r0);
                        let hi = run.end[a].min(run.end[b] - h).min(r1 - h);
                        want[(run.codes[a] as usize, run.codes[b] as usize)] += (hi - lo).max(0.0);
                    }
                }
            }
            let diff = (&got.counts[l] - &want).abs().max();
            assert!(diff < 1e-9, "{direction:?} h={h}: {diff}");
        }
    }
}

#[test]
fn grid_estimator_negative_offset_matches_brute_force() {
    let grid = Grid::new([0.0; 3], [1.0, 2.0, 0.5], [7, 5, 6]).unwrap();
    let mut s = Stream::new(8, &[3]);
    let codes: Vec<u32> = (0..grid.n_cells())
        .map(|_| {
            if s.uniform() < 0.1 {
                MISSING
            } else {
                s.categorical(&P) as u32
            }
        })
        .collect();
    let offset = [-1, 2, -1];
    let got = on_grid(&codes, &grid, None, offset, 3, 3).unwrap();
    let [nx, ny, nz] = grid.count.map(|c| c as i64);
    for l in 1..=3i64 {
        let mut want = Mat::zeros(3, 3);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (i2, j2, k2) = (i + l * offset[0], j + l * offset[1], k + l * offset[2]);
                    if !(0..nx).contains(&i2) || !(0..ny).contains(&j2) || !(0..nz).contains(&k2) {
                        continue;
                    }
                    let a = codes[(i + nx * (j + ny * k)) as usize];
                    let b = codes[(i2 + nx * (j2 + ny * k2)) as usize];
                    if a != MISSING && b != MISSING {
                        want[(a as usize, b as usize)] += 1.0;
                    }
                }
            }
        }
        assert_eq!(got.counts[l as usize - 1], want, "l={l}");
        let step = (1.0f64 + 16.0 + 0.25).sqrt();
        assert!((got.lags[l as usize - 1] - l as f64 * step).abs() < 1e-12);
    }
}

fn axis_rates(p: &[f64], r00: f64, r01: f64, r10: f64, r11: f64) -> RateMatrix {
    let partial = Mat::from_row_slice(3, 3, &[r00, r01, 0.0, r10, r11, 0.0, 0.0, 0.0, 0.0]);
    RateMatrix::from_partial(&partial, p, 2).unwrap()
}

#[test]
fn table_shell_stop_matches_brute_force_on_anisotropic_models() {
    let p = [0.3, 0.4, 0.3];
    let model = TransitionModel::new(
        p.to_vec(),
        [
            axis_rates(&p, -1.0, 1.0, 0.0, -1.0),
            axis_rates(&p, -1.0, 0.0, 0.0, -1.0),
            axis_rates(&p, -2.0, 0.0, 0.5, -1.5),
        ],
        [0.0; 3],
        2,
    )
    .unwrap();
    for size in [
        [1.0, 1.0, 1.0],
        [0.2, 3.0, 1.0],
        [0.05, 1.0, 4.0],
        [1.0, 0.1, 0.3],
    ] {
        let grid = Grid::new([0.0; 3], size, [21, 21, 7]).unwrap();
        for limit in [0.3, 0.05, 0.01, 1e-3] {
            let t = model.table(&grid, limit, None).unwrap();
            let mut brute = Vec::new();
            for dz in -6i64..=6 {
                for dy in -20i64..=20 {
                    for dx in -20i64..=20 {
                        let d = [dx, dy, dz];
                        if d != [0; 3] && model.closeness(grid.lag(d)) >= limit {
                            brute.push(d);
                        }
                    }
                }
            }
            let mut got = t.offsets.clone();
            got.sort();
            brute.sort();
            assert_eq!(got.len(), brute.len(), "size {size:?} limit {limit}");
            assert_eq!(got, brute, "size {size:?} limit {limit}");
        }
    }
}

#[test]
fn fit_markov_with_an_absent_category_and_a_start() {
    let r = carle();
    let lags: Vec<f64> = (1..=6).map(f64::from).collect();
    let exact: Vec<Mat> = lags
        .iter()
        .map(|&h| r.transition(h).matrix().clone())
        .collect();
    let mut probabilities = exact.clone();
    let mut tails = vec![vec![1.0, 1.0, 1.0]; lags.len()];
    for (p, t) in probabilities.iter_mut().zip(tails.iter_mut()) {
        for j in 0..3 {
            p[(1, j)] = f64::NAN;
        }
        t[1] = 0.0;
    }
    let exp = transiograms_transiogram::Experimental {
        lags: lags.clone(),
        mean_lags: lags.clone(),
        counts: exact,
        probabilities,
        tail_totals: tails,
        directed: true,
    };
    let fit = Model1D::fit_markov(&exp, Some(&P), 2, Some(&r)).unwrap();
    let Model1D::Markov { rates, .. } = &fit else {
        panic!()
    };
    assert!(rates.matrix().iter().all(|v| v.is_finite()), "{rates:?}");
    for &h in &lags {
        let t = fit.transition(h);
        let want = r.transition(h);
        for j in 0..3 {
            assert!((t[(0, j)] - want.matrix()[(0, j)]).abs() < 1e-4, "h={h}");
        }
    }
}

#[test]
fn transition_at_non_finite_lags() {
    let sill = Mat::from_fn(3, 3, |_, j| P[j]);
    for m in [Model1D::markov(carle()).unwrap(), entrywise()] {
        assert_eq!(m.transition(f64::INFINITY), sill);
        assert_eq!(m.transition(f64::NEG_INFINITY), sill);
        assert!(max_diff(&m.transition(1e9), &sill) < 1e-10);
        assert!(m.transition(f64::NAN).iter().all(|v| v.is_nan()));
    }
}

fn two_category(lengths: [[f64; 2]; 3], rotation: [f64; 3]) -> TransitionModel {
    let p = [0.5, 0.5];
    let rates: Vec<RateMatrix> = lengths
        .iter()
        .map(|l| RateMatrix::max_entropy(&p, l).unwrap())
        .collect();
    TransitionModel::new(p.to_vec(), rates.try_into().unwrap(), rotation, 1).unwrap()
}

/// Spec 4.5: the table holds every offset with closeness >= limit. A model whose long axis
/// runs along (2, 1) has no integer offset of shell 1 near that ray, so a stop after the first
/// empty shell quits before reaching (2, 1), whose closeness is about 0.956.
#[test]
fn table_keeps_every_offset_above_the_limit_for_a_rotated_model() {
    let azimuth = -(0.5f64).atan().to_degrees();
    let m = two_category([[100.0; 2], [0.05; 2], [1.0; 2]], [azimuth, 0.0, 0.0]);
    let grid = Grid::new([0.0; 3], [1.0; 3], [10, 10, 1]).unwrap();
    let limit = 0.01;
    let c = m.closeness(grid.lag([2, 1, 0]));
    assert!(c > 0.9, "closeness of (2, 1, 0) = {c}");
    let mut brute = Vec::new();
    for dy in -9i64..=9 {
        for dx in -9i64..=9 {
            let d = [dx, dy, 0];
            if d != [0; 3] && m.closeness(grid.lag(d)) >= limit {
                brute.push(d);
            }
        }
    }
    let t = m.table(&grid, limit, None).unwrap();
    let mut got = t.offsets.clone();
    got.sort();
    brute.sort();
    assert_eq!(got, brute, "table offsets vs brute force");
}

#[test]
fn table_matches_brute_force_on_rotated_anisotropic_models() {
    for rotation in [
        [-26.565, 0.0, 0.0],
        [35.0, 20.0, -15.0],
        [110.0, -60.0, 45.0],
    ] {
        let models = [
            (
                two_category([[30.0; 2], [0.3; 2], [2.0; 2]], rotation),
                Grid::new([0.0; 3], [1.0, 0.7, 0.5], [13, 11, 7]).unwrap(),
            ),
            (
                model3d(rotation),
                Grid::new([0.0; 3], [8.0, 6.0, 1.0], [13, 11, 7]).unwrap(),
            ),
        ];
        for (m, grid) in &models {
            for limit in [0.5, 0.05, 1e-3] {
                let mut brute = Vec::new();
                for dz in -6i64..=6 {
                    for dy in -10i64..=10 {
                        for dx in -12i64..=12 {
                            let d = [dx, dy, dz];
                            if d != [0; 3] && m.closeness(grid.lag(d)) >= limit {
                                brute.push(d);
                            }
                        }
                    }
                }
                let mut got = m.table(grid, limit, None).unwrap().offsets;
                got.sort();
                brute.sort();
                assert_eq!(got, brute, "rotation {rotation:?} limit {limit}");
            }
        }
    }
}

fn message(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

#[test]
fn on_grid_with_a_huge_offset_returns_instead_of_panicking() {
    let grid = Grid::new([0.0; 3], [1.0; 3], [2, 1, 1]).unwrap();
    for offset in [[i64::MAX, 0, 0], [i64::MIN, 0, 0], [1, i64::MIN, 0]] {
        let r = std::panic::catch_unwind(|| on_grid(&[0, 1], &grid, None, offset, 3, 2));
        match r {
            Ok(Ok(e)) => assert!(e.counts.iter().all(|c| *c == Mat::zeros(2, 2))),
            Ok(Err(_)) => {}
            Err(p) => panic!("on_grid panicked: {}", message(p)),
        }
    }
}

#[test]
fn point_estimator_with_large_coordinates_returns_instead_of_panicking() {
    let spec = PairSpec {
        direction: None,
        lag: 1.0,
        n_lags: 2,
        lag_tolerance: 0.5,
        angle_tolerance: 22.5,
        bandwidth: f64::INFINITY,
        vertical_bandwidth: f64::INFINITY,
        directed: true,
        pairs: PairFilter::All,
    };
    let xyz = [
        [1e20, 0.0, 0.0],
        [1e20, 0.0, 0.0],
        [-1e20, -1e300, 0.0],
        [-1e20, -1e300, 1.0],
    ];
    let r = std::panic::catch_unwind(|| experimental(&xyz, &[0, 1, 0, 1], None, 2, &spec));
    match r {
        Ok(Ok(e)) => {
            assert_eq!(e.counts[0][(0, 1)] + e.counts[0][(1, 0)], 2.0);
            assert_eq!(e.counts[1], Mat::zeros(2, 2));
        }
        Ok(Err(e)) => panic!("unexpected error {e}"),
        Err(p) => panic!("experimental panicked: {}", message(p)),
    }
}
