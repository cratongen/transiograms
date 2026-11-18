use varvite_core::synthetic::{drillholes, layered_field, rate_holes};
use varvite_core::{Direction, Error, Grid, MISSING, Mat, RateMatrix, SequenceSet, Stream};
use varvite_transiogram::{
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
