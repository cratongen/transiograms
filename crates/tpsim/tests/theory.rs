use varvite_core::rng::tag;
use varvite_core::synthetic::markov_codes;
use varvite_core::*;
use varvite_tpsim::*;
use varvite_transiogram::TransitionModel;

fn carle_embedded() -> Mat {
    Mat::from_row_slice(
        3,
        3,
        &[
            0.0,
            0.6,
            0.4,
            0.7,
            0.0,
            0.3,
            0.2105263158,
            0.7894736842,
            0.0,
        ],
    )
}

/// The M5 model: K = 3, lateral maximum entropy, Carle embedded chain vertically.
fn m5_model() -> TransitionModel {
    let p = [0.5, 0.3, 0.2];
    let lateral = [12.0, 8.0, 6.0];
    let vertical = [2.0, 1.0, 0.8];
    let pi = carle_embedded();
    TransitionModel::from_geology(
        &p,
        [&lateral, &lateral, &vertical],
        [None, None, Some(&pi)],
        2,
        [0.0; 3],
    )
    .unwrap()
}

/// An asymmetric, rotated model for the cokriging identities.
fn rotated_model() -> TransitionModel {
    let p = [0.5, 0.3, 0.2];
    let pi = carle_embedded();
    TransitionModel::from_geology(
        &p,
        [&[9.0, 6.0, 5.0], &[5.0, 4.0, 3.0], &[4.0, 2.0, 1.0]],
        [None, Some(&pi), Some(&pi)],
        2,
        [30.0, 10.0, 5.0],
    )
    .unwrap()
}

/// Vertical holes through `columns`, each a stationary draw of the model's z chain at the cell
/// size, bottom to top, snapped onto the grid.
fn vertical_holes(model: &TransitionModel, grid: &Grid, columns: &[[usize; 2]]) -> Snapped {
    let t = TransitionMatrix::new(model.transition([0.0, 0.0, grid.size[2]])).unwrap();
    let mut coords = vec![];
    let mut codes = vec![];
    for (h, &[i, j]) in columns.iter().enumerate() {
        let column = markov_codes(
            &t,
            None,
            grid.count[2],
            &mut Stream::new(0, &[tag::SAMPLE, h as u64]),
        )
        .unwrap();
        for (kz, &c) in column.iter().enumerate() {
            coords.push(grid.centroid(grid.index([i, j, kz])));
            codes.push(c);
        }
    }
    grid.snap(&coords, &codes, None).unwrap()
}

fn small_case() -> (TransitionModel, Grid, Snapped) {
    let model = m5_model();
    let grid = Grid::new([0.0; 3], [3.0, 3.0, 0.5], [12, 10, 8]).unwrap();
    let hard = vertical_holes(&model, &grid, &[[2, 2], [9, 3], [5, 8]]);
    (model, grid, hard)
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn single_datum_gives_the_transition_row() {
    let model = rotated_model();
    let x1 = [1.0, -2.0, 0.5];
    let targets = [
        [3.0, 1.0, 1.5],
        [-4.0, 0.5, -0.7],
        [1.0, -2.0, 2.5],
        [0.2, 9.0, 0.0],
    ];
    for x0 in targets {
        let t = model.transition([x0[0] - x1[0], x0[1] - x1[1], x0[2] - x1[2]]);
        for c in 0..3u32 {
            let pr = cokrige(&model, &[(x1, c)], x0, 1e-8).unwrap();
            let row: Vec<f64> = (0..3).map(|j| t[(c as usize, j)]).collect();
            assert!(max_abs(&pr, &row) < 1e-12, "{pr:?} vs {row:?}");
        }
    }
}

#[test]
fn probabilities_sum_to_one_before_correction() {
    let model = rotated_model();
    let mut s = Stream::new(11, &[0]);
    let mut corrected = 0;
    for trial in 0..200 {
        let n = 1 + trial % 12;
        let scale = if trial % 2 == 0 { 1.0 } else { 0.2 };
        let data: Vec<([f64; 3], Code)> = (0..n)
            .map(|_| {
                let x = [
                    scale * (20.0 * s.uniform() - 10.0),
                    scale * (20.0 * s.uniform() - 10.0),
                    scale * (4.0 * s.uniform() - 2.0),
                ];
                (x, s.below(3) as Code)
            })
            .collect();
        let mut pr = cokrige(&model, &data, [0.0; 3], 1e-8).unwrap();
        let total: f64 = pr.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "trial {trial}: sum {total}");
        if correct(&mut pr) {
            corrected += 1;
        }
        assert!(pr.iter().all(|&p| (0.0..=1.0).contains(&p)));
        assert!((pr.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    assert!(
        corrected > 0,
        "the random configurations never needed correction"
    );
}

#[test]
fn no_data_gives_proportions_and_bad_codes_raise() {
    let model = rotated_model();
    assert_eq!(
        cokrige(&model, &[], [0.0; 3], 1e-8).unwrap(),
        model.proportions
    );
    assert!(cokrige(&model, &[([0.0; 3], 3)], [1.0; 3], 1e-8).is_err());
    assert!(cokrige(&model, &[([0.0; 3], 0)], [1.0; 3], -1.0).is_err());
}

#[test]
fn prediction_is_a_distribution_and_honours_hard_data() {
    let (model, grid, hard) = small_case();
    let mut mask = vec![true; grid.n_cells()];
    mask[grid.index([11, 9, 7])] = false;
    let opts = Options::default();
    let pred = predict(&model, &grid, Some(&mask), &hard, &opts).unwrap();
    assert_eq!(pred.probabilities.len(), grid.n_cells() * 3);
    for cell in 0..grid.n_cells() {
        let row = &pred.probabilities[cell * 3..cell * 3 + 3];
        if !mask[cell] {
            assert!(row.iter().all(|p| p.is_nan()));
            assert_eq!(pred.most_likely[cell], MISSING);
            continue;
        }
        assert!(row.iter().all(|&p| (0.0..=1.0).contains(&p)), "{row:?}");
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
        assert_eq!(pred.probabilities[cell * 3 + code as usize], 1.0);
        assert_eq!(pred.most_likely[cell], code);
    }
    assert!((0.0..=1.0).contains(&pred.corrected));
}

#[test]
fn invalid_options_raise() {
    let (model, grid, hard) = small_case();
    let odd = Options {
        max_data: 5,
        ..Options::default()
    };
    assert!(simulate(&model, &grid, None, &hard, &odd).is_err());
    let mask = vec![false; grid.n_cells()];
    assert!(predict(&model, &grid, Some(&mask), &hard, &Options::default()).is_err());
    assert!(predict(&model, &grid, Some(&[true]), &hard, &Options::default()).is_err());
    let tiny = Options {
        memory: Some(16),
        ..Options::default()
    };
    let e = simulate(&model, &grid, None, &hard, &tiny)
        .unwrap_err()
        .to_string();
    assert!(e.contains("bytes"), "{e}");
}

#[test]
fn simulation_honours_hard_data_and_quench_never_increases() {
    let (model, grid, hard) = small_case();
    let mut mask = vec![true; grid.n_cells()];
    for i in 0..4 {
        mask[grid.index([i, 0, 0])] = false;
    }
    let opts = Options {
        n_realizations: 3,
        seed: 7,
        quench: Some(QuenchOptions {
            max_iter: 6,
            tol: 0.0,
            ..QuenchOptions::default()
        }),
        ..Options::default()
    };
    let sims = simulate(&model, &grid, Some(&mask), &hard, &opts).unwrap();
    assert_eq!((sims.n, sims.n_cells, sims.k), (3, grid.n_cells(), 3));
    for r in 0..3 {
        let codes = &sims.codes[r * grid.n_cells()..(r + 1) * grid.n_cells()];
        for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
            assert_eq!(codes[cell], code, "realization {r} cell {cell}");
        }
        for cell in 0..grid.n_cells() {
            assert_eq!(codes[cell] == MISSING, !mask[cell]);
        }
        let o = &sims.objective[r];
        assert!(o.len() >= 2);
        assert!(o.windows(2).all(|w| w[1] <= w[0]), "{o:?}");
        assert!(o.last().unwrap() < &o[0]);
        let lags = quench_lags(&model, &grid, 0.4);
        let recomputed = objective(codes, &grid, Some(&mask), &model, &lags, 1.0);
        assert!((recomputed - o.last().unwrap()).abs() < 1e-12);
        assert!((0.0..=1.0).contains(&sims.corrected[r]));
    }
    let again = simulate(&model, &grid, Some(&mask), &hard, &opts).unwrap();
    assert_eq!(again, sims);
    let other = simulate(
        &model,
        &grid,
        Some(&mask),
        &hard,
        &Options { seed: 8, ..opts },
    )
    .unwrap();
    assert_ne!(other.codes, sims.codes);
}

#[test]
fn quench_lags_in_2d_and_3d() {
    let model = m5_model();
    let g3 = Grid::new([0.0; 3], [1.0, 1.0, 0.25], [5, 5, 5]).unwrap();
    let all = quench_lags(&model, &g3, 1e-9);
    assert_eq!(all.len(), 13);
    for d in &all {
        assert!(!all.contains(&[-d[0], -d[1], -d[2]]));
    }
    let strict = quench_lags(&model, &g3, 0.999);
    assert_eq!(strict, vec![[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    let g2 = Grid::new([0.0; 3], [1.0, 1.0, 1.0], [5, 5, 1]).unwrap();
    assert_eq!(quench_lags(&model, &g2, 1e-9).len(), 4);
}

#[test]
fn threads_do_not_change_results() {
    let (model, grid, hard) = small_case();
    let opts = Options {
        n_realizations: 4,
        seed: 3,
        ..Options::default()
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                (
                    simulate(&model, &grid, None, &hard, &opts).unwrap(),
                    predict(&model, &grid, None, &hard, &opts).unwrap(),
                )
            })
    };
    let (s1, p1) = run(1);
    let (s4, p4) = run(4);
    assert_eq!(s1.codes, s4.codes);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&s1.corrected), bits(&s4.corrected));
    for (a, b) in s1.objective.iter().zip(&s4.objective) {
        assert_eq!(bits(a), bits(b));
    }
    assert_eq!(bits(&p1.probabilities), bits(&p4.probabilities));
}

/// tpsim.md section 5: `Var(p_hat_k) ~ |D|^-2 sum_x sum_y C_kk(y - x)` over the table lags.
fn proportion_sd(model: &TransitionModel, grid: &Grid, limit: f64) -> Vec<f64> {
    let table = model.table(grid, limit, None).unwrap();
    let p = &model.proportions;
    let n = grid.n_cells() as f64;
    (0..model.k())
        .map(|k| {
            let mut sum = n * p[k] * (1.0 - p[k]);
            for (d, t) in table.offsets.iter().zip(&table.transition) {
                let pairs: f64 = (0..3)
                    .map(|a| (grid.count[a] as i64 - d[a].abs()) as f64)
                    .product();
                sum += pairs * p[k] * (t[(k, k)] - p[k]);
            }
            (sum / (n * n)).sqrt()
        })
        .collect()
}

/// Ensemble-mean auto-transiograms along one axis for lags 1..=n_lags.
fn auto_transiograms(
    sims: &Realizations,
    grid: &Grid,
    axis: usize,
    n_lags: usize,
) -> Vec<Vec<f64>> {
    let k = sims.k;
    let mut mean = vec![vec![0.0; k]; n_lags];
    let mut used = vec![vec![0usize; k]; n_lags];
    for r in 0..sims.n {
        let codes = &sims.codes[r * sims.n_cells..(r + 1) * sims.n_cells];
        for l in 1..=n_lags {
            let mut d = [0i64; 3];
            d[axis] = l as i64;
            let mut f = vec![0.0; k];
            let mut rows = vec![0.0; k];
            for x in 0..sims.n_cells {
                if let Some(y) = grid.offset(x, d) {
                    let a = codes[x] as usize;
                    rows[a] += 1.0;
                    if codes[y] == codes[x] {
                        f[a] += 1.0;
                    }
                }
            }
            for c in 0..k {
                if rows[c] > 0.0 {
                    mean[l - 1][c] += f[c] / rows[c];
                    used[l - 1][c] += 1;
                }
            }
        }
    }
    for l in 0..n_lags {
        for c in 0..k {
            mean[l][c] /= used[l][c] as f64;
        }
    }
    mean
}

/// The M5 simulation on a 64 x 64 x 32 grid of cell `size`: 16 vertical holes, n = 20, seed 0,
/// with 100 percent hard agreement asserted in every realization.
fn m5_simulate(model: &TransitionModel, size: [f64; 3]) -> (Grid, Realizations) {
    let grid = Grid::new([0.0; 3], size, M5_COUNT).unwrap();
    let [nx, ny, _] = grid.count;
    let columns: Vec<[usize; 2]> = (0..16)
        .map(|h| [(2 * (h % 4) + 1) * nx / 8, (2 * (h / 4) + 1) * ny / 8])
        .collect();
    let hard = vertical_holes(model, &grid, &columns);
    let opts = Options {
        n_realizations: 20,
        seed: 0,
        ..Options::default()
    };
    let sims = simulate(model, &grid, None, &hard, &opts).unwrap();
    let n = grid.n_cells();
    for r in 0..sims.n {
        for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
            assert_eq!(
                sims.codes[r * n + cell],
                code,
                "realization {r} cell {cell}"
            );
        }
    }
    (grid, sims)
}

/// Per axis, RMSE between ensemble-mean auto-transiograms and `T(l Delta_v)` diagonals within
/// 0.05 for `l Delta_v <= 2 max_i L_i,v`.
fn assert_m5_auto_transiograms(model: &TransitionModel, grid: &Grid, sims: &Realizations) {
    let lengths = [
        [12.0, 8.0, 6.0],
        [12.0, 8.0, 6.0],
        model.mean_lengths([0.0, 0.0, 1.0]).try_into().unwrap(),
    ];
    for axis in 0..3 {
        let longest = lengths[axis].iter().copied().fold(0.0, f64::max);
        let n_lags = ((2.0 * longest / grid.size[axis]) + 1e-9).floor() as usize;
        let n_lags = n_lags.min(grid.count[axis] - 1);
        let exp = auto_transiograms(sims, grid, axis, n_lags);
        let mut sq = 0.0;
        for l in 1..=n_lags {
            let mut d = [0i64; 3];
            d[axis] = l as i64;
            let t = model.transition(grid.lag(d));
            for k in 0..3 {
                sq += (exp[l - 1][k] - t[(k, k)]).powi(2);
            }
        }
        let rmse = (sq / (3 * n_lags) as f64).sqrt();
        eprintln!("axis {axis}: auto-transiogram RMSE {rmse:.4} over {n_lags} lags");
        assert!(
            rmse < 0.05,
            "axis {axis}: auto-transiogram RMSE {rmse} over {n_lags} lags"
        );
    }
}

/// M5 acceptance (tpsim.md section 6, sizes as recorded under "Implementer decisions").
#[test]
fn m5_acceptance() {
    let model = m5_model();
    let grid = Grid::new([0.0; 3], M5_SIZE, M5_COUNT).unwrap();
    let sd = proportion_sd(&model, &grid, 0.01);
    for (k, s) in sd.iter().enumerate() {
        assert!(3.0 * s < 0.02, "category {k}: 3 sigma = {}", 3.0 * s);
    }
    let (grid, sims) = m5_simulate(&model, M5_SIZE);
    let mut counts = [0.0; 3];
    for &c in &sims.codes {
        counts[c as usize] += 1.0;
    }
    for k in 0..3 {
        let p = counts[k] / sims.codes.len() as f64;
        eprintln!(
            "category {k}: ensemble proportion {p:.4}, target {}",
            model.proportions[k]
        );
        assert!(
            (p - model.proportions[k]).abs() < 0.02,
            "category {k}: ensemble proportion {p}"
        );
    }
    assert_m5_auto_transiograms(&model, &grid, &sims);
}

/// M5 at the spec cell (1, 1, 0.25): hard agreement and auto-transiograms over the full spec lag
/// range (24 lags in x and y, 16 in z). The proportion tolerance is left out because its variance
/// assertion fails at these sizes (tpsim.md "Implementer decisions").
#[test]
fn m5_spec_cell_reproduces_hard_data_and_auto_transiograms() {
    let model = m5_model();
    let (grid, sims) = m5_simulate(&model, M5_SPEC_SIZE);
    assert_m5_auto_transiograms(&model, &grid, &sims);
}

const M5_SIZE: [f64; 3] = [6.0, 6.0, 1.0];
const M5_SPEC_SIZE: [f64; 3] = [1.0, 1.0, 0.25];
const M5_COUNT: [usize; 3] = [64, 64, 32];

fn model_with(p: &[f64]) -> TransitionModel {
    let pi = carle_embedded();
    let tiny = p[2] < 0.1;
    TransitionModel::from_geology(
        p,
        if tiny {
            [&[40.0, 4.0, 2.0], &[40.0, 4.0, 2.0], &[8.0, 1.0, 0.5]]
        } else {
            [&[9.0, 6.0, 5.0], &[5.0, 4.0, 3.0], &[4.0, 2.0, 1.0]]
        },
        [
            None,
            if tiny { None } else { Some(&pi) },
            if tiny { None } else { Some(&pi) },
        ],
        2,
        [30.0, 10.0, 5.0],
    )
    .unwrap()
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Full nK x nK system solved by a minimum-norm SVD pseudo-inverse, no reduction.
fn reference(model: &TransitionModel, data: &[([f64; 3], Code)], x0: [f64; 3]) -> Vec<f64> {
    let k = model.k();
    let n = data.len();
    let a = Mat::from_fn(n * k, n * k, |r, c| {
        model.transition(sub(data[c / k].0, data[r / k].0))[(r % k, c % k)]
    });
    let b = Mat::from_fn(n * k, k, |r, c| {
        model.transition(sub(x0, data[r / k].0))[(r % k, c)]
    });
    let svd = a.clone().svd(true, true);
    let mut sv: Vec<f64> = svd.singular_values.iter().copied().collect();
    sv.sort_by(|x, y| x.total_cmp(y));
    let eps = if n > 1 {
        0.5 * (sv[n - 2] + sv[n - 1])
    } else {
        0.0
    };
    let mut w = svd.solve(&b, eps).unwrap();
    for _ in 0..3 {
        let r = &b - &a * &w;
        w += svd.solve(&r, eps).unwrap();
    }
    let mut pr = vec![0.0; k];
    for (alpha, d) in data.iter().enumerate() {
        for j in 0..k {
            pr[j] += w[(alpha * k + d.1 as usize, j)];
        }
    }
    pr
}

#[test]
fn reduced_system_matches_full_pseudo_inverse() {
    for p in [vec![0.5, 0.3, 0.2], vec![0.9, 0.08, 0.02]] {
        let model = model_with(&p);
        let mut s = Stream::new(42, &[1]);
        for trial in 0..60 {
            let n = 1 + trial % 6;
            let data: Vec<([f64; 3], Code)> = (0..n)
                .map(|_| {
                    (
                        [
                            12.0 * s.uniform() - 6.0,
                            12.0 * s.uniform() - 6.0,
                            3.0 * s.uniform() - 1.5,
                        ],
                        s.below(3) as Code,
                    )
                })
                .collect();
            let got = cokrige(&model, &data, [0.3, -0.2, 0.1], 1e-12).unwrap();
            let want = reference(&model, &data, [0.3, -0.2, 0.1]);
            assert!(
                max_abs(&got, &want) < 1e-8,
                "p {p:?} trial {trial}: {got:?} vs {want:?}"
            );
        }
    }
}

#[test]
fn duplicate_datum_equals_single_datum() {
    let model = model_with(&[0.5, 0.3, 0.2]);
    let x1 = [1.0, -2.0, 0.5];
    let x0 = [3.0, 1.0, 1.5];
    for c in 0..3u32 {
        let one = cokrige(&model, &[(x1, c)], x0, 1e-8).unwrap();
        let two = cokrige(&model, &[(x1, c), (x1, c)], x0, 1e-8).unwrap();
        assert!(max_abs(&one, &two) < 1e-8, "code {c}: {one:?} vs {two:?}");
    }
}

#[test]
fn non_finite_coordinates_return_an_error_not_a_panic() {
    let model = model_with(&[0.5, 0.3, 0.2]);
    let r = std::panic::catch_unwind(|| {
        cokrige(&model, &[([f64::NAN, 0.0, 0.0], 0)], [1.0, 0.0, 0.0], 1e-8)
    });
    assert!(matches!(r, Ok(Err(_))), "NaN datum: {r:?}");
    let r = std::panic::catch_unwind(|| {
        cokrige(&model, &[([0.0; 3], 0)], [f64::INFINITY, 0.0, 0.0], 1e-8)
    });
    assert!(matches!(r, Ok(Err(_))), "infinite target: {r:?}");
}

/// Grid path: one hard datum, the prediction at every cell equals row c of `T(x0 - x1)`,
/// above and below the datum along an asymmetric vertical chain, on an anisotropic grid.
#[test]
fn grid_single_datum_matches_transition_row() {
    let model = model_with(&[0.5, 0.3, 0.2]);
    let grid = Grid::new([0.0; 3], [2.0, 1.5, 0.4], [5, 4, 9]).unwrap();
    let hcell = grid.index([2, 1, 4]);
    for c in 0..3u32 {
        let hard = Snapped {
            cells: vec![hcell],
            codes: vec![c],
            conflicts: 0,
            outside: 0,
        };
        let opts = Options {
            max_data: 2,
            closeness_limit: 1e-6,
            ..Options::default()
        };
        let pred = predict(&model, &grid, None, &hard, &opts).unwrap();
        let x1 = grid.centroid(hcell);
        for cell in 0..grid.n_cells() {
            if cell == hcell {
                continue;
            }
            let t = model.transition(sub(grid.centroid(cell), x1));
            let want: Vec<f64> = (0..3).map(|j| t[(c as usize, j)]).collect();
            let got = &pred.probabilities[cell * 3..cell * 3 + 3];
            assert!(
                max_abs(got, &want) < 1e-10,
                "code {c} cell {:?}: {got:?} vs {want:?}",
                grid.ijk(cell)
            );
        }
    }
}

/// Spec section 7: on a line x1 < x2 < x0, `W1 = 0` and `W2 = T(x0 - x2)`; only `Pr` is
/// observable publicly, so check the screened probabilities for every code pair.
#[test]
fn screening_probabilities_on_a_vertical_line() {
    let model = model_with(&[0.5, 0.3, 0.2]);
    let x1 = [0.0, 0.0, -1.1];
    let x2 = [0.0, 0.0, 0.3];
    let x0 = [0.0, 0.0, 1.2];
    let t02 = model.transition(sub(x0, x2));
    for c1 in 0..3u32 {
        for c2 in 0..3u32 {
            let pr = cokrige(&model, &[(x1, c1), (x2, c2)], x0, 1e-8).unwrap();
            let want: Vec<f64> = (0..3).map(|j| t02[(c2 as usize, j)]).collect();
            assert!(max_abs(&pr, &want) < 1e-10, "{c1} {c2}: {pr:?} vs {want:?}");
        }
    }
}

#[test]
fn huge_even_max_data_returns_an_error_not_a_panic() {
    let model = model_with(&[0.5, 0.3, 0.2]);
    let grid = Grid::new([0.0; 3], [1.0, 1.0, 0.5], [4, 3, 2]).unwrap();
    let hard = Snapped {
        cells: vec![0],
        codes: vec![1],
        conflicts: 0,
        outside: 0,
    };
    let opts = Options {
        max_data: usize::MAX - 1,
        quench: None,
        ..Options::default()
    };
    let r = std::panic::catch_unwind(|| predict(&model, &grid, None, &hard, &opts).is_ok());
    assert!(r.is_ok(), "predict panicked with max_data = usize::MAX - 1");
    let r = std::panic::catch_unwind(|| simulate(&model, &grid, None, &hard, &opts).is_ok());
    assert!(
        r.is_ok(),
        "simulate panicked with max_data = usize::MAX - 1"
    );
}

fn quench_model() -> TransitionModel {
    let pi = Mat::from_row_slice(3, 3, &[0.0, 0.6, 0.4, 0.7, 0.0, 0.3, 0.2, 0.8, 0.0]);
    TransitionModel::from_geology(
        &[0.5, 0.3, 0.2],
        [&[4.0, 3.0, 2.0], &[3.0, 2.0, 2.0], &[2.0, 1.0, 0.8]],
        [None, None, Some(&pi)],
        2,
        [0.0; 3],
    )
    .unwrap()
}

/// Converged zero-temperature quenching leaves every unfrozen cell at a local minimum of the
/// fully recomputed objective (brute force over every single-cell change), and the returned
/// history ends at the recomputed objective.
#[test]
fn converged_quench_is_a_brute_force_local_minimum() {
    let model = quench_model();
    let grid = Grid::new([0.0; 3], [1.5, 1.0, 0.4], [5, 4, 3]).unwrap();
    let n = grid.n_cells();
    let mut mask = vec![true; n];
    mask[7] = false;
    mask[30] = false;
    let mut s = Stream::new(9, &[0]);
    let mut codes: Vec<Code> = (0..n)
        .map(|x| if mask[x] { s.below(3) as Code } else { MISSING })
        .collect();
    let frozen: Vec<bool> = (0..n).map(|x| x % 5 == 0).collect();
    let start = codes.clone();
    let q = QuenchOptions {
        max_iter: 1000,
        tol: 0.0,
        proportion_weight: 0.6,
        ..QuenchOptions::default()
    };
    let history = quench(&mut codes, &frozen, &grid, Some(&mask), &model, &q, 4, 2).unwrap();
    assert!(history.len() < 1001, "did not converge");
    let lags = quench_lags(&model, &grid, q.limit);
    let o = objective(&codes, &grid, Some(&mask), &model, &lags, 0.6);
    assert!((o - history.last().unwrap()).abs() < 1e-12);
    for x in 0..n {
        if frozen[x] || !mask[x] {
            assert_eq!(codes[x], start[x]);
            continue;
        }
        for b in 0..3u32 {
            let mut c = codes.clone();
            c[x] = b;
            let ob = objective(&c, &grid, Some(&mask), &model, &lags, 0.6);
            assert!(ob >= o - 1e-12 * history[0], "cell {x} -> {b}: {ob} < {o}");
        }
    }
}

/// tpsim.md section 5 at `M5_SIZE`, brute force over every lag of the grid (no table
/// truncation, which `m5_acceptance` applies at closeness 0.01).
#[test]
fn implementer_m5_sizes_pass_the_untruncated_variance_assertion() {
    let model = m5_model();
    let size = M5_SIZE;
    let count = M5_COUNT.map(|c| c as i64);
    let p = &model.proportions;
    let n = (count[0] * count[1] * count[2]) as f64;
    let mut sum = [0.0f64; 3];
    for dz in -(count[2] - 1)..count[2] {
        for dy in -(count[1] - 1)..count[1] {
            for dx in -(count[0] - 1)..count[0] {
                let d = [dx, dy, dz];
                let pairs: f64 = (0..3).map(|a| (count[a] - d[a].abs()) as f64).product();
                let t = model.transition([
                    dx as f64 * size[0],
                    dy as f64 * size[1],
                    dz as f64 * size[2],
                ]);
                for k in 0..3 {
                    sum[k] += pairs * p[k] * (t[(k, k)] - p[k]);
                }
            }
        }
    }
    for k in 0..3 {
        let sd = (sum[k] / (n * n)).sqrt();
        eprintln!("category {k}: 3 sigma = {}", 3.0 * sd);
        assert!(3.0 * sd < 0.02, "category {k}: 3 sigma = {}", 3.0 * sd);
    }
}

fn row(model: &TransitionModel, h: [f64; 3], c: Code) -> Vec<f64> {
    let t = model.transition(h);
    (0..model.k()).map(|j| t[(c as usize, j)]).collect()
}

#[test]
fn target_on_a_datum_returns_its_code() {
    let model = model_with(&[0.5, 0.3, 0.2]);
    let x1 = [1.0, -2.0, 0.5];
    let x2 = [3.0, 1.0, 1.5];
    for c1 in 0..3u32 {
        for c2 in 0..3u32 {
            let pr = cokrige(&model, &[(x1, c1), (x2, c2)], x1, 1e-8).unwrap();
            let mut want = vec![0.0; 3];
            want[c1 as usize] = 1.0;
            assert!(max_abs(&pr, &want) < 1e-10, "{c1} {c2}: {pr:?}");
        }
    }
}

#[test]
fn duplicate_datum_with_zero_cutoff_equals_single_datum() {
    let model = model_with(&[0.5, 0.3, 0.2]);
    let x1 = [1.0, -2.0, 0.5];
    let x0 = [3.0, 1.0, 1.5];
    for c in 0..3u32 {
        let one = row(&model, sub(x0, x1), c);
        let two = cokrige(&model, &[(x1, c), (x1, c)], x0, 0.0);
        match two {
            Ok(two) => assert!(max_abs(&one, &two) < 1e-8, "code {c}: {one:?} vs {two:?}"),
            Err(e) => panic!("code {c}: svd_cutoff = 0 with a duplicate datum gave {e}"),
        }
    }
}

#[test]
fn near_duplicate_datum_equals_single_datum() {
    let model = model_with(&[0.5, 0.3, 0.2]);
    let x1 = [1.0, -2.0, 0.5];
    let x1b = [1.0 + 1e-9, -2.0, 0.5];
    let x0 = [3.0, 1.0, 1.5];
    for c in 0..3u32 {
        let one = row(&model, sub(x0, x1), c);
        let two = cokrige(&model, &[(x1, c), (x1b, c)], x0, 1e-8).unwrap();
        assert!(max_abs(&one, &two) < 1e-6, "code {c}: {one:?} vs {two:?}");
    }
}

#[test]
fn three_point_screening_on_a_line() {
    let model = model_with(&[0.5, 0.3, 0.2]);
    let x1 = [0.0, 0.0, -2.0];
    let x2 = [0.0, 0.0, -0.7];
    let x3 = [0.0, 0.0, 0.4];
    let x0 = [0.0, 0.0, 1.1];
    for c1 in 0..3u32 {
        for c2 in 0..3u32 {
            for c3 in 0..3u32 {
                let pr = cokrige(&model, &[(x1, c1), (x2, c2), (x3, c3)], x0, 1e-8).unwrap();
                let want = row(&model, sub(x0, x3), c3);
                assert!(
                    max_abs(&pr, &want) < 1e-10,
                    "{c1}{c2}{c3}: {pr:?} vs {want:?}"
                );
            }
        }
    }
}

#[test]
fn tiny_proportion_single_datum_and_sum() {
    let model = TransitionModel::from_geology(
        &[0.6, 0.399, 0.001],
        [&[6.0, 4.0, 0.5], &[6.0, 4.0, 0.5], &[3.0, 2.0, 0.25]],
        [None, None, None],
        0,
        [30.0, 10.0, 5.0],
    )
    .unwrap();
    let x1 = [1.0, -2.0, 0.5];
    let x0 = [3.0, 1.0, 1.5];
    for c in 0..3u32 {
        let pr = cokrige(&model, &[(x1, c)], x0, 1e-8).unwrap();
        assert!(max_abs(&pr, &row(&model, sub(x0, x1), c)) < 1e-12);
        let pr = cokrige(&model, &[(x1, c), ([0.0, 0.5, 2.0], 2)], x0, 1e-8).unwrap();
        assert!((pr.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{pr:?}");
    }
}

/// `predict` on a grid whose cell count fits in usize but whose byte count does not must return
/// `InvalidInput`, not overflow in the memory check.
#[test]
fn predict_on_an_absurd_grid_returns_an_error() {
    let model = model_with(&[0.5, 0.3, 0.2]);
    let grid = Grid::new([0.0; 3], [1.0; 3], [1 << 31, 1 << 31, 1]).unwrap();
    let hard = Snapped {
        cells: vec![],
        codes: vec![],
        conflicts: 0,
        outside: 0,
    };
    let opts = Options {
        quench: None,
        ..Options::default()
    };
    let r = std::panic::catch_unwind(|| predict(&model, &grid, None, &hard, &opts).is_err());
    assert!(matches!(r, Ok(true)), "predict on 2^62 cells: {r:?}");
    let r = std::panic::catch_unwind(|| simulate(&model, &grid, None, &hard, &opts).is_err());
    assert!(matches!(r, Ok(true)), "simulate on 2^62 cells: {r:?}");
}

/// A hand-built Snapped with the same cell twice and different codes must be rejected; otherwise
/// one hard datum would be silently dropped.
#[test]
fn duplicate_hard_cells_with_conflicting_codes_are_rejected() {
    let model = model_with(&[0.5, 0.3, 0.2]);
    let grid = Grid::new([0.0; 3], [1.0, 1.0, 0.5], [4, 3, 2]).unwrap();
    let hard = Snapped {
        cells: vec![5, 5],
        codes: vec![0, 1],
        conflicts: 0,
        outside: 0,
    };
    let opts = Options {
        quench: None,
        ..Options::default()
    };
    let r = simulate(&model, &grid, None, &hard, &opts);
    if let Ok(sims) = &r {
        let kept = sims.codes[5];
        panic!("accepted conflicting hard data at cell 5: codes 0 and 1, simulated {kept}");
    }
    let e = r.unwrap_err().to_string();
    assert!(e.contains("hard cell 5 appears more than once"), "{e}");
    assert!(predict(&model, &grid, None, &hard, &opts).is_err());
}

/// 100 000 data with K = 3 make a 200 001 x 200 001 reduced system (about 320 GB of f64).
/// `cokrige` must reject it with an error before allocating.
#[test]
fn cokrige_with_huge_data_returns_an_error() {
    let model = TransitionModel::from_geology(
        &[0.5, 0.3, 0.2],
        [&[4.0, 3.0, 2.0], &[4.0, 3.0, 2.0], &[2.0, 1.0, 0.8]],
        [None, None, None],
        0,
        [0.0; 3],
    )
    .unwrap();
    let data: Vec<([f64; 3], Code)> = (0..100_000)
        .map(|i| ([i as f64, 0.0, 0.0], (i % 3) as Code))
        .collect();
    let r = std::panic::catch_unwind(|| cokrige(&model, &data, [0.5, 0.5, 0.5], 1e-8).is_err());
    assert!(matches!(r, Ok(true)), "{r:?}");
}
