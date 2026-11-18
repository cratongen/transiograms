use varvite_core::rng::tag;
use varvite_core::synthetic::layered_field;
use varvite_core::{Code, Grid, MISSING, Mat, RateMatrix, Snapped, Stream};
use varvite_mcrf::{Neighbourhood, Options, conditional, conditional_eq13, predict, simulate};
use varvite_transiogram::TransitionModel;

const P: [f64; 3] = [0.5, 0.3, 0.2];

fn golden_q() -> RateMatrix {
    RateMatrix::new(Mat::from_row_slice(
        3,
        3,
        &[-0.25, 0.15, 0.10, 0.35, -0.5, 0.15, 0.1, 0.375, -0.475],
    ))
    .unwrap()
}

/// Asymmetric (non-reversible) golden generator on `axis`, maximum entropy on the others.
fn model_on(axis: usize) -> TransitionModel {
    let lateral = RateMatrix::max_entropy(&P, &[3.0, 2.0, 1.5]).unwrap();
    let mut rates = [lateral.clone(), lateral.clone(), lateral];
    rates[axis] = golden_q();
    TransitionModel::new(P.to_vec(), rates, [0.0; 3], 2).unwrap()
}

fn t(h: f64) -> Mat {
    golden_q().transition(h).matrix().clone()
}

fn snapped(cells: Vec<usize>, codes: Vec<Code>) -> Snapped {
    Snapped {
        cells,
        codes,
        conflicts: 0,
        outside: 0,
    }
}

fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tol, "{a:?} vs {b:?}");
    }
}

#[test]
fn eq14_equals_eq13_for_an_asymmetric_model() {
    let model = model_on(2);
    let neighbours: Vec<([f64; 3], Code)> = vec![
        ([0.0, 0.0, -0.75], 1),
        ([0.0, 0.0, 1.25], 2),
        ([0.0, 0.0, -2.5], 0),
        ([1.0, -0.5, 0.5], 1),
        ([-2.0, 1.0, -0.25], 2),
    ];
    let eq14 = conditional(&model, &neighbours).unwrap();
    assert!((eq14.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    for first in 0..neighbours.len() {
        close(
            &eq14,
            &conditional_eq13(&model, &neighbours, first).unwrap(),
            1e-12,
        );
    }
}

#[test]
fn single_neighbour_reduces_to_the_transiogram() {
    let model = model_on(2);
    for (h, c) in [
        ([0.0, 0.0, 0.6], 0),
        ([0.0, 0.0, -1.7], 1),
        ([0.4, -1.1, 0.3], 2),
    ] {
        let got = conditional(&model, &[(h, c)]).unwrap();
        let back = model.transition([-h[0], -h[1], -h[2]]);
        let row: Vec<f64> = (0..3).map(|k| back[(c as usize, k)]).collect();
        close(&got, &row, 1e-12);
    }
    let z = model.transition([0.0, 0.0, 1.0]);
    let up: Vec<f64> = (0..3).map(|k| z[(1, k)]).collect();
    close(
        &conditional(&model, &[([0.0, 0.0, -1.0], 1)]).unwrap(),
        &up,
        1e-12,
    );
    close(
        &up,
        &(0..3).map(|k| t(1.0)[(1, k)]).collect::<Vec<_>>(),
        1e-12,
    );
}

#[test]
fn neighbour_at_lag_zero_is_one_hot_and_no_data_gives_proportions() {
    let model = model_on(2);
    close(
        &conditional(&model, &[([0.0; 3], 2)]).unwrap(),
        &[0.0, 0.0, 1.0],
        0.0,
    );
    close(&conditional(&model, &[]).unwrap(), &P, 1e-15);
}

#[test]
fn one_dimensional_two_neighbours_is_the_markov_bridge() {
    let model = model_on(2);
    for (a, b, c1, c2) in [
        (0.5, 1.5, 0usize, 2usize),
        (2.0, 0.25, 1, 1),
        (1.0, 3.0, 2, 0),
    ] {
        let neighbours = [([0.0, 0.0, -a], c1 as Code), ([0.0, 0.0, b], c2 as Code)];
        let got = conditional(&model, &neighbours).unwrap();
        let (ta, tb, tab) = (t(a), t(b), t(a + b));
        let bridge: Vec<f64> = (0..3)
            .map(|k| ta[(c1, k)] * tb[(k, c2)] / tab[(c1, c2)])
            .collect();
        close(&got, &bridge, 1e-12);
    }
}

/// Maximum-entropy joint of (Z_0, Z_1..Z_4) under the star constraints by IPF on all 3^5 cells.
#[test]
fn mcp_equals_ipf_maximum_entropy_brute_force() {
    let model = model_on(2);
    let lags = [
        [0.0, 0.0, -0.5],
        [0.0, 0.0, 1.0],
        [1.5, 0.0, 0.0],
        [-0.5, 2.0, 0.25],
    ];
    let codes = [1usize, 0, 2, 1];
    let ts: Vec<Mat> = lags.iter().map(|&h| model.transition(h)).collect();
    let k = 3usize;
    let cells = k.pow(5);
    let digit = |cell: usize, v: usize| (cell / k.pow(v as u32)) % k;
    let mut joint = vec![1.0 / cells as f64; cells];
    for _ in 0..10_000 {
        let mut change = 0f64;
        for g in 0..4 {
            let mut margin = [[0.0f64; 3]; 3];
            for (cell, &v) in joint.iter().enumerate() {
                margin[digit(cell, 0)][digit(cell, g + 1)] += v;
            }
            for (cell, v) in joint.iter_mut().enumerate() {
                let (z0, zg) = (digit(cell, 0), digit(cell, g + 1));
                let target = P[z0] * ts[g][(z0, zg)];
                let new = *v * target / margin[z0][zg];
                change = change.max((new - *v).abs());
                *v = new;
            }
        }
        if change < 1e-15 {
            break;
        }
    }
    let at = |z0: usize| {
        z0 + (0..4)
            .map(|g| codes[g] * k.pow(g as u32 + 1))
            .sum::<usize>()
    };
    let total: f64 = (0..k).map(|z0| joint[at(z0)]).sum();
    let brute: Vec<f64> = (0..k).map(|z0| joint[at(z0)] / total).collect();
    let neighbours: Vec<([f64; 3], Code)> = lags
        .iter()
        .zip(codes)
        .map(|(&h, c)| (h, c as Code))
        .collect();
    close(&conditional(&model, &neighbours).unwrap(), &brute, 1e-10);
}

/// Layer cake constant in x from a 1D chain in z; data on one column, predicted along it.
#[test]
fn layer_cake_reproduces_the_one_dimensional_chain_conditional() {
    let grid = Grid::new([0.0; 3], [1.0, 1.0, 0.25], [8, 1, 60]).unwrap();
    let field = layered_field(&grid, &golden_q(), 3).unwrap();
    let model = model_on(2);
    let column = 3;
    let data: Vec<usize> = (2..60)
        .step_by(6)
        .map(|z| grid.index([column, 0, z]))
        .collect();
    let hard = snapped(data.clone(), data.iter().map(|&c| field[c]).collect());
    let opts = Options {
        closeness_limit: 1e-6,
        ..Options::default()
    };
    let pred = predict(&model, &grid, None, &hard, &opts).unwrap();
    let dz = 0.25;
    for z in 0..60 {
        let cell = grid.index([column, 0, z]);
        let row = &pred.probabilities[cell * 3..cell * 3 + 3];
        let below = data.iter().rev().find(|&&c| grid.ijk(c)[2] <= z);
        let above = data.iter().find(|&&c| grid.ijk(c)[2] >= z);
        let expected: Vec<f64> = match (below, above) {
            (Some(&b), Some(&a)) if a == b => (0..3)
                .map(|k| (k as Code == field[a]) as u8 as f64)
                .collect(),
            (Some(&b), Some(&a)) => {
                let (hb, ha) = (
                    (z - grid.ijk(b)[2]) as f64 * dz,
                    (grid.ijk(a)[2] - z) as f64 * dz,
                );
                let (cb, ca) = (field[b] as usize, field[a] as usize);
                (0..3)
                    .map(|k| t(hb)[(cb, k)] * t(ha)[(k, ca)] / t(hb + ha)[(cb, ca)])
                    .collect()
            }
            (Some(&b), None) => {
                let h = (z - grid.ijk(b)[2]) as f64 * dz;
                (0..3).map(|k| t(h)[(field[b] as usize, k)]).collect()
            }
            (None, Some(&a)) => {
                let h = (grid.ijk(a)[2] - z) as f64 * dz;
                let ca = field[a] as usize;
                let w: Vec<f64> = (0..3).map(|k| P[k] * t(h)[(k, ca)]).collect();
                let s: f64 = w.iter().sum();
                w.iter().map(|v| v / s).collect()
            }
            (None, None) => unreachable!(),
        };
        close(row, &expected, 1e-12);
    }
}

fn scattered(grid: &Grid, n: usize, seed: u64) -> Snapped {
    let mut s = Stream::new(seed, &[tag::SAMPLE]);
    let mut cells: Vec<usize> = (0..n)
        .map(|_| s.below(grid.n_cells() as u64) as usize)
        .collect();
    cells.sort();
    cells.dedup();
    let codes = cells.iter().map(|_| s.below(3) as Code).collect();
    snapped(cells, codes)
}

#[test]
fn prediction_rows_are_distributions_and_honour_hard_data() {
    let grid = Grid::new([0.0; 3], [1.0, 1.0, 0.25], [12, 10, 8]).unwrap();
    let model = model_on(2);
    let mask: Vec<bool> = (0..grid.n_cells()).map(|c| grid.ijk(c)[0] != 11).collect();
    let mut hard = scattered(&grid, 40, 5);
    let keep: Vec<bool> = hard.cells.iter().map(|&c| mask[c]).collect();
    hard.codes = hard
        .codes
        .iter()
        .zip(&keep)
        .filter(|x| *x.1)
        .map(|x| *x.0)
        .collect();
    hard.cells = hard
        .cells
        .iter()
        .zip(&keep)
        .filter(|x| *x.1)
        .map(|x| *x.0)
        .collect();
    for neighbourhood in [Neighbourhood::Sectors, Neighbourhood::Nearest(6)] {
        let opts = Options {
            neighbourhood,
            ..Options::default()
        };
        let pred = predict(&model, &grid, Some(&mask), &hard, &opts).unwrap();
        assert_eq!(pred.corrected, 0.0);
        for cell in 0..grid.n_cells() {
            let row = &pred.probabilities[cell * 3..cell * 3 + 3];
            if !mask[cell] {
                assert!(row.iter().all(|p| p.is_nan()));
                assert_eq!(pred.most_likely[cell], MISSING);
                continue;
            }
            assert!(row.iter().all(|&p| p >= 0.0));
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!((0.0..=1.0 + 1e-12).contains(&pred.entropy[cell]));
        }
        for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
            let row = &pred.probabilities[cell * 3..cell * 3 + 3];
            for k in 0..3 {
                assert_eq!(row[k], (k as Code == code) as u8 as f64);
            }
            assert_eq!(pred.most_likely[cell], code);
        }
    }
}

#[test]
fn prediction_without_data_returns_the_proportions() {
    let grid = Grid::new([0.0; 3], [1.0, 1.0, 1.0], [6, 5, 4]).unwrap();
    let model = model_on(2);
    let pred = predict(
        &model,
        &grid,
        None,
        &snapped(vec![], vec![]),
        &Options::default(),
    )
    .unwrap();
    for row in pred.probabilities.chunks(3) {
        close(row, &P, 1e-15);
    }
}

#[test]
fn invalid_hard_data_raise() {
    let grid = Grid::new([0.0; 3], [1.0; 3], [4, 4, 1]).unwrap();
    let model = model_on(0);
    let o = Options::default();
    assert!(predict(&model, &grid, None, &snapped(vec![16], vec![0]), &o).is_err());
    assert!(predict(&model, &grid, None, &snapped(vec![1], vec![3]), &o).is_err());
    let mask = vec![false; 16];
    assert!(simulate(&model, &grid, Some(&mask), &snapped(vec![1], vec![0]), &o).is_err());
    assert!(
        simulate(
            &model,
            &grid,
            Some(&[true; 3][..]),
            &snapped(vec![], vec![]),
            &o
        )
        .is_err()
    );
}

/// Along a line the nearest informed node on each side screens the rest, so sequential MCRF
/// draws from the exact chain: realizations reproduce p and T(l dx) up to sampling error.
#[test]
fn one_dimensional_realizations_reproduce_proportions_and_transiograms() {
    let nx = 2000;
    let dx = 0.5;
    let grid = Grid::new([0.0; 3], [dx, 1.0, 1.0], [nx, 1, 1]).unwrap();
    let model = model_on(0);
    let n = 60;
    let opts = Options {
        n_realizations: n,
        seed: 11,
        closeness_limit: 1e-6,
        ..Options::default()
    };
    let real = simulate(&model, &grid, None, &snapped(vec![], vec![]), &opts).unwrap();
    assert_eq!((real.n, real.n_cells, real.k), (n, nx, 3));
    assert!(real.objective.is_empty() && real.corrected == vec![0.0; n]);
    let lags = [1usize, 2, 4, 8];
    let mut stats: Vec<Vec<f64>> = Vec::new();
    for r in 0..n {
        let codes = &real.codes[r * nx..(r + 1) * nx];
        let mut row: Vec<f64> = (0..3)
            .map(|k| codes.iter().filter(|&&c| c == k).count() as f64 / nx as f64)
            .collect();
        for &l in &lags {
            let mut f = [[0.0f64; 3]; 3];
            for x in 0..nx - l {
                f[codes[x] as usize][codes[x + l] as usize] += 1.0;
            }
            for i in 0..3 {
                let total: f64 = f[i].iter().sum();
                row.extend((0..3).map(|j| f[i][j] / total));
            }
        }
        stats.push(row);
    }
    let mut target = P.to_vec();
    for &l in &lags {
        let m = t(l as f64 * dx);
        target.extend((0..9).map(|e| m[(e / 3, e % 3)]));
    }
    for (s, &want) in target.iter().enumerate() {
        let mean = stats.iter().map(|r| r[s]).sum::<f64>() / n as f64;
        let var = stats.iter().map(|r| (r[s] - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let se = (var / n as f64).sqrt();
        assert!(se < 0.01, "statistic {s}: standard error {se} too large");
        assert!(
            (mean - want).abs() <= 4.0 * se + 2e-3,
            "statistic {s}: {mean} vs {want} (se {se})"
        );
    }
}

#[test]
fn same_seed_gives_identical_output_for_any_thread_count() {
    let grid = Grid::new([0.0; 3], [1.0, 1.0, 0.25], [14, 12, 10]).unwrap();
    let model = model_on(2);
    let hard = scattered(&grid, 50, 9);
    let mask: Vec<bool> = (0..grid.n_cells()).map(|c| grid.ijk(c)[1] != 0).collect();
    let keep: Vec<usize> = (0..hard.cells.len())
        .filter(|&i| mask[hard.cells[i]])
        .collect();
    let hard = snapped(
        keep.iter().map(|&i| hard.cells[i]).collect(),
        keep.iter().map(|&i| hard.codes[i]).collect(),
    );
    let opts = Options {
        n_realizations: 5,
        seed: 42,
        ..Options::default()
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                (
                    simulate(&model, &grid, Some(&mask), &hard, &opts).unwrap(),
                    predict(&model, &grid, Some(&mask), &hard, &opts).unwrap(),
                )
            })
    };
    let (one, p1) = run(1);
    let (four, p4) = run(4);
    assert_eq!(one, four);
    assert_eq!(p1.probabilities.len(), p4.probabilities.len());
    for (a, b) in p1.probabilities.iter().zip(&p4.probabilities) {
        assert!(a.to_bits() == b.to_bits());
    }
    let n_cells = grid.n_cells();
    for r in 0..5 {
        let codes = &one.codes[r * n_cells..(r + 1) * n_cells];
        for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
            assert_eq!(codes[cell], code);
        }
        for cell in 0..n_cells {
            assert_eq!(codes[cell] == MISSING, !mask[cell]);
        }
    }
    let other = simulate(
        &model,
        &grid,
        Some(&mask),
        &hard,
        &Options { seed: 43, ..opts },
    )
    .unwrap();
    assert_ne!(one.codes, other.codes);
}
