use std::mem::size_of;

use rayon::prelude::*;
use transiograms_core::{Grid, Mat, Result, invalid, memory};

use crate::model1d::reverse;
use crate::model3d::TransitionModel;

/// Offsets with closeness >= limit, both signs, excluding 0, sorted by closeness descending,
/// ties by (dz, dy, dx) lexicographic ascending.
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    pub offsets: Vec<[i64; 3]>,
    pub closeness: Vec<f64>,
    pub transition: Vec<Mat>,
}

impl Table {
    pub fn get(&self, d: [i64; 3]) -> Option<usize> {
        self.offsets.iter().position(|&o| o == d)
    }
}

fn zyx(d: [i64; 3]) -> [i64; 3] {
    [d[2], d[1], d[0]]
}

type Kept = ([i64; 3], f64);

impl TransitionModel {
    /// Half-extents in cells of a box that holds every offset with closeness >= limit.
    ///
    /// With `A_v = sum_{i != q} |r^v_ii|`, `-|h| tr R_phi >= sum_{i != q} |h| |r_ii(phi)|
    /// >= |(A_v h'_v)_v|` (Minkowski), so closeness >= limit confines `h'` to the ellipsoid
    /// `|(A_v h'_v)_v| <= (K - 1) ln(1 / limit)`.
    fn reach(&self, grid: &Grid, limit: f64) -> [i64; 3] {
        let full = grid.count.map(|c| c as i64 - 1);
        let k = self.k();
        if k == 1 {
            return full;
        }
        let bound = ((k - 1) as f64 * (1.0 / limit).ln()).max(0.0);
        let q = self.background;
        let a: [f64; 3] = std::array::from_fn(|v| {
            let m = self.rates[v].matrix();
            (0..k).filter(|&i| i != q).map(|i| m[(i, i)].abs()).sum()
        });
        let axes = self.axes();
        std::array::from_fn(|g| {
            let s: f64 = (0..3).map(|v| (axes[v][g] / a[v]).powi(2)).sum();
            let cells = (bound * s.sqrt() / grid.size[g]).ceil() + 1.0;
            if cells < full[g] as f64 {
                cells as i64
            } else {
                full[g]
            }
        })
    }

    pub fn table(&self, grid: &Grid, limit: f64, memory: Option<u64>) -> Result<Table> {
        if !limit.is_finite() {
            return invalid(format!("closeness limit must be finite, got {limit}"));
        }
        let e = self.reach(grid, limit);
        let rows = |dz: i64| if dz == 0 { 0 } else { -e[1] }..=e[1];
        let row = move |dz: i64, dy: i64| {
            let lo = if dz == 0 && dy == 0 { 1 } else { -e[0] };
            (lo..=e[0]).map(move |dx| [dx, dy, dz])
        };
        let keep = |d: [i64; 3]| -> Result<Option<f64>> {
            let c = self.try_closeness(grid.lag(d))?;
            Ok((c >= limit).then_some(c))
        };
        let mut planes = Vec::new();
        for dz in 0..=e[2] {
            let n = rows(dz)
                .into_par_iter()
                .map(|dy| row(dz, dy).try_fold(0u64, |n, d| Ok(n + keep(d)?.is_some() as u64)))
                .try_reduce(|| 0, |a, b| Ok(a + b))?;
            planes.push(n);
        }
        let half: u64 = planes.iter().sum();
        let largest = planes.iter().copied().max().unwrap_or(0);
        let k = self.k() as u64;
        let mat = size_of::<Mat>() as u64;
        let cells = k * k * 8;
        let kept = size_of::<Kept>() as u64;
        let per_half = kept
            + (mat + cells)
            + 2 * size_of::<usize>() as u64
            + 2 * (size_of::<[i64; 3]>() + size_of::<f64>()) as u64
            + 2 * mat
            + cells;
        let scratch = rayon::current_num_threads() as u64 * 64 * (mat + cells) + (1 << 20);
        let needed = half
            .saturating_mul(per_half)
            .saturating_add(largest.saturating_mul(4 * kept))
            .saturating_add(scratch);
        memory::check(needed, memory, "transition table")?;

        let half = half as usize;
        let mut kept: Vec<Kept> = Vec::with_capacity(half);
        for (dz, &n) in (0..=e[2]).zip(&planes) {
            if n == 0 {
                continue;
            }
            let plane: Vec<Kept> = rows(dz)
                .into_par_iter()
                .flat_map_iter(|dy| {
                    row(dz, dy).filter_map(|d| {
                        let c = self.closeness(grid.lag(d));
                        (c >= limit).then_some((d, c))
                    })
                })
                .collect();
            kept.extend(plane);
        }
        let mut positive: Vec<Mat> = kept
            .par_iter()
            .map(|&(d, _)| self.transition(grid.lag(d)))
            .collect();
        let entry = |e: usize| {
            let (d, c) = kept[e / 2];
            (
                if e.is_multiple_of(2) {
                    d
                } else {
                    d.map(|v| -v)
                },
                c,
            )
        };
        let mut order: Vec<usize> = (0..2 * half).collect();
        order.sort_unstable_by(|&a, &b| {
            let (a, b) = (entry(a), entry(b));
            b.1.total_cmp(&a.1).then(zyx(a.0).cmp(&zyx(b.0)))
        });
        let offsets = order.iter().map(|&e| entry(e).0).collect();
        let closeness = order.iter().map(|&e| entry(e).1).collect();
        drop(kept);
        let mut transition: Vec<Mat> = order
            .par_iter()
            .map(|&e| {
                if e.is_multiple_of(2) {
                    Mat::zeros(0, 0)
                } else {
                    reverse(&positive[e / 2], &self.proportions)
                }
            })
            .collect();
        for (slot, &e) in transition.iter_mut().zip(&order) {
            if e.is_multiple_of(2) {
                *slot = std::mem::replace(&mut positive[e / 2], Mat::zeros(0, 0));
            }
        }
        Ok(Table {
            offsets,
            closeness,
            transition,
        })
    }
}
