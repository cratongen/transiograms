//! Axis-aligned grids (x fastest), snapped hard data and per-cell results.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::codes::{Code, MISSING};
use crate::error::{Result, invalid};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grid {
    pub origin: [f64; 3],
    pub size: [f64; 3],
    pub count: [usize; 3],
}

impl Grid {
    /// `origin` is the corner of the first cell; sizes positive; counts at least 1.
    pub fn new(origin: [f64; 3], size: [f64; 3], count: [usize; 3]) -> Result<Self> {
        for a in 0..3 {
            if !origin[a].is_finite() {
                return invalid(format!("grid origin[{a}] = {} is not finite", origin[a]));
            }
            if !(size[a] > 0.0 && size[a].is_finite()) {
                return invalid(format!(
                    "grid size[{a}] = {} must be positive and finite",
                    size[a]
                ));
            }
            if count[a] == 0 {
                return invalid(format!("grid count[{a}] must be at least 1"));
            }
        }
        if count
            .iter()
            .try_fold(1usize, |a, &c| a.checked_mul(c))
            .is_none()
        {
            return invalid(format!("grid with counts {count:?} has too many cells"));
        }
        Ok(Self {
            origin,
            size,
            count,
        })
    }

    pub fn n_cells(&self) -> usize {
        self.count.iter().product()
    }

    pub fn is_2d(&self) -> bool {
        self.count[2] == 1
    }

    pub fn index(&self, ijk: [usize; 3]) -> usize {
        ijk[0] + self.count[0] * (ijk[1] + self.count[1] * ijk[2])
    }

    pub fn ijk(&self, index: usize) -> [usize; 3] {
        let [nx, ny, _] = self.count;
        [index % nx, (index / nx) % ny, index / (nx * ny)]
    }

    pub fn centroid(&self, index: usize) -> [f64; 3] {
        let ijk = self.ijk(index);
        std::array::from_fn(|a| self.origin[a] + (ijk[a] as f64 + 0.5) * self.size[a])
    }

    /// Cells are `[o + i s, o + (i + 1) s)`; the far face of the last cell is included.
    pub fn locate(&self, xyz: [f64; 3]) -> Option<usize> {
        let mut ijk = [0usize; 3];
        for a in 0..3 {
            let f = (xyz[a] - self.origin[a]) / self.size[a];
            if !(f >= 0.0 && f <= self.count[a] as f64) {
                return None;
            }
            ijk[a] = (f.floor() as usize).min(self.count[a] - 1);
        }
        Some(self.index(ijk))
    }

    pub fn offset(&self, index: usize, d: [i64; 3]) -> Option<usize> {
        let ijk = self.ijk(index);
        let mut out = [0usize; 3];
        for a in 0..3 {
            let v = ijk[a] as i64 + d[a];
            if v < 0 || v >= self.count[a] as i64 {
                return None;
            }
            out[a] = v as usize;
        }
        Some(self.index(out))
    }

    /// `d * size`.
    pub fn lag(&self, d: [i64; 3]) -> [f64; 3] {
        std::array::from_fn(|a| d[a] as f64 * self.size[a])
    }

    pub fn snap(
        &self,
        coords: &[[f64; 3]],
        codes: &[Code],
        mask: Option<&[bool]>,
    ) -> Result<Snapped> {
        if coords.len() != codes.len() {
            return invalid(format!(
                "{} coordinates but {} codes",
                coords.len(),
                codes.len()
            ));
        }
        if let Some(m) = mask
            && m.len() != self.n_cells()
        {
            return invalid(format!(
                "mask has {} values for {} cells",
                m.len(),
                self.n_cells()
            ));
        }
        let mut outside = 0;
        let mut cells: BTreeMap<usize, (f64, Code, bool)> = BTreeMap::new();
        for (&xyz, &code) in coords.iter().zip(codes) {
            let cell = match self.locate(xyz) {
                Some(c) if code != MISSING && mask.is_none_or(|m| m[c]) => c,
                _ => {
                    outside += 1;
                    continue;
                }
            };
            let c = self.centroid(cell);
            let d2: f64 = (0..3).map(|a| (xyz[a] - c[a]).powi(2)).sum();
            match cells.get_mut(&cell) {
                None => {
                    cells.insert(cell, (d2, code, false));
                }
                Some(entry) => {
                    entry.2 |= entry.1 != code;
                    if d2 < entry.0 {
                        entry.0 = d2;
                        entry.1 = code;
                    }
                }
            }
        }
        Ok(Snapped {
            cells: cells.keys().copied().collect(),
            codes: cells.values().map(|v| v.1).collect(),
            conflicts: cells.values().filter(|v| v.2).count(),
            outside,
        })
    }
}

/// Points outside the grid or mask, or with MISSING codes, are skipped and counted in `outside`.
/// Several points in one cell: nearest to the centroid wins, ties to the lowest input index;
/// `conflicts` counts cells whose points disagree. `cells` ascending.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapped {
    pub cells: Vec<usize>,
    pub codes: Vec<Code>,
    pub conflicts: usize,
    pub outside: usize,
}

/// Per-cell probabilities (n_cells * k, row per cell); inactive cells hold NaN, most_likely MISSING.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub k: usize,
    pub probabilities: Vec<f64>,
    pub most_likely: Vec<Code>,
    pub entropy: Vec<f64>,
    pub corrected: f64,
}

/// `codes` is n * n_cells, realization-major; inactive cells hold MISSING.
#[derive(Clone, Debug, PartialEq)]
pub struct Realizations {
    pub n: usize,
    pub n_cells: usize,
    pub k: usize,
    pub codes: Vec<Code>,
    pub corrected: Vec<f64>,
    pub objective: Vec<Vec<f64>>,
}

impl Prediction {
    /// Argmax with ties to the lowest index; entropy `-sum p ln p / ln k` (0 when k = 1).
    pub fn from_probabilities(k: usize, probabilities: Vec<f64>, corrected: f64) -> Self {
        assert!(
            k > 0 && probabilities.len().is_multiple_of(k),
            "probabilities must hold k values per cell"
        );
        let mut most_likely = Vec::with_capacity(probabilities.len() / k);
        let mut entropy = Vec::with_capacity(probabilities.len() / k);
        for row in probabilities.chunks(k) {
            if row.iter().any(|p| p.is_nan()) {
                most_likely.push(MISSING);
                entropy.push(f64::NAN);
                continue;
            }
            let best = (0..k).fold(0, |b, i| if row[i] > row[b] { i } else { b });
            most_likely.push(best as Code);
            let h: f64 = row.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
            entropy.push(if k == 1 { 0.0 } else { h / (k as f64).ln() });
        }
        Self {
            k,
            probabilities,
            most_likely,
            entropy,
            corrected,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Grid {
        Grid::new([10.0, 20.0, -5.0], [2.0, 1.0, 0.5], [4, 3, 2]).unwrap()
    }

    #[test]
    fn index_round_trips_and_locates_centroids() {
        let g = grid();
        assert_eq!(g.n_cells(), 24);
        for i in 0..g.n_cells() {
            assert_eq!(g.index(g.ijk(i)), i);
            assert_eq!(g.locate(g.centroid(i)), Some(i));
        }
        assert_eq!(g.index([1, 2, 1]), 1 + 4 * (2 + 3));
        assert_eq!(g.locate([18.0, 23.0, -4.0]), Some(23));
        assert_eq!(g.locate([18.01, 23.0, -4.0]), None);
        assert_eq!(g.locate([9.99, 20.0, -5.0]), None);
        assert_eq!(g.offset(0, [-1, 0, 0]), None);
        assert_eq!(g.offset(0, [3, 2, 1]), Some(23));
        assert_eq!(g.lag([1, -2, 3]), [2.0, -2.0, 1.5]);
        assert!(Grid::new([0.0; 3], [1.0, 0.0, 1.0], [1, 1, 1]).is_err());
        assert!(Grid::new([0.0; 3], [1.0; 3], [1, 0, 1]).is_err());
    }

    #[test]
    fn snap_nearest_wins_ties_to_first_and_counts() {
        let g = grid();
        let c = g.centroid(5);
        let coords = [
            [c[0] + 0.5, c[1], c[2]],
            [c[0] - 0.5, c[1], c[2]],
            [c[0] + 0.1, c[1], c[2]],
            g.centroid(1),
            g.centroid(1),
            [0.0, 0.0, 0.0],
            g.centroid(2),
            g.centroid(3),
        ];
        let codes = [0, 1, 2, 1, 2, 0, MISSING, 1];
        let mut mask = vec![true; g.n_cells()];
        mask[3] = false;
        let s = g.snap(&coords, &codes, Some(&mask)).unwrap();
        assert_eq!(s.cells, vec![1, 5]);
        assert_eq!(s.codes, vec![1, 2]);
        assert_eq!(s.conflicts, 2);
        assert_eq!(s.outside, 3);
    }

    #[test]
    fn prediction_argmax_and_entropy() {
        let p =
            Prediction::from_probabilities(2, vec![0.5, 0.5, 1.0, 0.0, f64::NAN, f64::NAN], 0.0);
        assert_eq!(p.most_likely, vec![0, 0, MISSING]);
        assert!((p.entropy[0] - 1.0).abs() < 1e-15 && p.entropy[1] == 0.0 && p.entropy[2].is_nan());
    }
}
