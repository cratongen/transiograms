//! Drillhole sequences: holes of top-down intervals, runs along a direction, beds and
//! regular resampling.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::codes::{Code, MISSING, check_codes};
use crate::error::{Result, invalid};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Up,
    Down,
}

/// One drillhole, rows sorted by `from`, intervals non-overlapping (`to[i] <= from[i+1] + 1e-9`).
#[derive(Clone, Debug, PartialEq)]
pub struct Hole {
    pub id: String,
    pub from: Vec<f64>,
    pub to: Vec<f64>,
    pub codes: Vec<Code>,
    pub rows: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SequenceSet {
    pub holes: Vec<Hole>,
    pub n_categories: usize,
}

/// A maximal stretch of one hole with no missing code and no gap, in direction order.
/// `start[i] < end[i]` are positions along the direction.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    pub hole: usize,
    pub start: Vec<f64>,
    pub end: Vec<f64>,
    pub codes: Vec<Code>,
    pub rows: Vec<usize>,
}

/// A bed: adjacent same-code intervals of one run merged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bed {
    pub code: Code,
    pub thickness: f64,
    pub open_start: bool,
    pub open_end: bool,
}

impl SequenceSet {
    /// Rows grouped by hole id in first-appearance order; `rows` keeps input row indices.
    pub fn from_intervals(
        holes: &[String],
        from: &[f64],
        to: &[f64],
        codes: &[Code],
        k: usize,
    ) -> Result<Self> {
        let n = codes.len();
        if holes.len() != n || from.len() != n || to.len() != n {
            return invalid(format!(
                "holes, from, to and codes must have one value per row: {}, {}, {}, {}",
                holes.len(),
                from.len(),
                to.len(),
                n
            ));
        }
        check_codes(codes, k)?;
        let mut order: HashMap<&str, usize> = HashMap::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (row, id) in holes.iter().enumerate() {
            let g = *order.entry(id.as_str()).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(row);
        }
        let mut out = Vec::with_capacity(groups.len());
        for mut rows in groups {
            let id = holes[rows[0]].clone();
            for &r in &rows {
                if !(from[r].is_finite() && to[r].is_finite()) {
                    return invalid(format!("hole {id:?} row {r}: depths must be finite"));
                }
                if from[r] >= to[r] {
                    return invalid(format!(
                        "hole {id:?} at depth {}: from must be below to",
                        from[r]
                    ));
                }
            }
            rows.sort_by(|&a, &b| from[a].total_cmp(&from[b]));
            for w in rows.windows(2) {
                if to[w[0]] > from[w[1]] + 1e-9 {
                    return invalid(format!(
                        "hole {id:?} at depth {}: intervals overlap ({}..{} and {}..{})",
                        from[w[1]], from[w[0]], to[w[0]], from[w[1]], to[w[1]]
                    ));
                }
            }
            out.push(Hole {
                id,
                from: rows.iter().map(|&r| from[r]).collect(),
                to: rows.iter().map(|&r| to[r]).collect(),
                codes: rows.iter().map(|&r| codes[r]).collect(),
                rows,
            });
        }
        Ok(Self {
            holes: out,
            n_categories: k,
        })
    }

    /// Each sample becomes `[depth - step/2, depth + step/2]`.
    pub fn from_samples(
        holes: &[String],
        depth: &[f64],
        codes: &[Code],
        step: f64,
        k: usize,
    ) -> Result<Self> {
        if !(step > 0.0 && step.is_finite()) {
            return invalid(format!("step must be positive and finite, got {step}"));
        }
        let from: Vec<f64> = depth.iter().map(|d| d - step / 2.0).collect();
        let to: Vec<f64> = depth.iter().map(|d| d + step / 2.0).collect();
        Self::from_intervals(holes, &from, &to, codes, k)
    }

    /// Each inner Vec is one hole with unit step; hole ids are "0", "1", ...
    pub fn from_regular(codes: &[Vec<Code>], k: usize) -> Result<Self> {
        let mut holes = Vec::with_capacity(codes.len());
        let mut row = 0;
        for (h, c) in codes.iter().enumerate() {
            check_codes(c, k)?;
            holes.push(Hole {
                id: h.to_string(),
                from: (0..c.len()).map(|i| i as f64).collect(),
                to: (0..c.len()).map(|i| i as f64 + 1.0).collect(),
                codes: c.clone(),
                rows: (row..row + c.len()).collect(),
            });
            row += c.len();
        }
        Ok(Self {
            holes,
            n_categories: k,
        })
    }

    /// Split at MISSING codes and at gaps larger than `gap`.
    pub fn runs(&self, direction: Direction, gap: f64) -> Vec<Run> {
        let mut out = Vec::new();
        for (h, hole) in self.holes.iter().enumerate() {
            let mut pieces: Vec<Vec<usize>> = Vec::new();
            let mut current: Vec<usize> = Vec::new();
            for i in 0..hole.codes.len() {
                if hole.codes[i] == MISSING {
                    if !current.is_empty() {
                        pieces.push(std::mem::take(&mut current));
                    }
                    continue;
                }
                if let Some(&last) = current.last()
                    && hole.from[i] - hole.to[last] > gap
                {
                    pieces.push(std::mem::take(&mut current));
                }
                current.push(i);
            }
            if !current.is_empty() {
                pieces.push(current);
            }
            if direction == Direction::Up {
                pieces.reverse();
                pieces.iter_mut().for_each(|p| p.reverse());
            }
            for piece in pieces {
                let (start, end) = match direction {
                    Direction::Down => (
                        piece.iter().map(|&i| hole.from[i]).collect(),
                        piece.iter().map(|&i| hole.to[i]).collect(),
                    ),
                    Direction::Up => (
                        piece.iter().map(|&i| -hole.to[i]).collect(),
                        piece.iter().map(|&i| -hole.from[i]).collect(),
                    ),
                };
                out.push(Run {
                    hole: h,
                    start,
                    end,
                    codes: piece.iter().map(|&i| hole.codes[i]).collect(),
                    rows: piece.iter().map(|&i| hole.rows[i]).collect(),
                });
            }
        }
        out
    }

    /// Per run; when `merge` is false, adjacent same-code intervals stay separate beds.
    /// The first bed of a run has `open_start`, the last `open_end`.
    pub fn beds(&self, direction: Direction, gap: f64, merge: bool) -> Vec<Vec<Bed>> {
        self.runs(direction, gap)
            .into_iter()
            .map(|run| {
                let mut beds: Vec<Bed> = Vec::new();
                for i in 0..run.codes.len() {
                    let thickness = run.end[i] - run.start[i];
                    match beds.last_mut() {
                        Some(bed) if merge && bed.code == run.codes[i] => {
                            bed.thickness += thickness
                        }
                        _ => beds.push(Bed {
                            code: run.codes[i],
                            thickness,
                            open_start: false,
                            open_end: false,
                        }),
                    }
                }
                if let Some(first) = beds.first_mut() {
                    first.open_start = true;
                }
                if let Some(last) = beds.last_mut() {
                    last.open_end = true;
                }
                beds
            })
            .collect()
    }

    /// Point-sampled codes per run (one Vec per run, possibly empty) at positions
    /// `start + (i + 1/2) step < end of run`. A position inside an internal gap takes the
    /// code of the interval before it.
    pub fn regular(&self, direction: Direction, step: f64, gap: f64) -> Result<Vec<Vec<Code>>> {
        if !(step > 0.0 && step.is_finite()) {
            return invalid(format!("step must be positive and finite, got {step}"));
        }
        Ok(self
            .runs(direction, gap)
            .into_iter()
            .map(|run| {
                let start = run.start[0];
                let end = *run.end.last().unwrap();
                let mut codes = Vec::new();
                let mut j = 0;
                loop {
                    let x = start + (codes.len() as f64 + 0.5) * step;
                    if x >= end {
                        break;
                    }
                    while j + 1 < run.codes.len() && run.start[j + 1] <= x {
                        j += 1;
                    }
                    codes.push(run.codes[j]);
                }
                codes
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn example() -> SequenceSet {
        let holes = ids(&["b", "a", "b", "a", "a", "a", "a"]);
        let from = [1.0, 0.0, 0.0, 1.0, 2.0, 3.5, 4.5];
        let to = [3.0, 1.0, 1.0, 2.0, 3.0, 4.5, 5.0];
        let codes = [1, 0, 0, 0, MISSING, 1, 2];
        SequenceSet::from_intervals(&holes, &from, &to, &codes, 3).unwrap()
    }

    #[test]
    fn groups_by_first_appearance_and_sorts_by_depth() {
        let s = example();
        assert_eq!(s.holes[0].id, "b");
        assert_eq!(s.holes[0].rows, vec![2, 0]);
        assert_eq!(s.holes[1].rows, vec![1, 3, 4, 5, 6]);
    }

    #[test]
    fn gaps_and_missing_split_runs() {
        let s = example();
        let down = s.runs(Direction::Down, 0.1);
        let rows: Vec<Vec<usize>> = down.iter().map(|r| r.rows.clone()).collect();
        assert_eq!(rows, vec![vec![2, 0], vec![1, 3], vec![5, 6]]);
        let wide = s.runs(Direction::Down, 1.0);
        assert_eq!(wide.len(), 3);
        let up = s.runs(Direction::Up, 0.1);
        let rows: Vec<Vec<usize>> = up.iter().map(|r| r.rows.clone()).collect();
        assert_eq!(rows, vec![vec![0, 2], vec![6, 5], vec![3, 1]]);
        assert_eq!(up[1].start, vec![-5.0, -4.5]);
        assert_eq!(up[1].end, vec![-4.5, -3.5]);
        assert!(
            up.iter()
                .all(|r| r.start.iter().zip(&r.end).all(|(a, b)| a < b))
        );
    }

    #[test]
    fn overlap_names_the_hole() {
        let e =
            SequenceSet::from_intervals(&ids(&["x", "x"]), &[0.0, 0.5], &[1.0, 2.0], &[0, 1], 2)
                .unwrap_err()
                .to_string();
        assert!(e.contains("\"x\"") && e.contains("0.5"), "{e}");
        assert!(SequenceSet::from_intervals(&ids(&["x"]), &[1.0], &[1.0], &[0], 2).is_err());
        assert!(SequenceSet::from_intervals(&ids(&["x"]), &[0.0], &[1.0], &[2], 2).is_err());
    }

    #[test]
    fn beds_merge_and_flag_open_ends() {
        let s = SequenceSet::from_regular(&[vec![0, 0, 1, 1, 1, 0]], 2).unwrap();
        let merged = s.beds(Direction::Down, 1e-6, true);
        assert_eq!(
            merged[0]
                .iter()
                .map(|b| (b.code, b.thickness))
                .collect::<Vec<_>>(),
            vec![(0, 2.0), (1, 3.0), (0, 1.0)]
        );
        assert!(merged[0][0].open_start && !merged[0][0].open_end);
        assert!(merged[0][2].open_end && !merged[0][1].open_start);
        assert_eq!(s.beds(Direction::Down, 1e-6, false)[0].len(), 6);
        let up = s.beds(Direction::Up, 1e-6, true);
        assert_eq!(up[0][0].code, 0);
        assert_eq!(up[0][0].thickness, 1.0);
    }

    #[test]
    fn regular_reproduces_samples() {
        let depth = [10.0, 10.5, 11.0, 11.5, 13.0, 13.5];
        let codes = [0, 1, 1, 0, 2, 2];
        let holes = ids(&["h"; 6]);
        let s = SequenceSet::from_samples(&holes, &depth, &codes, 0.5, 3).unwrap();
        assert_eq!(
            s.regular(Direction::Down, 0.5, 1e-6).unwrap(),
            vec![vec![0, 1, 1, 0], vec![2, 2]]
        );
        assert_eq!(
            s.regular(Direction::Up, 0.5, 1e-6).unwrap(),
            vec![vec![2, 2], vec![0, 1, 1, 0]]
        );
        assert_eq!(
            s.regular(Direction::Down, 1.0, 1e-6).unwrap(),
            vec![vec![1, 0], vec![2]]
        );
    }
}
