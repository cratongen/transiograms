use std::collections::HashMap;

use rayon::prelude::*;
use varvite_core::{Grid, Mat, Result, invalid, memory};

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

fn positive(d: [i64; 3]) -> bool {
    zyx(d).into_iter().find(|&v| v != 0).is_some_and(|v| v > 0)
}

/// Offsets with `max |d_v| = s` and `|d_v| < count[v]` and a positive (dz, dy, dx) sign.
fn shell(s: i64, count: [usize; 3]) -> Vec<[i64; 3]> {
    let r = count.map(|c| s.min(c as i64 - 1));
    let mut out = Vec::new();
    for dz in -r[2]..=r[2] {
        for dy in -r[1]..=r[1] {
            for dx in -r[0]..=r[0] {
                let d = [dx, dy, dz];
                if d.iter().map(|v| v.abs()).max() == Some(s) && positive(d) {
                    out.push(d);
                }
            }
        }
    }
    out
}

impl TransitionModel {
    pub fn table(&self, grid: &Grid, limit: f64, memory: Option<u64>) -> Result<Table> {
        if !limit.is_finite() {
            return invalid(format!("closeness limit must be finite, got {limit}"));
        }
        let k = self.k() as u64;
        let per_offset = (k * k + 4) * 8;
        let reach = *grid.count.iter().max().expect("three axes") as i64 - 1;
        let mut kept: Vec<([i64; 3], f64)> = Vec::new();
        for s in 1..=reach {
            let candidates = shell(s, grid.count);
            let closeness: Vec<f64> = candidates
                .par_iter()
                .map(|&d| self.try_closeness(grid.lag(d)))
                .collect::<Result<_>>()?;
            let found: Vec<([i64; 3], f64)> = candidates
                .into_iter()
                .zip(closeness)
                .filter(|&(_, c)| c >= limit)
                .collect();
            if found.is_empty() {
                break;
            }
            let n = 2 * (kept.len() + found.len()) as u64;
            memory::check(n * per_offset, memory, "transition table")?;
            kept.extend(found);
        }
        let transitions: Vec<Mat> = kept
            .par_iter()
            .map(|&(d, _)| self.transition(grid.lag(d)))
            .collect();
        let index: HashMap<[i64; 3], usize> =
            kept.iter().enumerate().map(|(i, &(d, _))| (d, i)).collect();
        let mut entries: Vec<([i64; 3], f64)> = kept
            .iter()
            .flat_map(|&(d, c)| [(d, c), (d.map(|v| -v), c)])
            .collect();
        entries.sort_by(|a, b| b.1.total_cmp(&a.1).then(zyx(a.0).cmp(&zyx(b.0))));
        let transition = entries
            .iter()
            .map(|&(d, _)| match index.get(&d) {
                Some(&i) => transitions[i].clone(),
                None => reverse(&transitions[index[&d.map(|v| -v)]], &self.proportions),
            })
            .collect();
        Ok(Table {
            offsets: entries.iter().map(|e| e.0).collect(),
            closeness: entries.iter().map(|e| e.1).collect(),
            transition,
        })
    }
}
