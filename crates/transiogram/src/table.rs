use varvite_core::{Grid, Mat, Result};

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
        todo!()
    }
}

impl TransitionModel {
    pub fn table(&self, grid: &Grid, limit: f64, memory: Option<u64>) -> Result<Table> {
        todo!()
    }
}
