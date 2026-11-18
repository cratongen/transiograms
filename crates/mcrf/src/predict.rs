use varvite_core::{Grid, Prediction, Result, Snapped};
use varvite_transiogram::TransitionModel;

use crate::Options;

pub fn predict(
    model: &TransitionModel,
    grid: &Grid,
    mask: Option<&[bool]>,
    hard: &Snapped,
    opts: &Options,
) -> Result<Prediction> {
    todo!()
}
