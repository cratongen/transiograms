use varvite_core::{Grid, Realizations, Result, Snapped};
use varvite_transiogram::TransitionModel;

use crate::Options;

pub fn simulate(
    model: &TransitionModel,
    grid: &Grid,
    mask: Option<&[bool]>,
    hard: &Snapped,
    opts: &Options,
) -> Result<Realizations> {
    todo!()
}
