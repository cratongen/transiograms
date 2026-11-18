use varvite_core::Code;
use varvite_transiogram::TransitionModel;

/// Eq. 14: `p_k prod_g t_{k, c_g}(h_g)` normalised, `h_g = x_g - x_0`, in log space.
/// None when every candidate has zero likelihood.
pub fn conditional(model: &TransitionModel, neighbours: &[([f64; 3], Code)]) -> Option<Vec<f64>> {
    todo!()
}

/// Li and Zhang Eq. 13 with `first` the index of the neighbour the chain comes from; test oracle.
pub fn conditional_eq13(
    model: &TransitionModel,
    neighbours: &[([f64; 3], Code)],
    first: usize,
) -> Option<Vec<f64>> {
    todo!()
}
