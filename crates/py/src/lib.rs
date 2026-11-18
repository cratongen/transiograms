use pyo3::prelude::*;
use pyo3::types::PyType;

mod args;
mod core;
// >>> chain
mod chain;
// <<< chain
// >>> hmm
mod hmm;
// <<< hmm
// >>> hsmm
mod hsmm;
// <<< hsmm
// >>> transiogram
mod transiogram;
// <<< transiogram
// >>> tpsim
mod tpsim;
// <<< tpsim
// >>> mcrf
mod mcrf;
// <<< mcrf

/// Raises `varvite.errors.<kind>`, a subclass of both `VarviteError` and a builtin.
pub(crate) fn error(kind: &str, message: impl ToString) -> PyErr {
    Python::attach(|py| {
        match py
            .import("varvite.errors")
            .and_then(|m| m.getattr(kind))
            .and_then(|c| Ok(c.cast_into::<PyType>()?))
        {
            Ok(class) => PyErr::from_type(class, message.to_string()),
            Err(e) => e,
        }
    })
}

pub(crate) fn invalid(message: impl ToString) -> PyErr {
    error("InvalidInput", message)
}

pub(crate) fn convergence(message: impl ToString) -> PyErr {
    error("ConvergenceError", message)
}

#[pymodule]
fn _varvite(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    core::register(m)?;
    // >>> chain
    chain::register(m)?;
    // <<< chain
    // >>> hmm
    hmm::register(m)?;
    // <<< hmm
    // >>> hsmm
    hsmm::register(m)?;
    // <<< hsmm
    // >>> transiogram
    transiogram::register(m)?;
    // <<< transiogram
    // >>> tpsim
    tpsim::register(m)?;
    // <<< tpsim
    // >>> mcrf
    mcrf::register(m)?;
    // <<< mcrf
    Ok(())
}
