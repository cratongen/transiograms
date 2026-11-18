#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Convergence(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(Error::InvalidInput(message.into()))
}

pub fn convergence<T>(message: impl Into<String>) -> Result<T> {
    Err(Error::Convergence(message.into()))
}
