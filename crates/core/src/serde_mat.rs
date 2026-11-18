//! `Mat` as row-major nested lists, for `#[serde(with = "varvite_core::serde_mat")]`.
//! The `option` and `vec` submodules cover `Option<Mat>` and `Vec<Mat>`, and `values` covers
//! `Vec<f64>`. NaN is written as JSON `null` and read back as NaN, because serde_json writes
//! NaN as `null` but cannot read `null` as `f64`.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Mat;

fn wrap(v: impl Iterator<Item = f64>) -> Vec<Option<f64>> {
    v.map(|x| (!x.is_nan()).then_some(x)).collect()
}

fn unwrap(v: Vec<Option<f64>>) -> impl Iterator<Item = f64> {
    v.into_iter().map(|x| x.unwrap_or(f64::NAN))
}

fn rows(m: &Mat) -> Vec<Vec<Option<f64>>> {
    m.row_iter().map(|r| wrap(r.iter().copied())).collect()
}

fn from_rows(rows: Vec<Vec<Option<f64>>>) -> Result<Mat, String> {
    let n = rows.len();
    let m = rows.first().map_or(0, Vec::len);
    if let Some(i) = rows.iter().position(|r| r.len() != m) {
        return Err(format!(
            "matrix row {i} has {} entries, row 0 has {m}",
            rows[i].len()
        ));
    }
    Ok(Mat::from_row_iterator(
        n,
        m,
        rows.into_iter().flat_map(unwrap),
    ))
}

pub fn serialize<S: Serializer>(m: &Mat, s: S) -> Result<S::Ok, S::Error> {
    rows(m).serialize(s)
}

pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Mat, D::Error> {
    from_rows(Vec::deserialize(d)?).map_err(D::Error::custom)
}

pub mod option {
    use super::*;

    pub fn serialize<S: Serializer>(m: &Option<Mat>, s: S) -> Result<S::Ok, S::Error> {
        m.as_ref().map(rows).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Mat>, D::Error> {
        Option::<Vec<Vec<Option<f64>>>>::deserialize(d)?
            .map(from_rows)
            .transpose()
            .map_err(D::Error::custom)
    }
}

pub mod vec {
    use super::*;

    pub fn serialize<S: Serializer>(ms: &[Mat], s: S) -> Result<S::Ok, S::Error> {
        ms.iter().map(rows).collect::<Vec<_>>().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Mat>, D::Error> {
        Vec::<Vec<Vec<Option<f64>>>>::deserialize(d)?
            .into_iter()
            .map(from_rows)
            .collect::<Result<_, _>>()
            .map_err(D::Error::custom)
    }
}

pub mod values {
    use super::*;

    pub fn serialize<S: Serializer>(v: &[f64], s: S) -> Result<S::Ok, S::Error> {
        wrap(v.iter().copied()).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
        Ok(unwrap(Vec::deserialize(d)?).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Holder {
        #[serde(with = "crate::serde_mat")]
        a: Mat,
        #[serde(with = "crate::serde_mat::option")]
        b: Option<Mat>,
        #[serde(with = "crate::serde_mat::vec")]
        c: Vec<Mat>,
    }

    #[test]
    fn row_major_round_trip() {
        let a = Mat::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let h = Holder {
            a: a.clone(),
            b: None,
            c: vec![a.clone(), Mat::zeros(0, 0)],
        };
        let text = serde_json::to_string(&h).unwrap();
        assert!(
            text.starts_with(r#"{"a":[[1.0,2.0,3.0],[4.0,5.0,6.0]]"#),
            "{text}"
        );
        assert_eq!(serde_json::from_str::<Holder>(&text).unwrap(), h);
        assert!(
            serde_json::from_str::<Holder>(r#"{"a":[[1.0],[1.0,2.0]],"b":null,"c":[]}"#).is_err()
        );
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct WithNan {
        #[serde(with = "crate::serde_mat")]
        a: Mat,
        #[serde(with = "crate::serde_mat::values")]
        v: Vec<f64>,
    }

    #[test]
    fn nan_round_trips_as_null() {
        let h = WithNan {
            a: Mat::from_row_slice(1, 2, &[f64::NAN, 0.5]),
            v: vec![1.0, f64::NAN],
        };
        let text = serde_json::to_string(&h).unwrap();
        assert_eq!(text, r#"{"a":[[null,0.5]],"v":[1.0,null]}"#);
        let back: WithNan = serde_json::from_str(&text).unwrap();
        assert!(back.a[(0, 0)].is_nan() && back.a[(0, 1)] == 0.5);
        assert!(back.v[0] == 1.0 && back.v[1].is_nan());
    }
}
