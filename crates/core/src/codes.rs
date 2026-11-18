use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::{Result, invalid};

pub type Code = u32;
pub const MISSING: Code = u32::MAX;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Categories {
    pub names: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colors: Option<Vec<String>>,
}

impl Categories {
    /// Non-empty, unique names, at most 65535; colors, when given, one per name.
    pub fn new(names: Vec<String>, colors: Option<Vec<String>>) -> Result<Self> {
        if names.is_empty() {
            return invalid("give at least one category name");
        }
        if names.len() > 65535 {
            return invalid(format!(
                "give at most 65535 categories, not {}",
                names.len()
            ));
        }
        let mut seen = HashSet::new();
        for (i, name) in names.iter().enumerate() {
            if name.is_empty() {
                return invalid(format!("category name {i} is empty"));
            }
            if !seen.insert(name.as_str()) {
                return invalid(format!("category name {name:?} appears twice"));
            }
        }
        if let Some(colors) = &colors
            && colors.len() != names.len()
        {
            return invalid(format!(
                "give one color per category: {} colors for {} names",
                colors.len(),
                names.len()
            ));
        }
        Ok(Self { names, colors })
    }

    /// Sorted unique non-null labels; numeric sort when every label parses as i64.
    pub fn from_labels<'a>(labels: impl IntoIterator<Item = Option<&'a str>>) -> Result<Self> {
        let unique: HashSet<&str> = labels.into_iter().flatten().collect();
        let mut names: Vec<&str> = unique.into_iter().collect();
        if names.iter().all(|s| s.parse::<i64>().is_ok()) {
            names.sort_by_key(|s| s.parse::<i64>().unwrap());
        } else {
            names.sort_unstable();
        }
        Self::new(names.into_iter().map(String::from).collect(), None)
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    pub fn code(&self, name: &str) -> Option<Code> {
        self.names.iter().position(|n| n == name).map(|i| i as Code)
    }

    pub fn name(&self, code: Code) -> Option<&str> {
        self.names.get(code as usize).map(String::as_str)
    }

    /// None -> MISSING; unknown labels -> InvalidInput listing up to 10 with their counts.
    pub fn encode<'a>(
        &self,
        labels: impl IntoIterator<Item = Option<&'a str>>,
    ) -> Result<Vec<Code>> {
        let lookup: HashMap<&str, Code> = self
            .names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i as Code))
            .collect();
        let mut unknown: BTreeMap<&str, usize> = BTreeMap::new();
        let codes = labels
            .into_iter()
            .map(|label| match label {
                None => MISSING,
                Some(s) => match lookup.get(s) {
                    Some(&c) => c,
                    None => {
                        *unknown.entry(s).or_default() += 1;
                        MISSING
                    }
                },
            })
            .collect();
        if unknown.is_empty() {
            return Ok(codes);
        }
        let listed: Vec<String> = unknown
            .iter()
            .take(10)
            .map(|(s, n)| format!("{s:?} ({n})"))
            .collect();
        invalid(format!(
            "{} labels are not in the categories: {}",
            unknown.len(),
            listed.join(", ")
        ))
    }
}

/// Every code < k or MISSING, else InvalidInput naming the first bad row.
pub fn check_codes(codes: &[Code], k: usize) -> Result<()> {
    match codes.iter().position(|&c| c != MISSING && c as usize >= k) {
        Some(row) => invalid(format!(
            "codes must be integers in 0..{} or -1 for missing: row {row} has {}",
            k.saturating_sub(1),
            codes[row]
        )),
        None => Ok(()),
    }
}

/// max(code) + 1 over non-missing codes; 0 when all are missing.
pub fn n_categories(codes: &[Code]) -> usize {
    codes
        .iter()
        .filter(|&&c| c != MISSING)
        .map(|&c| c as usize + 1)
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn validates_names_and_colors() {
        assert!(Categories::new(vec![], None).is_err());
        assert!(Categories::new(names(&["a", "a"]), None).is_err());
        assert!(Categories::new(names(&["a", ""]), None).is_err());
        assert!(Categories::new(names(&["a", "b"]), Some(names(&["#fff"]))).is_err());
        let c = Categories::new(names(&["sand", "clay"]), Some(names(&["#ff0", "#888"]))).unwrap();
        assert_eq!(c.code("clay"), Some(1));
        assert_eq!(c.name(0), Some("sand"));
        assert_eq!(c.name(2), None);
    }

    #[test]
    fn labels_sort_numerically_when_all_integers() {
        let c = Categories::from_labels([Some("10"), Some("2"), None, Some("2")]).unwrap();
        assert_eq!(c.names, names(&["2", "10"]));
        let c = Categories::from_labels([Some("b"), Some("a"), Some("10")]).unwrap();
        assert_eq!(c.names, names(&["10", "a", "b"]));
    }

    #[test]
    fn encode_lists_unknown_labels() {
        let c = Categories::new(names(&["a", "b"]), None).unwrap();
        assert_eq!(
            c.encode([Some("b"), None, Some("a")]).unwrap(),
            vec![1, MISSING, 0]
        );
        let e = c.encode([Some("x"), Some("x"), Some("y")]).unwrap_err();
        assert!(e.to_string().contains("\"x\" (2)"), "{e}");
    }

    #[test]
    fn json_rejects_unknown_keys() {
        let c: Categories = serde_json::from_str(r#"{"names": ["a"]}"#).unwrap();
        assert_eq!(c.colors, None);
        assert!(serde_json::from_str::<Categories>(r#"{"names": ["a"], "z": 1}"#).is_err());
    }

    #[test]
    fn code_checks() {
        assert!(check_codes(&[0, 1, MISSING], 2).is_ok());
        let e = check_codes(&[0, 2], 2).unwrap_err();
        assert!(e.to_string().contains("row 1"));
        assert_eq!(n_categories(&[MISSING, 3, 1]), 4);
        assert_eq!(n_categories(&[MISSING]), 0);
    }
}
