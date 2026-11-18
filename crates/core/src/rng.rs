//! Seed derivation shared bit for bit with boitata, and xoshiro256++ streams keyed by index.

/// SplitMix64 finaliser; identical to boitata's `splitmix`.
pub fn splitmix(z: u64) -> u64 {
    let z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed of realization (or stream) `k` under `seed`; identical to boitata's `realization_seed`.
pub fn realization_seed(seed: u64, k: u64) -> u64 {
    splitmix(splitmix(seed) ^ k)
}

pub mod tag {
    pub const PATH: u64 = 1;
    pub const DRAW: u64 = 2;
    pub const QUENCH: u64 = 3;
    pub const INIT: u64 = 4;
    pub const FOLD: u64 = 5;
    pub const SAMPLE: u64 = 6;
    pub const BOOTSTRAP: u64 = 7;
}

const TWO_POW_M53: f64 = 1.0 / (1u64 << 53) as f64;

/// xoshiro256++ stream whose key is `keys.fold(seed, realization_seed)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stream {
    s: [u64; 4],
}

impl Stream {
    pub fn new(seed: u64, keys: &[u64]) -> Self {
        let key = keys.iter().fold(seed, |s, &k| realization_seed(s, k));
        let mut s = [0u64; 4];
        for (i, v) in s.iter_mut().enumerate() {
            *v = splitmix(key.wrapping_add((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        }
        if s == [0; 4] {
            s[0] = 1;
        }
        Self { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// [0, 1) with 53 random bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * TWO_POW_M53
    }

    /// (0, 1).
    pub fn uniform_open(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * TWO_POW_M53
    }

    /// Uniform on 0..n by Lemire's method with rejection; n >= 1.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n >= 1, "below(n) needs n >= 1");
        let mut m = self.next_u64() as u128 * n as u128;
        if (m as u64) < n {
            let t = n.wrapping_neg() % n;
            while (m as u64) < t {
                m = self.next_u64() as u128 * n as u128;
            }
        }
        (m >> 64) as u64
    }

    /// Standard normal, Box-Muller cosine branch.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform_open();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    pub fn exponential(&mut self) -> f64 {
        -self.uniform_open().ln()
    }

    /// Gamma(shape, 1) by Marsaglia and Tsang; shape < 1 by boosting.
    pub fn gamma(&mut self, shape: f64) -> f64 {
        assert!(
            shape > 0.0 && shape.is_finite(),
            "gamma shape must be positive and finite"
        );
        if shape < 1.0 {
            return self.gamma(shape + 1.0) * self.uniform_open().powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.uniform_open();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    /// Inverse CDF over unnormalised non-negative weights, linear scan from index 0.
    pub fn categorical(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        assert!(
            total.is_finite() && total > 0.0,
            "categorical weights must have a finite positive sum"
        );
        let target = self.uniform() * total;
        let mut cumulative = 0.0;
        let mut last = 0;
        for (i, &w) in weights.iter().enumerate() {
            if w <= 0.0 {
                continue;
            }
            cumulative += w;
            last = i;
            if cumulative > target {
                return i;
            }
        }
        last
    }

    /// Fisher-Yates from the end.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_values() {
        assert_eq!(splitmix(0), 0xE220_A839_7B1D_CDAF);
        assert_eq!(realization_seed(0, 0), splitmix(splitmix(0)));
    }

    #[test]
    fn matches_boitata_formula_on_1000_pairs() {
        // Independent Python evaluation of boitata's splitmix(splitmix(seed) ^ k) for
        // seed = 1000 + 7919 i, k = 31 i (i < 1000): wrapping sum and xor of the results.
        let (mut sum, mut xor) = (0u64, 0u64);
        for i in 0..1000u64 {
            let v = realization_seed(1000 + 7919 * i, 31 * i);
            sum = sum.wrapping_add(v);
            xor ^= v;
        }
        assert_eq!((sum, xor), BOITATA_1000);
        assert_eq!(realization_seed(42, 7), BOITATA_42_7);
    }

    const BOITATA_1000: (u64, u64) = (0x757d_2f1a_8c41_85d2, 0x4c4c_4df1_59ca_8384);
    const BOITATA_42_7: u64 = 0x1606_2d6c_1339_e500;

    #[test]
    fn nearby_seeds_share_no_key() {
        let keys: std::collections::HashSet<u64> = (0..100)
            .flat_map(|s| (0..1000).map(move |k| realization_seed(s, k)))
            .collect();
        assert_eq!(keys.len(), 100_000);
    }

    #[test]
    fn stream_key_is_the_fold_of_realization_seeds() {
        let a = Stream::new(realization_seed(realization_seed(9, 2), 5), &[]);
        let b = Stream::new(9, &[2, 5]);
        assert_eq!(a, b);
        assert_ne!(Stream::new(9, &[5, 2]), b);
    }

    fn moments(xs: impl Iterator<Item = f64>) -> (f64, f64, f64) {
        let v: Vec<f64> = xs.collect();
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, var, n)
    }

    #[test]
    fn uniform_moments_within_5_sigma() {
        let mut s = Stream::new(1, &[]);
        let (mean, var, n) = moments((0..1_000_000).map(|_| s.uniform()));
        assert!(
            (mean - 0.5).abs() < 5.0 * (1.0 / 12.0 / n).sqrt(),
            "mean {mean}"
        );
        // Var of the sample variance of U(0,1): (mu4 - sigma^4) / n with mu4 = 1/80.
        let sd_var = ((1.0 / 80.0 - 1.0 / 144.0) / n).sqrt();
        assert!((var - 1.0 / 12.0).abs() < 5.0 * sd_var, "variance {var}");
        let mut s = Stream::new(1, &[]);
        assert!((0..100_000).all(|_| {
            let u = s.uniform_open();
            u > 0.0 && u < 1.0
        }));
    }

    #[test]
    fn below_is_uniform_chi_square() {
        let mut s = Stream::new(2, &[]);
        let n = 7u64;
        let draws = 700_000;
        let mut counts = [0f64; 7];
        for _ in 0..draws {
            counts[s.below(n) as usize] += 1.0;
        }
        let e = draws as f64 / n as f64;
        let x2: f64 = counts.iter().map(|c| (c - e).powi(2) / e).sum();
        assert!(crate::special::chi2_sf(x2, 6.0) > 1e-3, "chi2 {x2}");
        assert_eq!(s.below(1), 0);
    }

    #[test]
    fn normal_and_gamma_moments() {
        let mut s = Stream::new(3, &[]);
        let (mean, var, n) = moments((0..1_000_000).map(|_| s.normal()));
        assert!(mean.abs() < 5.0 / n.sqrt(), "normal mean {mean}");
        assert!(
            (var - 1.0).abs() < 5.0 * (2.0 / n).sqrt(),
            "normal variance {var}"
        );
        for a in [0.3f64, 1.0, 5.0] {
            let mut s = Stream::new(4, &[a.to_bits()]);
            let (mean, var, n) = moments((0..400_000).map(|_| s.gamma(a)));
            assert!(
                (mean - a).abs() < 5.0 * (a / n).sqrt(),
                "gamma({a}) mean {mean}"
            );
            // Var of the sample variance: (mu4 - sigma^4) / n, mu4 = 3a^2 + 6a for Gamma(a).
            let sd_var = ((3.0 * a * a + 6.0 * a - a * a) / n).sqrt();
            assert!((var - a).abs() < 5.0 * sd_var, "gamma({a}) variance {var}");
        }
        let mut s = Stream::new(5, &[]);
        let (mean, _, n) = moments((0..400_000).map(|_| s.exponential()));
        assert!((mean - 1.0).abs() < 5.0 / n.sqrt());
    }

    #[test]
    fn categorical_skips_zero_weights_and_matches_frequencies() {
        let mut s = Stream::new(6, &[]);
        let w = [0.0, 2.0, 0.0, 1.0, 1.0];
        let mut counts = [0f64; 5];
        let draws = 400_000;
        for _ in 0..draws {
            counts[s.categorical(&w)] += 1.0;
        }
        assert_eq!(counts[0] + counts[2], 0.0);
        for (i, p) in [(1, 0.5), (3, 0.25), (4, 0.25)] {
            let sd = (p * (1.0 - p) / draws as f64).sqrt();
            assert!((counts[i] / draws as f64 - p).abs() < 5.0 * sd);
        }
    }

    #[test]
    fn shuffle_is_a_permutation_and_reproducible() {
        let mut v: Vec<u32> = (0..50).collect();
        Stream::new(8, &[1]).shuffle(&mut v);
        let mut w: Vec<u32> = (0..50).collect();
        Stream::new(8, &[1]).shuffle(&mut w);
        assert_eq!(v, w);
        assert_ne!(v, (0..50).collect::<Vec<_>>());
        v.sort();
        assert_eq!(v, (0..50).collect::<Vec<_>>());
    }
}
