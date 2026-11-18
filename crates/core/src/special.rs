//! Log-gamma, digamma, regularized upper incomplete gamma and the chi-square survival function.

use std::f64::consts::PI;

const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// ln |Gamma(x)|: Lanczos (g = 7, n = 9), reflection below 0.5.
pub fn ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        return (PI / (PI * x).sin().abs()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = LANCZOS[0];
    let t = x + 7.5;
    for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Recurrence up to x >= 10, then the asymptotic series to x^-12.
pub fn digamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }
    if x < 0.0 {
        return digamma(1.0 - x) - PI / (PI * x).tan();
    }
    let mut x = x;
    let mut result = 0.0;
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let z = 1.0 / (x * x);
    let series = z
        * (-1.0 / 12.0
            + z * (1.0 / 120.0
                + z * (-1.0 / 252.0
                    + z * (1.0 / 240.0 + z * (-1.0 / 132.0 + z * 691.0 / 32760.0)))));
    result + x.ln() - 0.5 / x + series
}

/// Regularized upper incomplete gamma Q(a, x), a > 0, x >= 0.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x.is_nan() || a.is_nan() || a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let log_prefactor = -x + a * x.ln() - ln_gamma(a);
    if x < a + 1.0 {
        let mut ap = a;
        let mut term = 1.0 / a;
        let mut sum = term;
        for _ in 0..10_000 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * 1e-17 {
                break;
            }
        }
        1.0 - sum * log_prefactor.exp()
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-17 {
                break;
            }
        }
        log_prefactor.exp() * h
    }
}

/// P(X > x) for a chi-square with `df` degrees of freedom.
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(df / 2.0, x / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    #[test]
    fn ln_gamma_tabulated() {
        for (x, v) in [
            (0.5, 0.572_364_942_924_700_1),
            (0.1, 2.252_712_651_734_206),
            (3.0, std::f64::consts::LN_2),
            (10.0, 12.801_827_480_081_469),
            (100.0, 359.134_205_369_575_4),
            (-0.5, 1.265_512_123_484_645_4),
        ] {
            assert!(
                close(ln_gamma(x), v, 1e-12),
                "ln_gamma({x}) = {}",
                ln_gamma(x)
            );
        }
        assert!(ln_gamma(1.0).abs() < 1e-14 && ln_gamma(2.0).abs() < 1e-14);
    }

    #[test]
    fn digamma_tabulated() {
        for (x, v) in [
            (1.0, -0.577_215_664_901_532_9),
            (0.5, -1.963_510_026_021_423_5),
            (2.5, 0.703_156_640_645_243_2),
            (10.0, 2.251_752_589_066_721),
            (100.0, 4.600_161_852_738_087),
            (-0.5, 0.036_489_973_978_576_52),
        ] {
            assert!(close(digamma(x), v, 1e-12), "digamma({x}) = {}", digamma(x));
        }
        assert!(digamma(0.0).is_nan() && digamma(-2.0).is_nan());
    }

    #[test]
    fn chi2_sf_closed_forms_and_quantiles() {
        for x in [0.1f64, 1.0, 2.5, 7.0, 15.0, 40.0, 90.0] {
            let h = x / 2.0;
            let e = (-h).exp();
            assert!(close(chi2_sf(x, 2.0), e, 1e-12), "df 2 at {x}");
            assert!(close(chi2_sf(x, 4.0), e * (1.0 + h), 1e-12), "df 4 at {x}");
            assert!(
                close(chi2_sf(x, 6.0), e * (1.0 + h + h * h / 2.0), 1e-12),
                "df 6 at {x}"
            );
        }
        for (x, df) in [
            (3.841_458_820_694_124, 1.0),
            (11.070_497_693_516_351, 5.0),
            (18.307_038_053_275_146, 10.0),
        ] {
            assert!(
                close(chi2_sf(x, df), 0.05, 1e-12),
                "df {df}: {}",
                chi2_sf(x, df)
            );
        }
        assert_eq!(chi2_sf(0.0, 3.0), 1.0);
        assert_eq!(gamma_q(2.0, f64::INFINITY), 0.0);
    }
}
