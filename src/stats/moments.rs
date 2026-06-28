//! Numerically-stable online moments (Welford + higher-order M2/M3/M4).
//!
//! All accumulation is in f64. We track the central moments M2..M4 with the
//! standard streaming update so mean/variance/kurtosis are computed in a single
//! pass without catastrophic cancellation.

#[derive(Debug, Clone, Default)]
pub struct Moments {
    n: u64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl Moments {
    pub fn new() -> Self {
        Moments::default()
    }

    /// Welford / Terriberry update for the first four central moments.
    pub fn push(&mut self, x: f64) {
        let n1 = self.n as f64;
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;
        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Population variance.
    pub fn variance(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.m2 / self.n as f64
        }
    }

    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Sum of squares (Σx²), recovered from the central moment: Σx² = M2 + n·mean².
    pub fn sum_squares(&self) -> f64 {
        self.m2 + self.n as f64 * self.mean * self.mean
    }

    /// Excess kurtosis (0 for a normal distribution). Returns 0 when undefined.
    pub fn excess_kurtosis(&self) -> f64 {
        if self.n == 0 || self.m2 == 0.0 {
            return 0.0;
        }
        let n = self.n as f64;
        n * self.m4 / (self.m2 * self.m2) - 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_variance() {
        let mut m = Moments::new();
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            m.push(x);
        }
        assert!((m.mean() - 5.0).abs() < 1e-9);
        // population variance of this classic set is 4.0
        assert!((m.variance() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn kurtosis_normal_ish_is_small() {
        // Symmetric, light-tailed-ish data -> excess kurtosis well below a
        // heavy-tailed spike.
        let mut m = Moments::new();
        for x in [-1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0] {
            m.push(x);
        }
        assert!(m.excess_kurtosis() < 1.0);
    }
}
