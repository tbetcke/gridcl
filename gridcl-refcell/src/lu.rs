//! Dense LU factorisation with partial pivoting, so gridcl-refcell needs no linear-algebra
//! dependency (docs/design.md, "Tabulation and quadrature").

use crate::RefCellError;

/// The LU factors of a square matrix, P A = L U, stored in place (row-major): L below the
/// diagonal with a unit diagonal, U on and above it.
#[derive(Clone, Debug)]
pub(crate) struct Lu {
    n: usize,
    lu: Vec<f64>,
    /// Row k of P A is row `perm[k]` of A.
    perm: Vec<usize>,
}

impl Lu {
    /// Factors the n × n row-major matrix `a`.
    ///
    /// # Errors
    ///
    /// [`RefCellError::Singular`] if a pivot is zero or at most n · ε times the largest entry
    /// of `a` in magnitude: the matrix is singular to working precision.
    ///
    /// # Panics
    ///
    /// If `a.len() != n * n`.
    pub(crate) fn factor(mut a: Vec<f64>, n: usize) -> Result<Self, RefCellError> {
        assert_eq!(a.len(), n * n, "Lu::factor: expected an {n} x {n} matrix");
        let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let tiny = n as f64 * f64::EPSILON * scale;
        let mut perm: Vec<usize> = (0..n).collect();
        for k in 0..n {
            // The pivot: the largest entry of column k on or below the diagonal.
            let (p, max) = (k..n)
                .map(|i| (i, a[i * n + k].abs()))
                .fold((k, -1.0), |best, c| if c.1 > best.1 { c } else { best });
            if max <= tiny || max == 0.0 {
                return Err(RefCellError::Singular);
            }
            if p != k {
                for j in 0..n {
                    a.swap(k * n + j, p * n + j);
                }
                perm.swap(k, p);
            }
            let pivot = a[k * n + k];
            for i in k + 1..n {
                let l = a[i * n + k] / pivot;
                a[i * n + k] = l;
                if l != 0.0 {
                    for j in k + 1..n {
                        a[i * n + j] -= l * a[k * n + j];
                    }
                }
            }
        }
        Ok(Self { n, lu: a, perm })
    }

    /// Solves A X = B in place for the n × m row-major right-hand sides `b`.
    ///
    /// # Panics
    ///
    /// If `b.len()` is not a multiple of n.
    pub(crate) fn solve(&self, b: &mut [f64]) {
        let n = self.n;
        assert!(
            n > 0 && b.len().is_multiple_of(n),
            "Lu::solve: right-hand sides do not have {n} rows"
        );
        let m = b.len() / n;
        // Permute: row k of P B.
        let mut x: Vec<f64> = self
            .perm
            .iter()
            .flat_map(|&r| b[r * m..(r + 1) * m].to_vec())
            .collect();
        // Forward substitution with the unit lower triangle.
        for i in 0..n {
            for k in 0..i {
                let l = self.lu[i * n + k];
                if l != 0.0 {
                    for j in 0..m {
                        x[i * m + j] -= l * x[k * m + j];
                    }
                }
            }
        }
        // Back substitution with the upper triangle.
        for i in (0..n).rev() {
            for k in i + 1..n {
                let u = self.lu[i * n + k];
                if u != 0.0 {
                    for j in 0..m {
                        x[i * m + j] -= u * x[k * m + j];
                    }
                }
            }
            let d = self.lu[i * n + i];
            for j in 0..m {
                x[i * m + j] /= d;
            }
        }
        b.copy_from_slice(&x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matvec(a: &[f64], x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
            .collect()
    }

    #[test]
    fn solves_a_known_system() {
        // A x = b with x = (1, 2, 3).
        let a = vec![4.0, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 4.0];
        let mut b = matvec(&a, &[1.0, 2.0, 3.0]);
        Lu::factor(a, 3).unwrap().solve(&mut b);
        for (got, want) in b.iter().zip([1.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-15, "{b:?}");
        }
    }

    #[test]
    fn pivots_on_a_zero_leading_entry() {
        let a = vec![0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 4.0, -3.0, 8.0];
        let x = [-1.0, 0.5, 2.0];
        let mut b = matvec(&a, &x);
        Lu::factor(a, 3).unwrap().solve(&mut b);
        for (got, want) in b.iter().zip(x) {
            assert!((got - want).abs() < 1e-14, "{b:?}");
        }
    }

    #[test]
    fn several_right_hand_sides_give_the_inverse() {
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let mut b = vec![1.0, 0.0, 0.0, 1.0];
        Lu::factor(a, 2).unwrap().solve(&mut b);
        let want = [0.6, -0.2, -0.2, 0.4];
        for (got, w) in b.iter().zip(want) {
            assert!((got - w).abs() < 1e-15, "{b:?}");
        }
    }

    #[test]
    fn rank_deficient_is_singular() {
        // The third row is the sum of the first two.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 9.0];
        assert_eq!(Lu::factor(a, 3).unwrap_err(), RefCellError::Singular);
        assert_eq!(
            Lu::factor(vec![0.0; 4], 2).unwrap_err(),
            RefCellError::Singular
        );
    }
}
