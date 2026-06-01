//! Agent phase space — positions=beliefs, momenta=action tendencies, symplectic form ω

use nalgebra::{DVector, DMatrix, Dyn, Const};
use serde::{Serialize, Deserialize};

/// A point in the agent's phase space.
/// Positions (q) represent beliefs, momenta (p) represent action tendencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasePoint {
    /// Belief positions (generalized coordinates)
    pub q: DVector<f64>,
    /// Action momentum (generalized momenta)
    pub p: DVector<f64>,
}

impl PhasePoint {
    /// Create a new phase point with given positions and momenta.
    pub fn new(q: DVector<f64>, p: DVector<f64>) -> Self {
        assert_eq!(q.nrows(), p.nrows(), "Position and momentum dimensions must match");
        Self { q, p }
    }

    /// Create a zero phase point of given dimension.
    pub fn zeros(dim: usize) -> Self {
        Self {
            q: DVector::zeros(dim),
            p: DVector::zeros(dim),
        }
    }

    /// Dimension of the phase space (degrees of freedom).
    pub fn dim(&self) -> usize {
        self.q.nrows()
    }

    /// Total phase space dimension (2n for n degrees of freedom).
    pub fn total_dim(&self) -> usize {
        2 * self.dim()
    }

    /// Combine into a single vector [q; p].
    pub fn to_vector(&self) -> DVector<f64> {
        let n = self.dim();
        let mut v = DVector::zeros(2 * n);
        v.rows_mut(0, n).copy_from(&self.q);
        v.rows_mut(n, n).copy_from(&self.p);
        v
    }

    /// Extract from a combined vector [q; p].
    pub fn from_vector(v: &DVector<f64>, dim: usize) -> Self {
        Self {
            q: v.rows(0, dim).into(),
            p: v.rows(dim, dim).into(),
        }
    }

    /// Euclidean norm of the phase point.
    pub fn norm(&self) -> f64 {
        self.q.norm().hypot(self.p.norm())
    }

    /// Add two phase points.
    pub fn add(&self, other: &PhasePoint) -> PhasePoint {
        PhasePoint {
            q: &self.q + &other.q,
            p: &self.p + &other.p,
        }
    }

    /// Scale a phase point.
    pub fn scale(&self, s: f64) -> PhasePoint {
        PhasePoint {
            q: &self.q * s,
            p: &self.p * s,
        }
    }

    /// Inner product with another phase point.
    pub fn dot(&self, other: &PhasePoint) -> f64 {
        self.q.dot(&other.q) + self.p.dot(&other.p)
    }
}

/// The symplectic form ω for a 2n-dimensional phase space.
/// Represented as the canonical symplectic matrix J = [[0, I], [-I, 0]].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymplecticForm {
    /// Dimension n (phase space is 2n-dimensional)
    dim: usize,
    /// Cached symplectic matrix J
    matrix: DMatrix<f64>,
}

impl SymplecticForm {
    /// Create the canonical symplectic form for n degrees of freedom.
    pub fn canonical(dim: usize) -> Self {
        let n = dim;
        let size = 2 * n;
        let mut mat = DMatrix::zeros(size, size);

        // Top-right block: I_n
        for i in 0..n {
            mat[(i, n + i)] = 1.0;
        }
        // Bottom-left block: -I_n
        for i in 0..n {
            mat[(n + i, i)] = -1.0;
        }

        Self { dim: n, matrix: mat }
    }

    /// Get the dimension n.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the symplectic matrix J.
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Apply the symplectic form to two phase points: ω(u, v) = u^T J v.
    pub fn apply(&self, u: &PhasePoint, v: &PhasePoint) -> f64 {
        let uv = u.to_vector();
        let vv = v.to_vector();
        uv.tr_dot(&(&self.matrix * &vv))
    }

    /// Check if a matrix M preserves the symplectic form: M^T J M = J.
    pub fn is_symplectic_matrix(&self, m: &DMatrix<f64>) -> bool {
        let jtjm = m.transpose() * &self.matrix * m;
        (jtjm - &self.matrix).norm() < 1e-10
    }
}

/// The agent's phase space: a symplectic manifold equipped with ω.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseSpace {
    /// Number of degrees of freedom
    pub dim: usize,
    /// The symplectic form
    pub omega: SymplecticForm,
}

impl PhaseSpace {
    /// Create a new phase space with n degrees of freedom.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            omega: SymplecticForm::canonical(dim),
        }
    }

    /// Create a zero phase point in this space.
    pub fn origin(&self) -> PhasePoint {
        PhasePoint::zeros(self.dim)
    }

    /// Evaluate the symplectic form on two tangent vectors.
    pub fn symplectic_product(&self, u: &PhasePoint, v: &PhasePoint) -> f64 {
        self.omega.apply(u, v)
    }

    /// Compute the phase space volume element (Liouville measure).
    /// For canonical coordinates, this is ω^n / n!.
    pub fn volume_element(&self) -> f64 {
        1.0 // Canonical form has unit volume element
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;

    #[test]
    fn test_phase_point_creation() {
        let q = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let p = DVector::from_vec(vec![4.0, 5.0, 6.0]);
        let pt = PhasePoint::new(q, p);
        assert_eq!(pt.dim(), 3);
        assert_eq!(pt.total_dim(), 6);
    }

    #[test]
    fn test_phase_point_zeros() {
        let pt = PhasePoint::zeros(4);
        assert_eq!(pt.dim(), 4);
        assert_eq!(pt.q.norm(), 0.0);
        assert_eq!(pt.p.norm(), 0.0);
    }

    #[test]
    fn test_phase_point_vector_roundtrip() {
        let q = DVector::from_vec(vec![1.0, 2.0]);
        let p = DVector::from_vec(vec![3.0, 4.0]);
        let pt = PhasePoint::new(q.clone(), p.clone());
        let v = pt.to_vector();
        assert_eq!(v.len(), 4);
        let pt2 = PhasePoint::from_vector(&v, 2);
        assert!((pt2.q - q).norm() < 1e-10);
        assert!((pt2.p - p).norm() < 1e-10);
    }

    #[test]
    fn test_phase_point_add() {
        let pt1 = PhasePoint::new(DVector::from_vec(vec![1.0]), DVector::from_vec(vec![2.0]));
        let pt2 = PhasePoint::new(DVector::from_vec(vec![3.0]), DVector::from_vec(vec![4.0]));
        let sum = pt1.add(&pt2);
        assert_eq!(sum.q[0], 4.0);
        assert_eq!(sum.p[0], 6.0);
    }

    #[test]
    fn test_phase_point_scale() {
        let pt = PhasePoint::new(DVector::from_vec(vec![2.0]), DVector::from_vec(vec![3.0]));
        let scaled = pt.scale(2.0);
        assert_eq!(scaled.q[0], 4.0);
        assert_eq!(scaled.p[0], 6.0);
    }

    #[test]
    fn test_symplectic_form_canonical() {
        let omega = SymplecticForm::canonical(2);
        let mat = omega.matrix();
        assert_eq!(mat.nrows(), 4);
        assert_eq!(mat.ncols(), 4);
        // J should be antisymmetric
        assert!((mat + mat.transpose()).norm() < 1e-10);
        // J^2 = -I
        let j2 = mat * mat;
        let neg_id = -DMatrix::identity(4, 4);
        assert!((j2 - neg_id).norm() < 1e-10);
    }

    #[test]
    fn test_symplectic_product() {
        let omega = SymplecticForm::canonical(2);
        let u = PhasePoint::new(
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 0.0]),
        );
        let v = PhasePoint::new(
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
        );
        // ω(q1, p1) = q1^T J p1 = 1
        let prod = omega.apply(&u, &v);
        assert!((prod - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_symplectic_form_skew_symmetry() {
        let omega = SymplecticForm::canonical(3);
        let u = PhasePoint::new(
            DVector::from_vec(vec![1.0, 2.0, 3.0]),
            DVector::from_vec(vec![4.0, 5.0, 6.0]),
        );
        let v = PhasePoint::new(
            DVector::from_vec(vec![7.0, 8.0, 9.0]),
            DVector::from_vec(vec![10.0, 11.0, 12.0]),
        );
        let prod_uv = omega.apply(&u, &v);
        let prod_vu = omega.apply(&v, &u);
        assert!((prod_uv + prod_vu).abs() < 1e-10); // ω(u,v) = -ω(v,u)
    }

    #[test]
    fn test_is_symplectic_matrix() {
        let omega = SymplecticForm::canonical(2);
        // Identity is symplectic
        let identity = DMatrix::identity(4, 4);
        assert!(omega.is_symplectic_matrix(&identity));
        // J itself is symplectic
        assert!(omega.is_symplectic_matrix(omega.matrix()));
    }

    #[test]
    fn test_phase_space_creation() {
        let ps = PhaseSpace::new(3);
        assert_eq!(ps.dim, 3);
        let origin = ps.origin();
        assert_eq!(origin.dim(), 3);
    }

    #[test]
    fn test_phase_point_norm() {
        let pt = PhasePoint::new(
            DVector::from_vec(vec![3.0, 4.0]),
            DVector::zeros(2),
        );
        assert!((pt.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_point_dot() {
        let pt1 = PhasePoint::new(
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        );
        let pt2 = PhasePoint::new(
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        );
        assert!((pt1.dot(&pt2) - 2.0).abs() < 1e-10);
    }
}
