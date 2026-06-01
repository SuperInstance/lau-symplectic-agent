//! Symplectic decision making — decisions are canonical transformations preserving ω

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};
use crate::phase_space::{PhasePoint, PhaseSpace, SymplecticForm};
use crate::hamiltonian::Hamiltonian;

/// A canonical (symplectic) transformation of phase space.
/// These preserve the symplectic form: M^T J M = J.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalTransformation {
    /// The transformation matrix (must be symplectic)
    matrix: DMatrix<f64>,
    /// Optional translation (affine part)
    offset: Option<PhasePoint>,
    /// Dimension of phase space
    dim: usize,
}

impl CanonicalTransformation {
    /// Create a new canonical transformation from a matrix.
    /// Does NOT check symplecticity — use validate() for that.
    pub fn new(matrix: DMatrix<f64>, dim: usize) -> Self {
        assert_eq!(matrix.nrows(), 2 * dim);
        assert_eq!(matrix.ncols(), 2 * dim);
        Self { matrix, offset: None, dim }
    }

    /// Create with offset.
    pub fn with_offset(mut self, offset: PhasePoint) -> Self {
        assert_eq!(offset.dim(), self.dim);
        self.offset = Some(offset);
        self
    }

    /// Identity transformation.
    pub fn identity(dim: usize) -> Self {
        Self {
            matrix: DMatrix::identity(2 * dim, 2 * dim),
            offset: None,
            dim,
        }
    }

    /// Compose two canonical transformations: T2 ∘ T1.
    pub fn compose(&self, other: &CanonicalTransformation) -> CanonicalTransformation {
        assert_eq!(self.dim, other.dim);
        let new_matrix = &other.matrix * &self.matrix;
        let new_offset = match (&self.offset, &other.offset) {
            (None, None) => None,
            (Some(off), None) => Some(off.clone()),
            (None, Some(off)) => Some(off.clone()),
            (Some(off1), Some(off2)) => {
                let v1 = off1.to_vector();
                let transformed = &other.matrix * v1;
                let combined = transformed + off2.to_vector();
                Some(PhasePoint::from_vector(&combined, self.dim))
            }
        };
        CanonicalTransformation { matrix: new_matrix, offset: new_offset, dim: self.dim }
    }

    /// Apply the transformation to a phase point.
    pub fn apply(&self, point: &PhasePoint) -> PhasePoint {
        let v = point.to_vector();
        let transformed = &self.matrix * v;
        let result = match &self.offset {
            Some(off) => transformed + off.to_vector(),
            None => transformed,
        };
        PhasePoint::from_vector(&result, self.dim)
    }

    /// Validate that this is truly a symplectic transformation.
    pub fn validate(&self) -> bool {
        let omega = SymplecticForm::canonical(self.dim);
        omega.is_symplectic_matrix(&self.matrix)
    }

    /// Inverse transformation.
    pub fn inverse(&self) -> CanonicalTransformation {
        let inv_matrix = self.matrix.clone().try_inverse()
            .expect("Transformation matrix must be invertible");
        let inv_offset = self.offset.as_ref().map(|off| {
            let neg = -off.to_vector();
            let transformed = &inv_matrix * neg;
            PhasePoint::from_vector(&transformed, self.dim)
        });
        CanonicalTransformation {
            matrix: inv_matrix,
            offset: inv_offset,
            dim: self.dim,
        }
    }

    /// Get the matrix.
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Get the dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// A symplectic decision: applying a canonical transformation to agent state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymplecticDecision {
    /// The transformation
    pub transformation: CanonicalTransformation,
    /// Decision confidence (0 to 1)
    pub confidence: f64,
    /// Decision label
    pub label: String,
}

impl SymplecticDecision {
    /// Make a symplectic decision by applying the transformation.
    pub fn execute(&self, state: &PhasePoint) -> PhasePoint {
        self.transformation.apply(state)
    }

    /// Create a decision from a Hamiltonian flow for time dt.
    /// For a separable H, this constructs the flow map approximately.
    pub fn from_hamiltonian_flow<H: Hamiltonian + ?Sized>(
        h: &H,
        state: &PhasePoint,
        dt: f64,
        label: String,
    ) -> Self {
        let dim = state.dim();
        // Compute the linearized flow using Jacobian of equations of motion
        let (dqdt, dpdt) = h.equations_of_motion(state);
        let eps = 1e-7;
        let size = 2 * dim;
        let mut flow = DMatrix::identity(size, size);

        // Approximate the flow matrix via finite differences of the vector field
        let f0 = {
            let mut v = DVector::zeros(size);
            v.rows_mut(0, dim).copy_from(&dqdt);
            v.rows_mut(dim, dim).copy_from(&dpdt);
            v
        };

        for j in 0..size {
            let mut state_plus = state.to_vector();
            state_plus[j] += eps;
            let pt_plus = PhasePoint::from_vector(&state_plus, dim);
            let (dq_plus, dp_plus) = h.equations_of_motion(&pt_plus);

            let mut f_plus = DVector::zeros(size);
            f_plus.rows_mut(0, dim).copy_from(&dq_plus);
            f_plus.rows_mut(dim, dim).copy_from(&dp_plus);

            let df = (f_plus - &f0) / eps;
            // Flow: x(dt) ≈ x(0) + dt * f(x(0)) + higher order
            // The Jacobian contribution to the flow matrix
            for i in 0..size {
                flow[(i, j)] += dt * df[i];
            }
        }

        let confidence = 1.0 / (1.0 + dt * dt); // decreases with larger time steps
        let transformation = CanonicalTransformation::new(flow, dim);

        SymplecticDecision {
            transformation,
            confidence,
            label,
        }
    }
}

/// Batch of symplectic decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionBatch {
    pub decisions: Vec<SymplecticDecision>,
}

impl DecisionBatch {
    /// Execute all decisions in sequence.
    pub fn execute(&self, state: &PhasePoint) -> PhasePoint {
        self.decisions.iter().fold(state.clone(), |s, d| d.execute(&s))
    }

    /// Check if all decisions preserve the symplectic form.
    pub fn all_symplectic(&self) -> bool {
        self.decisions.iter().all(|d| d.transformation.validate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonian::SeparableHamiltonian;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity_transformation() {
        let t = CanonicalTransformation::identity(2);
        assert!(t.validate());
        let pt = PhasePoint::new(
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![3.0, 4.0]),
        );
        let result = t.apply(&pt);
        assert!((result.q - pt.q).norm() < 1e-10);
        assert!((result.p - pt.p).norm() < 1e-10);
    }

    #[test]
    fn test_identity_compose() {
        let id = CanonicalTransformation::identity(2);
        let t = CanonicalTransformation::identity(2);
        let composed = id.compose(&t);
        assert!(composed.validate());
    }

    #[test]
    fn test_symplectic_decision_execute() {
        let id = CanonicalTransformation::identity(1);
        let decision = SymplecticDecision {
            transformation: id,
            confidence: 0.95,
            label: "test".to_string(),
        };
        let pt = PhasePoint::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![2.0]),
        );
        let result = decision.execute(&pt);
        assert!((result.q[0] - 1.0).abs() < 1e-10);
        assert!((result.p[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_decision_batch() {
        let batch = DecisionBatch {
            decisions: vec![
                SymplecticDecision {
                    transformation: CanonicalTransformation::identity(1),
                    confidence: 0.9,
                    label: "step1".to_string(),
                },
                SymplecticDecision {
                    transformation: CanonicalTransformation::identity(1),
                    confidence: 0.8,
                    label: "step2".to_string(),
                },
            ],
        };
        let pt = PhasePoint::new(
            DVector::from_vec(vec![5.0]),
            DVector::from_vec(vec![3.0]),
        );
        let result = batch.execute(&pt);
        assert!((result.q[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_transformation() {
        let id = CanonicalTransformation::identity(2);
        let inv = id.inverse();
        assert!(inv.validate());
        let pt = PhasePoint::new(
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![3.0, 4.0]),
        );
        let roundtrip = id.apply(&inv.apply(&pt));
        assert!((roundtrip.q - pt.q).norm() < 1e-10);
        assert!((roundtrip.p - pt.p).norm() < 1e-10);
    }

    #[test]
    fn test_from_hamiltonian_flow() {
        let h = SeparableHamiltonian::uniform(2, 1.0, 1.0);
        let pt = PhasePoint::new(
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        );
        let decision = SymplecticDecision::from_hamiltonian_flow(
            &h, &pt, 0.01, "flow".to_string(),
        );
        assert_eq!(decision.label, "flow");
        assert!(decision.confidence > 0.0 && decision.confidence <= 1.0);
    }

    #[test]
    fn test_symplectic_rotation() {
        // A rotation in phase space is symplectic: [[cos θ, sin θ], [-sin θ, cos θ]]
        let theta = std::f64::consts::PI / 4.0;
        let dim = 1;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let mat = DMatrix::from_row_slice(2, 2, &[
            cos_t, sin_t,
            -sin_t, cos_t,
        ]);
        let t = CanonicalTransformation::new(mat, dim);
        assert!(t.validate());
    }

    #[test]
    fn test_transformation_with_offset() {
        let t = CanonicalTransformation::identity(1)
            .with_offset(PhasePoint::new(
                DVector::from_vec(vec![1.0]),
                DVector::from_vec(vec![2.0]),
            ));
        let pt = PhasePoint::new(
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![0.0]),
        );
        let result = t.apply(&pt);
        assert!((result.q[0] - 1.0).abs() < 1e-10);
        assert!((result.p[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_decision_batch_all_symplectic() {
        let batch = DecisionBatch {
            decisions: vec![
                SymplecticDecision {
                    transformation: CanonicalTransformation::identity(1),
                    confidence: 1.0,
                    label: "a".to_string(),
                },
            ],
        };
        assert!(batch.all_symplectic());
    }
}
