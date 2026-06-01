//! Liouville's theorem for agent diversity — phase space volume = diversity, preserved under symplectic flow

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::phase_space::{PhasePoint, PhaseSpace};

/// Compute the phase space volume of a parallelepiped spanned by phase points.
/// For k vectors in 2n-dimensional space, this is sqrt(det(V^T V)) where V is the matrix of vectors.
pub fn parallelepiped_volume(vectors: &[DVector<f64>]) -> f64 {
    if vectors.is_empty() {
        return 0.0;
    }
    let dim = vectors[0].nrows();
    let k = vectors.len();
    let mut vmat = DMatrix::zeros(dim, k);
    for (j, v) in vectors.iter().enumerate() {
        vmat.column_mut(j).copy_from(v);
    }
    let gram = &vmat.transpose() * &vmat;
    gram.determinant().abs().sqrt()
}

/// A region in phase space for volume computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseRegion {
    /// Center of the region
    pub center: PhasePoint,
    /// Basis vectors spanning the region (as deviations from center)
    pub basis: Vec<DVector<f64>>,
}

impl PhaseRegion {
    /// Create a new phase region.
    pub fn new(center: PhasePoint, basis: Vec<DVector<f64>>) -> Self {
        let total_dim = center.total_dim();
        for b in &basis {
            assert_eq!(b.nrows(), total_dim);
        }
        Self { center, basis }
    }

    /// Volume of the region (from the parallelepiped).
    pub fn volume(&self) -> f64 {
        parallelepiped_volume(&self.basis)
    }

    /// Dimension of the region (number of basis vectors).
    pub fn region_dim(&self) -> usize {
        self.basis.len()
    }

    /// The vertices of the parallelepiped.
    pub fn vertices(&self) -> Vec<PhasePoint> {
        let dim = self.center.dim();
        let c = self.center.to_vector();
        let mut verts = vec![];
        let k = self.basis.len();
        let n_combos = 1usize << k;

        for mask in 0..n_combos {
            let mut v = c.clone();
            for i in 0..k {
                if mask & (1 << i) != 0 {
                    v += &self.basis[i];
                }
            }
            verts.push(PhasePoint::from_vector(&v, dim));
        }
        verts
    }

    /// Apply a linear transformation to the region.
    pub fn transform(&self, matrix: &DMatrix<f64>) -> PhaseRegion {
        let new_center_vec = matrix * self.center.to_vector();
        let new_center = PhasePoint::from_vector(&new_center_vec, self.center.dim());
        let new_basis: Vec<DVector<f64>> = self.basis.iter()
            .map(|b| matrix * b)
            .collect();
        PhaseRegion::new(new_center, new_basis)
    }
}

/// Agent diversity measured as phase space volume.
/// Liouville's theorem guarantees this is preserved under Hamiltonian flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDiversity {
    /// The agents' region in phase space
    pub region: PhaseRegion,
    /// Diversity = phase space volume
    pub volume: f64,
}

impl AgentDiversity {
    /// Compute diversity from a collection of agent states.
    pub fn from_agents(agents: &[PhasePoint]) -> Self {
        assert!(!agents.is_empty(), "Need at least one agent");
        let dim = agents[0].dim();

        // Center of mass
        let n = agents.len() as f64;
        let mut center_q = DVector::zeros(dim);
        let mut center_p = DVector::zeros(dim);
        for a in agents {
            center_q += &a.q;
            center_p += &a.p;
        }
        center_q /= n;
        center_p /= n;
        let center = PhasePoint::new(center_q, center_p);

        // Basis vectors as deviations from center
        let basis: Vec<DVector<f64>> = agents.iter()
            .map(|a| {
                let diff = a.add(&center.scale(-1.0));
                diff.to_vector()
            })
            .collect();

        let region = PhaseRegion::new(center, basis);
        let volume = region.volume();
        Self { region, volume }
    }

    /// Create from explicit region.
    pub fn from_region(region: PhaseRegion) -> Self {
        let volume = region.volume();
        Self { region, volume }
    }

    /// Check if diversity is preserved after transformation.
    /// Returns the ratio of new volume to old volume.
    pub fn preservation_ratio(&self, transformed: &AgentDiversity) -> f64 {
        if self.volume < 1e-15 {
            return 1.0; // Degenerate case
        }
        transformed.volume / self.volume
    }
}

/// Verify Liouville's theorem: phase space volume should be preserved.
pub fn verify_liouville(
    initial_volume: f64,
    final_volume: f64,
    tolerance: f64,
) -> bool {
    if initial_volume < 1e-15 {
        return true;
    }
    ((final_volume - initial_volume) / initial_volume).abs() < tolerance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::StormerVerlet;
    use crate::hamiltonian::SeparableHamiltonian;
    use approx::assert_relative_eq;

    #[test]
    fn test_parallelepiped_volume_unit() {
        let v1 = DVector::from_vec(vec![1.0, 0.0]);
        let v2 = DVector::from_vec(vec![0.0, 1.0]);
        let vol = parallelepiped_volume(&[v1, v2]);
        assert_relative_eq!(vol, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parallelepiped_volume_scaled() {
        let v1 = DVector::from_vec(vec![2.0, 0.0]);
        let v2 = DVector::from_vec(vec![0.0, 3.0]);
        let vol = parallelepiped_volume(&[v1, v2]);
        assert_relative_eq!(vol, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parallelepiped_volume_3d() {
        let v1 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let v2 = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        let v3 = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        let vol = parallelepiped_volume(&[v1, v2, v3]);
        assert_relative_eq!(vol, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parallelepiped_volume_empty() {
        assert_eq!(parallelepiped_volume(&[]), 0.0);
    }

    #[test]
    fn test_phase_region_creation() {
        let center = PhasePoint::new(
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![0.0]),
        );
        let basis = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let region = PhaseRegion::new(center, basis);
        assert_eq!(region.region_dim(), 2);
        assert_relative_eq!(region.volume(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_phase_region_vertices() {
        let center = PhasePoint::new(
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![0.0]),
        );
        let basis = vec![
            DVector::from_vec(vec![1.0, 0.0]),
        ];
        let region = PhaseRegion::new(center, basis);
        let verts = region.vertices();
        assert_eq!(verts.len(), 2); // 2^1 vertices
    }

    #[test]
    fn test_phase_region_transform() {
        let center = PhasePoint::new(
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![0.0]),
        );
        let basis = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let region = PhaseRegion::new(center, basis);
        let vol_before = region.volume();

        // Identity transform
        let identity = DMatrix::identity(2, 2);
        let transformed = region.transform(&identity);
        assert_relative_eq!(transformed.volume(), vol_before, epsilon = 1e-10);
    }

    #[test]
    fn test_agent_diversity_from_agents() {
        let agents = vec![
            PhasePoint::new(
                DVector::from_vec(vec![1.0, 0.0]),
                DVector::from_vec(vec![0.0, 0.0]),
            ),
            PhasePoint::new(
                DVector::from_vec(vec![-1.0, 0.0]),
                DVector::from_vec(vec![0.0, 0.0]),
            ),
        ];
        let div = AgentDiversity::from_agents(&agents);
        assert!(div.volume > 0.0);
    }

    #[test]
    fn test_diversity_preservation_ratio() {
        let div1 = AgentDiversity::from_region(PhaseRegion::new(
            PhasePoint::zeros(1),
            vec![DVector::from_vec(vec![1.0, 0.0]), DVector::from_vec(vec![0.0, 1.0])],
        ));
        let div2 = AgentDiversity::from_region(PhaseRegion::new(
            PhasePoint::zeros(1),
            vec![DVector::from_vec(vec![2.0, 0.0]), DVector::from_vec(vec![0.0, 2.0])],
        ));
        let ratio = div1.preservation_ratio(&div2);
        assert_relative_eq!(ratio, 4.0, epsilon = 1e-10); // Area scales as 2^2 = 4
    }

    #[test]
    fn test_verify_liouville() {
        assert!(verify_liouville(1.0, 1.0, 0.01));
        assert!(verify_liouville(1.0, 1.005, 0.01));
        assert!(!verify_liouville(1.0, 2.0, 0.01));
    }

    #[test]
    fn test_liouville_theorem_numerical() {
        // Verify that Störmer-Verlet approximately preserves phase space volume
        let h = SeparableHamiltonian::uniform(2, 1.0, 1.0);
        let integrator = StormerVerlet::new(0.01);

        // Create a small region
        let center = PhasePoint::new(
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        );
        let eps = 0.01;
        let basis = vec![
            DVector::from_vec(vec![eps, 0.0, 0.0, 0.0]),
            DVector::from_vec(vec![0.0, eps, 0.0, 0.0]),
            DVector::from_vec(vec![0.0, 0.0, eps, 0.0]),
            DVector::from_vec(vec![0.0, 0.0, 0.0, eps]),
        ];

        let region_initial = PhaseRegion::new(center.clone(), basis.clone());
        let vol_initial = region_initial.volume();

        // Transform all vertices through the integrator
        let verts_initial = region_initial.vertices();
        let verts_final: Vec<PhasePoint> = verts_initial.iter()
            .map(|v| integrator.integrate(v, &h, 100).state)
            .collect();

        let div_final = AgentDiversity::from_agents(&verts_final);

        // Liouville's theorem: volume should be approximately preserved
        let ratio = vol_initial / div_final.volume;
        assert!((ratio - 1.0).abs() < 0.1, "Volume ratio: {}", ratio);
    }

    #[test]
    fn test_verify_liouville_degenerate() {
        assert!(verify_liouville(0.0, 0.0, 0.01));
    }
}
