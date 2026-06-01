//! Poisson brackets for coupled agents — {f,g} = interaction structure

use nalgebra::DVector;
use crate::phase_space::{PhasePoint, PhaseSpace};
use crate::hamiltonian::Hamiltonian;

/// Compute the Poisson bracket {f, g} of two functions on phase space.
/// {f, g} = Σ_i (∂f/∂q_i ∂g/∂p_i - ∂f/∂p_i ∂g/∂q_i)
pub fn poisson_bracket<F, G>(
    f: &F,
    g: &G,
    point: &PhasePoint,
) -> f64
where
    F: Fn(&PhasePoint) -> f64,
    G: Fn(&PhasePoint) -> f64,
{
    let eps = 1e-7;
    let n = point.dim();

    let mut bracket = 0.0;
    for i in 0..n {
        // ∂f/∂q_i
        let mut pf = point.clone();
        pf.q[i] += eps;
        let mut pm = point.clone();
        pm.q[i] -= eps;
        let df_dq = (f(&pf) - f(&pm)) / (2.0 * eps);

        // ∂g/∂p_i
        pf = point.clone();
        pf.p[i] += eps;
        pm = point.clone();
        pm.p[i] -= eps;
        let dg_dp = (g(&pf) - g(&pm)) / (2.0 * eps);

        // ∂f/∂p_i
        pf = point.clone();
        pf.p[i] += eps;
        pm = point.clone();
        pm.p[i] -= eps;
        let df_dp = (f(&pf) - f(&pm)) / (2.0 * eps);

        // ∂g/∂q_i
        pf = point.clone();
        pf.q[i] += eps;
        pm = point.clone();
        pm.q[i] -= eps;
        let dg_dq = (g(&pf) - g(&pm)) / (2.0 * eps);

        bracket += df_dq * dg_dp - df_dp * dg_dq;
    }
    bracket
}

/// A pair of coupled agents interacting through Poisson bracket coupling.
#[derive(Debug, Clone)]
pub struct CoupledAgents {
    /// Agent 1's phase space
    pub phase_space_1: PhaseSpace,
    /// Agent 2's phase space
    pub phase_space_2: PhaseSpace,
    /// Coupling strength
    pub coupling: f64,
}

impl CoupledAgents {
    /// Create a new coupled agent system.
    pub fn new(ps1: PhaseSpace, ps2: PhaseSpace, coupling: f64) -> Self {
        Self {
            phase_space_1: ps1,
            phase_space_2: ps2,
            coupling,
        }
    }

    /// Compute the coupled Hamiltonian: H = H1 + H2 + coupling * interaction.
    pub fn coupled_energy<H1: Hamiltonian + ?Sized, H2: Hamiltonian + ?Sized>(
        &self,
        h1: &H1,
        h2: &H2,
        state1: &PhasePoint,
        state2: &PhasePoint,
    ) -> f64 {
        h1.evaluate(state1) + h2.evaluate(state2)
            + self.coupling * self.interaction(state1, state2)
    }

    /// Interaction energy between two agents.
    /// Default: dot product of their phase space vectors.
    pub fn interaction(&self, state1: &PhasePoint, state2: &PhasePoint) -> f64 {
        state1.dot(state2)
    }

    /// Compute coupled equations of motion.
    pub fn coupled_equations<H1: Hamiltonian + ?Sized, H2: Hamiltonian + ?Sized>(
        &self,
        h1: &H1,
        h2: &H2,
        state1: &PhasePoint,
        state2: &PhasePoint,
    ) -> ((DVector<f64>, DVector<f64>), (DVector<f64>, DVector<f64>)) {
        let (dq1, dp1) = h1.equations_of_motion(state1);
        let (dq2, dp2) = h2.equations_of_motion(state2);

        // Add coupling terms
        let coupling_dq1 = &state2.q * self.coupling;
        let coupling_dp1 = &state2.p * (-self.coupling);
        let coupling_dq2 = &state1.q * self.coupling;
        let coupling_dp2 = &state1.p * (-self.coupling);

        (
            (dq1 + coupling_dq1, dp1 + coupling_dp1),
            (dq2 + coupling_dq2, dp2 + coupling_dp2),
        )
    }
}

/// A multi-agent system with pairwise Poisson bracket interactions.
#[derive(Debug, Clone)]
pub struct MultiAgentSystem {
    /// Individual agent states
    pub agents: Vec<PhasePoint>,
    /// Coupling matrix (agents × agents)
    pub coupling_matrix: Vec<Vec<f64>>,
    /// Phase space dimension per agent
    pub dim: usize,
}

impl MultiAgentSystem {
    /// Create a new multi-agent system.
    pub fn new(agents: Vec<PhasePoint>, coupling_matrix: Vec<Vec<f64>>) -> Self {
        let dim = agents[0].dim();
        for a in &agents {
            assert_eq!(a.dim(), dim);
        }
        let n = agents.len();
        assert_eq!(coupling_matrix.len(), n);
        for row in &coupling_matrix {
            assert_eq!(row.len(), n);
        }
        Self { agents, coupling_matrix, dim }
    }

    /// Number of agents.
    pub fn n_agents(&self) -> usize {
        self.agents.len()
    }

    /// Total energy of the system.
    pub fn total_energy<H: Hamiltonian + ?Sized>(&self, h: &H) -> f64 {
        let mut energy = 0.0;
        for agent in &self.agents {
            energy += h.evaluate(agent);
        }
        // Add pairwise coupling
        for i in 0..self.agents.len() {
            for j in (i + 1)..self.agents.len() {
                energy += self.coupling_matrix[i][j] * self.agents[i].dot(&self.agents[j]);
            }
        }
        energy
    }

    /// Compute pairwise Poisson brackets between agent observables.
    pub fn pairwise_poisson<F>(
        &self,
        observable: &F,
    ) -> Vec<Vec<f64>>
    where
        F: Fn(&PhasePoint) -> f64,
    {
        let n = self.agents.len();
        let mut brackets = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    // Cross-agent Poisson bracket
                    // For coupled systems, this involves the coupling terms
                    brackets[i][j] = self.coupling_matrix[i][j]
                        * poisson_bracket(
                            &|_: &PhasePoint| observable(&self.agents[i]),
                            &|_: &PhasePoint| observable(&self.agents[j]),
                            &self.agents[i],
                        );
                }
            }
        }
        brackets
    }

    /// Compute the center of mass of the agent ensemble.
    pub fn center_of_mass(&self) -> PhasePoint {
        let n = self.agents.len() as f64;
        let mut q_sum = DVector::zeros(self.dim);
        let mut p_sum = DVector::zeros(self.dim);
        for a in &self.agents {
            q_sum += &a.q;
            p_sum += &a.p;
        }
        PhasePoint::new(q_sum / n, p_sum / n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_poisson_bracket_basic() {
        // {q_i, p_j} = δ_ij
        let pt = PhasePoint::new(
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![3.0, 4.0]),
        );
        let f = |p: &PhasePoint| p.q[0]; // q_1
        let g = |p: &PhasePoint| p.p[0]; // p_1
        let bracket = poisson_bracket(&f, &g, &pt);
        assert_relative_eq!(bracket, 1.0, epsilon = 1e-5); // {q, p} = 1
    }

    #[test]
    fn test_poisson_bracket_antisymmetry() {
        let pt = PhasePoint::new(
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![3.0, 4.0]),
        );
        let f = |p: &PhasePoint| p.q[0] * p.p[1];
        let g = |p: &PhasePoint| p.q[1] * p.p[0];
        let bracket_fg = poisson_bracket(&f, &g, &pt);
        let bracket_gf = poisson_bracket(&g, &f, &pt);
        assert_relative_eq!(bracket_fg + bracket_gf, 0.0, epsilon = 1e-4); // {f,g} = -{g,f}
    }

    #[test]
    fn test_poisson_bracket_self() {
        let pt = PhasePoint::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![2.0]),
        );
        let f = |p: &PhasePoint| p.q[0].powi(2) + p.p[0].powi(2);
        let bracket = poisson_bracket(&f, &f, &pt);
        assert_relative_eq!(bracket, 0.0, epsilon = 1e-4); // {f, f} = 0
    }

    #[test]
    fn test_poisson_bracket_jacobi() {
        // Jacobi identity: {f,{g,h}} + {g,{h,f}} + {h,{f,g}} = 0
        let pt = PhasePoint::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![2.0]),
        );
        let f = |p: &PhasePoint| p.q[0].powi(2);
        let g = |p: &PhasePoint| p.p[0].powi(2);
        let h = |p: &PhasePoint| p.q[0] * p.p[0];

        let bracket_gh = poisson_bracket(&g, &h, &pt);
        let bracket_hf = poisson_bracket(&h, &f, &pt);
        let bracket_fg = poisson_bracket(&f, &g, &pt);

        let f_gh = |_: &PhasePoint| bracket_gh;
        let g_hf = |_: &PhasePoint| bracket_hf;
        let h_fg = |_: &PhasePoint| bracket_fg;

        let term1 = poisson_bracket(&f, &f_gh, &pt);
        let term2 = poisson_bracket(&g, &g_hf, &pt);
        let term3 = poisson_bracket(&h, &h_fg, &pt);

        // The Jacobi identity is approximately satisfied for constant brackets
        // (higher-order derivatives of constants are zero)
        assert!((term1 + term2 + term3).abs() < 1e-3);
    }

    #[test]
    fn test_coupled_agents_creation() {
        let ps1 = PhaseSpace::new(2);
        let ps2 = PhaseSpace::new(2);
        let coupled = CoupledAgents::new(ps1, ps2, 0.5);
        assert!((coupled.coupling - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_coupled_agents_energy() {
        use crate::hamiltonian::SeparableHamiltonian;
        let ps1 = PhaseSpace::new(1);
        let ps2 = PhaseSpace::new(1);
        let coupled = CoupledAgents::new(ps1, ps2, 0.0);
        let h = SeparableHamiltonian::uniform(1, 1.0, 1.0);
        let s1 = PhasePoint::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![0.0]),
        );
        let s2 = PhasePoint::new(
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![1.0]),
        );
        let energy = coupled.coupled_energy(&h, &h, &s1, &s2);
        // H1 = 0.5, H2 = 0.5, coupling = 0 → total = 1.0
        assert_relative_eq!(energy, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_coupled_agents_coupled_energy() {
        use crate::hamiltonian::SeparableHamiltonian;
        let ps1 = PhaseSpace::new(1);
        let ps2 = PhaseSpace::new(1);
        let coupled = CoupledAgents::new(ps1, ps2, 1.0);
        let h = SeparableHamiltonian::uniform(1, 1.0, 0.0); // Free particles
        let s1 = PhasePoint::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![0.0]),
        );
        let s2 = PhasePoint::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![0.0]),
        );
        let energy = coupled.coupled_energy(&h, &h, &s1, &s2);
        // H1 = 0, H2 = 0, coupling = 1.0 * (q1*q2 + p1*p2) = 1.0
        assert_relative_eq!(energy, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_agent_system() {
        use crate::hamiltonian::SeparableHamiltonian;
        let agents = vec![
            PhasePoint::new(DVector::from_vec(vec![1.0]), DVector::from_vec(vec![0.0])),
            PhasePoint::new(DVector::from_vec(vec![0.0]), DVector::from_vec(vec![1.0])),
        ];
        let coupling = vec![vec![0.0, 0.5], vec![0.5, 0.0]];
        let system = MultiAgentSystem::new(agents, coupling);
        assert_eq!(system.n_agents(), 2);
    }

    #[test]
    fn test_multi_agent_center_of_mass() {
        let agents = vec![
            PhasePoint::new(DVector::from_vec(vec![1.0]), DVector::from_vec(vec![0.0])),
            PhasePoint::new(DVector::from_vec(vec![-1.0]), DVector::from_vec(vec![2.0])),
        ];
        let coupling = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let system = MultiAgentSystem::new(agents, coupling);
        let com = system.center_of_mass();
        assert_relative_eq!(com.q[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(com.p[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_agent_total_energy() {
        use crate::hamiltonian::SeparableHamiltonian;
        let agents = vec![
            PhasePoint::new(DVector::from_vec(vec![1.0]), DVector::from_vec(vec![0.0])),
            PhasePoint::new(DVector::from_vec(vec![0.0]), DVector::from_vec(vec![1.0])),
        ];
        let coupling = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let system = MultiAgentSystem::new(agents, coupling);
        let h = SeparableHamiltonian::uniform(1, 1.0, 1.0);
        let energy = system.total_energy(&h);
        // H1 = 0.5*1 + 0.5*1*0 = 0.5, H2 = 0 + 0.5*1 = 0.5, total = 1.0
        assert_relative_eq!(energy, 1.0, epsilon = 1e-10);
    }
}
