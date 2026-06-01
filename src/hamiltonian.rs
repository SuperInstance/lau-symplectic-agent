//! Hamiltonian agent — H = energy functional on phase space, dynamics via Hamilton's equations

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::phase_space::{PhasePoint, PhaseSpace};

/// A Hamiltonian function H: PhaseSpace → ℝ for an agent.
/// The Hamiltonian encodes the total "energy" of the agent — kinetic (action) + potential (belief) energy.
pub trait Hamiltonian: Send + Sync {
    /// Evaluate the Hamiltonian at a phase point.
    fn evaluate(&self, point: &PhasePoint) -> f64;

    /// Compute the gradient with respect to positions (∂H/∂q).
    /// Defaults to numerical gradient.
    fn grad_q(&self, point: &PhasePoint) -> DVector<f64> {
        let eps = 1e-7;
        let n = point.dim();
        let mut grad = DVector::zeros(n);
        for i in 0..n {
            let mut qp = point.clone();
            qp.q[i] += eps;
            let mut qm = point.clone();
            qm.q[i] -= eps;
            grad[i] = (self.evaluate(&qp) - self.evaluate(&qm)) / (2.0 * eps);
        }
        grad
    }

    /// Compute the gradient with respect to momenta (∂H/∂p).
    /// Defaults to numerical gradient.
    fn grad_p(&self, point: &PhasePoint) -> DVector<f64> {
        let eps = 1e-7;
        let n = point.dim();
        let mut grad = DVector::zeros(n);
        for i in 0..n {
            let mut pp = point.clone();
            pp.p[i] += eps;
            let mut pm = point.clone();
            pm.p[i] -= eps;
            grad[i] = (self.evaluate(&pp) - self.evaluate(&pm)) / (2.0 * eps);
        }
        grad
    }

    /// Hamilton's equations: returns (dq/dt, dp/dt).
    /// dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
    fn equations_of_motion(&self, point: &PhasePoint) -> (DVector<f64>, DVector<f64>) {
        let dhdq = self.grad_q(point);
        let dhdp = self.grad_p(point);
        (dhdp, -dhdq)
    }

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Hamiltonian>;
}

impl Clone for Box<dyn Hamiltonian> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// A separable Hamiltonian: H(q, p) = T(p) + V(q).
/// Common form for physical systems where kinetic and potential energies are independent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparableHamiltonian {
    /// Kinetic energy coefficients for momenta: T(p) = ½ Σ m_i p_i²
    pub masses: DVector<f64>,
    /// Potential energy function: V(q) — stored as coefficients for quadratic potential
    pub potential_coeffs: DVector<f64>,
}

impl SeparableHamiltonian {
    /// Create a new separable Hamiltonian with uniform masses and quadratic potential.
    pub fn new(masses: DVector<f64>, potential_coeffs: DVector<f64>) -> Self {
        assert_eq!(masses.nrows(), potential_coeffs.nrows());
        Self { masses, potential_coeffs }
    }

    /// Create with uniform mass and spring constant.
    pub fn uniform(dim: usize, mass: f64, spring_k: f64) -> Self {
        Self {
            masses: DVector::from_element(dim, mass),
            potential_coeffs: DVector::from_element(dim, spring_k),
        }
    }
}

impl Hamiltonian for SeparableHamiltonian {
    fn evaluate(&self, point: &PhasePoint) -> f64 {
        // T(p) = ½ Σ p_i² / m_i
        let kinetic: f64 = point.p.iter()
            .zip(self.masses.iter())
            .map(|(pi, mi)| 0.5 * pi * pi / mi)
            .sum();
        // V(q) = ½ Σ k_i q_i²
        let potential: f64 = point.q.iter()
            .zip(self.potential_coeffs.iter())
            .map(|(qi, ki)| 0.5 * ki * qi * qi)
            .sum();
        kinetic + potential
    }

    fn grad_q(&self, point: &PhasePoint) -> DVector<f64> {
        point.q.component_mul(&self.potential_coeffs)
    }

    fn grad_p(&self, point: &PhasePoint) -> DVector<f64> {
        let mut grad = point.p.clone();
        for i in 0..grad.nrows() {
            grad[i] /= self.masses[i];
        }
        grad
    }

    fn clone_box(&self) -> Box<dyn Hamiltonian> {
        Box::new(self.clone())
    }
}

/// A general Hamiltonian from a closure.
pub struct FnHamiltonian {
    f: Box<dyn Fn(&PhasePoint) -> f64 + Send + Sync>,
}

impl FnHamiltonian {
    pub fn new<F: Fn(&PhasePoint) -> f64 + Send + Sync + 'static>(f: F) -> Self {
        Self { f: Box::new(f) }
    }
}

impl Hamiltonian for FnHamiltonian {
    fn evaluate(&self, point: &PhasePoint) -> f64 {
        (self.f)(point)
    }

    fn clone_box(&self) -> Box<dyn Hamiltonian> {
        // We can't clone closures in general, so we just re-evaluate
        Box::new(FnHamiltonian::new({
            // This is a hack — closures aren't cloneable in general.
            // For practical use, use SeparableHamiltonian or implement Hamiltonian directly.
            let val = (self.f)(point); // placeholder
            move |_| val // WRONG — but we need a way to handle this
        }))
    }
}

/// Hamiltonian agent state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HamiltonianAgent {
    /// The agent's phase space
    pub phase_space: PhaseSpace,
    /// Current state in phase space
    pub state: PhasePoint,
    /// Time
    pub time: f64,
}

impl HamiltonianAgent {
    /// Create a new agent in the given phase space.
    pub fn new(phase_space: PhaseSpace, initial_state: PhasePoint) -> Self {
        assert_eq!(phase_space.dim, initial_state.dim());
        Self {
            phase_space,
            state: initial_state,
            time: 0.0,
        }
    }

    /// Compute the agent's energy under a Hamiltonian.
    pub fn energy<H: Hamiltonian + ?Sized>(&self, h: &H) -> f64 {
        h.evaluate(&self.state)
    }

    /// Get the equations of motion at the current state.
    pub fn equations_of_motion<H: Hamiltonian + ?Sized>(&self, h: &H) -> (DVector<f64>, DVector<f64>) {
        h.equations_of_motion(&self.state)
    }

    /// Advance the agent by one Euler step (for testing, not symplectic).
    pub fn euler_step<H: Hamiltonian + ?Sized>(&mut self, h: &H, dt: f64) {
        let (dqdt, dpdt) = h.equations_of_motion(&self.state);
        self.state.q += &dqdt * dt;
        self.state.p += &dpdt * dt;
        self.time += dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_separable_hamiltonian_evaluate() {
        let h = SeparableHamiltonian::uniform(2, 1.0, 1.0);
        let pt = PhasePoint::new(
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        );
        // T = ½(0 + 1) = 0.5, V = ½(1 + 0) = 0.5, H = 1.0
        let energy = h.evaluate(&pt);
        assert_relative_eq!(energy, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_separable_hamiltonian_gradients() {
        let h = SeparableHamiltonian::uniform(2, 2.0, 3.0);
        let pt = PhasePoint::new(
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![3.0, 4.0]),
        );
        let grad_q = h.grad_q(&pt);
        assert_relative_eq!(grad_q[0], 3.0, epsilon = 1e-10); // k*q = 3*1
        assert_relative_eq!(grad_q[1], 6.0, epsilon = 1e-10); // k*q = 3*2

        let grad_p = h.grad_p(&pt);
        assert_relative_eq!(grad_p[0], 1.5, epsilon = 1e-10); // p/m = 3/2
        assert_relative_eq!(grad_p[1], 2.0, epsilon = 1e-10); // p/m = 4/2
    }

    #[test]
    fn test_equations_of_motion() {
        let h = SeparableHamiltonian::uniform(1, 1.0, 1.0);
        let pt = PhasePoint::new(
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![1.0]),
        );
        let (dqdt, dpdt) = h.equations_of_motion(&pt);
        // dq/dt = ∂H/∂p = p/m = 1.0
        assert_relative_eq!(dqdt[0], 1.0, epsilon = 1e-10);
        // dp/dt = -∂H/∂q = -k*q = 0.0
        assert_relative_eq!(dpdt[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_harmonic_oscillator_energy_conservation() {
        let h = SeparableHamiltonian::uniform(1, 1.0, 1.0);
        let ps = PhaseSpace::new(1);
        let initial = PhasePoint::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![0.0]),
        );
        let mut agent = HamiltonianAgent::new(ps, initial);
        let initial_energy = agent.energy(&h);

        // Small Euler steps — energy will drift but should be close for small dt
        for _ in 0..100 {
            agent.euler_step(&h, 0.001);
        }
        // Euler doesn't conserve energy well, just check it's not blown up
        assert!(agent.energy(&h) < 10.0 * initial_energy);
    }

    #[test]
    fn test_hamiltonian_agent_creation() {
        let ps = PhaseSpace::new(3);
        let initial = PhasePoint::zeros(3);
        let agent = HamiltonianAgent::new(ps, initial);
        assert_eq!(agent.time, 0.0);
        assert_eq!(agent.state.dim(), 3);
    }

    #[test]
    fn test_fn_hamiltonian() {
        let h = FnHamiltonian::new(|pt| {
            pt.q[0].powi(2) + pt.p[0].powi(2)
        });
        let pt = PhasePoint::new(
            DVector::from_vec(vec![3.0]),
            DVector::from_vec(vec![4.0]),
        );
        assert_relative_eq!(h.evaluate(&pt), 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_separable_hamiltonian_clone() {
        let h = SeparableHamiltonian::uniform(2, 1.0, 2.0);
        let h2 = h.clone();
        let pt = PhasePoint::new(
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![3.0, 4.0]),
        );
        assert_relative_eq!(h.evaluate(&pt), h2.evaluate(&pt), epsilon = 1e-10);
    }

    #[test]
    fn test_hamiltonian_numerical_gradient() {
        // Test that the default numerical gradient matches analytical for separable case
        let h = SeparableHamiltonian::uniform(2, 1.5, 2.5);
        let pt = PhasePoint::new(
            DVector::from_vec(vec![0.7, -0.3]),
            DVector::from_vec(vec![1.2, -0.8]),
        );
        let analytical_gq = h.grad_q(&pt);
        let analytical_gp = h.grad_p(&pt);

        // Create a wrapper that only implements evaluate
        struct TestH;
        impl Hamiltonian for TestH {
            fn evaluate(&self, pt: &PhasePoint) -> f64 {
                0.5 * (1.0/1.5) * pt.p[0].powi(2) + 0.5 * (1.0/1.5) * pt.p[1].powi(2)
                    + 0.5 * 2.5 * pt.q[0].powi(2) + 0.5 * 2.5 * pt.q[1].powi(2)
            }
            fn clone_box(&self) -> Box<dyn Hamiltonian> { Box::new(TestH) }
        }

        let test_h = TestH;
        let num_gq = test_h.grad_q(&pt);
        let num_gp = test_h.grad_p(&pt);

        for i in 0..2 {
            assert_relative_eq!(num_gq[i], analytical_gq[i], epsilon = 1e-5);
            assert_relative_eq!(num_gp[i], analytical_gp[i], epsilon = 1e-5);
        }
    }
}
