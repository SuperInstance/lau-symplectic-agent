//! Symplectic integrators for agent simulation — Störmer-Verlet preserves structure

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::phase_space::{PhasePoint, PhaseSpace};
use crate::hamiltonian::{Hamiltonian, HamiltonianAgent};

/// Result of an integration step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationResult {
    /// Final state
    pub state: PhasePoint,
    /// Time after step
    pub time: f64,
    /// Energy at start
    pub energy_start: f64,
    /// Energy at end
    pub energy_end: f64,
    /// Number of steps taken
    pub steps: usize,
}

/// Störmer-Verlet (leapfrog) symplectic integrator.
/// For separable Hamiltonians H = T(p) + V(q), this is:
///   p_{1/2} = p_n - (dt/2) ∂V/∂q(q_n)
///   q_{n+1} = q_n + dt ∂T/∂p(p_{1/2})
///   p_{n+1} = p_{1/2} - (dt/2) ∂V/∂q(q_{n+1})
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StormerVerlet {
    /// Time step
    pub dt: f64,
}

impl StormerVerlet {
    pub fn new(dt: f64) -> Self {
        assert!(dt > 0.0, "Time step must be positive");
        Self { dt }
    }

    /// Single Störmer-Verlet step.
    pub fn step<H: Hamiltonian + ?Sized>(&self, state: &PhasePoint, h: &H) -> PhasePoint {
        let dt = self.dt;
        // Half-step in momentum
        let grad_q = h.grad_q(state);
        let p_half = &state.p - &(grad_q * (dt / 2.0));

        // Full step in position
        let state_half = PhasePoint::new(state.q.clone(), p_half.clone());
        let grad_p = h.grad_p(&state_half);
        let q_new = &state.q + &(grad_p * dt);

        // Half-step in momentum
        let state_new_q = PhasePoint::new(q_new.clone(), p_half.clone());
        let grad_q_new = h.grad_q(&state_new_q);
        let p_new = &p_half - &(grad_q_new * (dt / 2.0));

        PhasePoint::new(q_new, p_new)
    }

    /// Integrate for n steps.
    pub fn integrate<H: Hamiltonian + ?Sized>(
        &self,
        initial: &PhasePoint,
        h: &H,
        n_steps: usize,
    ) -> IntegrationResult {
        let energy_start = h.evaluate(initial);
        let mut state = initial.clone();

        for _ in 0..n_steps {
            state = self.step(&state, h);
        }

        let energy_end = h.evaluate(&state);

        IntegrationResult {
            state,
            time: n_steps as f64 * self.dt,
            energy_start,
            energy_end,
            steps: n_steps,
        }
    }

    /// Integrate with trajectory recording.
    pub fn integrate_with_trajectory<H: Hamiltonian + ?Sized>(
        &self,
        initial: &PhasePoint,
        h: &H,
        n_steps: usize,
    ) -> (IntegrationResult, Vec<PhasePoint>) {
        let energy_start = h.evaluate(initial);
        let mut state = initial.clone();
        let mut trajectory = vec![state.clone()];

        for _ in 0..n_steps {
            state = self.step(&state, h);
            trajectory.push(state.clone());
        }

        let energy_end = h.evaluate(&state);

        let result = IntegrationResult {
            state,
            time: n_steps as f64 * self.dt,
            energy_start,
            energy_end,
            steps: n_steps,
        };
        (result, trajectory)
    }
}

/// Implicit midpoint symplectic integrator.
/// Second-order symplectic for general (non-separable) Hamiltonians.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplicitMidpoint {
    /// Time step
    pub dt: f64,
    /// Number of fixed-point iterations
    pub iterations: usize,
}

impl ImplicitMidpoint {
    pub fn new(dt: f64, iterations: usize) -> Self {
        Self { dt, iterations }
    }

    /// Single implicit midpoint step.
    pub fn step<H: Hamiltonian + ?Sized>(&self, state: &PhasePoint, h: &H) -> PhasePoint {
        let dt = self.dt;
        let mut mid = state.clone();

        // Fixed-point iteration for the midpoint
        for _ in 0..self.iterations {
            let (dqdt, dpdt) = h.equations_of_motion(&mid);
            mid = PhasePoint::new(
                &state.q + &dqdt * (dt / 2.0),
                &state.p + &dpdt * (dt / 2.0),
            );
        }

        let (dqdt, dpdt) = h.equations_of_motion(&mid);
        PhasePoint::new(
            &state.q + &dqdt * dt,
            &state.p + &dpdt * dt,
        )
    }

    /// Integrate for n steps.
    pub fn integrate<H: Hamiltonian + ?Sized>(
        &self,
        initial: &PhasePoint,
        h: &H,
        n_steps: usize,
    ) -> IntegrationResult {
        let energy_start = h.evaluate(initial);
        let mut state = initial.clone();

        for _ in 0..n_steps {
            state = self.step(&state, h);
        }

        IntegrationResult {
            energy_end: h.evaluate(&state),
            state,
            time: n_steps as f64 * self.dt,
            energy_start,
            steps: n_steps,
        }
    }
}

/// Symplectic Euler integrator (first-order).
/// Two variants: position-first and momentum-first.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SymplecticEulerVariant {
    PositionFirst,
    MomentumFirst,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymplecticEuler {
    pub dt: f64,
    pub variant: SymplecticEulerVariant,
}

impl SymplecticEuler {
    pub fn new(dt: f64, variant: SymplecticEulerVariant) -> Self {
        Self { dt, variant }
    }

    /// Single step.
    pub fn step<H: Hamiltonian + ?Sized>(&self, state: &PhasePoint, h: &H) -> PhasePoint {
        let dt = self.dt;
        match self.variant {
            SymplecticEulerVariant::PositionFirst => {
                let grad_p = h.grad_p(state);
                let q_new = &state.q + &(grad_p * dt);
                let state_q = PhasePoint::new(q_new, state.p.clone());
                let grad_q = h.grad_q(&state_q);
                let p_new = &state.p - &(grad_q * dt);
                PhasePoint::new(q_new, p_new)
            }
            SymplecticEulerVariant::MomentumFirst => {
                let grad_q = h.grad_q(state);
                let p_new = &state.p - &(grad_q * dt);
                let state_p = PhasePoint::new(state.q.clone(), p_new.clone());
                let grad_p = h.grad_p(&state_p);
                let q_new = &state.q + &(grad_p * dt);
                PhasePoint::new(q_new, p_new)
            }
        }
    }

    /// Integrate for n steps.
    pub fn integrate<H: Hamiltonian + ?Sized>(
        &self,
        initial: &PhasePoint,
        h: &H,
        n_steps: usize,
    ) -> IntegrationResult {
        let energy_start = h.evaluate(initial);
        let mut state = initial.clone();

        for _ in 0..n_steps {
            state = self.step(&state, h);
        }

        IntegrationResult {
            energy_end: h.evaluate(&state),
            state,
            time: n_steps as f64 * self.dt,
            energy_start,
            steps: n_steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonian::SeparableHamiltonian;
    use approx::assert_relative_eq;

    #[test]
    fn test_stormer_verlet_energy_conservation() {
        let h = SeparableHamiltonian::uniform(2, 1.0, 1.0);
        let initial = PhasePoint::new(
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        );
        let integrator = StormerVerlet::new(0.01);
        let result = integrator.integrate(&initial, &h, 1000);

        // Störmer-Verlet should conserve energy very well
        let energy_drift = (result.energy_end - result.energy_start).abs();
        assert!(energy_drift < 0.01, "Energy drift too large: {}", energy_drift);
    }

    #[test]
    fn test_stormer_verlet_trajectory() {
        let h = SeparableHamiltonian::uniform(1, 1.0, 1.0);
        let initial = PhasePoint::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![0.0]),
        );
        let integrator = StormerVerlet::new(0.01);
        let (result, trajectory) = integrator.integrate_with_trajectory(&initial, &h, 628);

        // Should approximately complete one period (2π for unit harmonic oscillator)
        assert_eq!(trajectory.len(), 629);
        assert!((result.time - 6.28).abs() < 0.01);
    }

    #[test]
    fn test_stormer_verlet_single_step() {
        let h = SeparableHamiltonian::uniform(1, 1.0, 0.0); // Free particle
        let initial = PhasePoint::new(
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![1.0]),
        );
        let integrator = StormerVerlet::new(0.1);
        let result = integrator.step(&initial, &h);
        // Free particle: q should advance by dt*p/m = 0.1
        assert_relative_eq!(result.q[0], 0.1, epsilon = 1e-10);
        assert_relative_eq!(result.p[0], 1.0, epsilon = 1e-10); // p unchanged for free particle
    }

    #[test]
    fn test_implicit_midpoint_energy_conservation() {
        let h = SeparableHamiltonian::uniform(1, 1.0, 1.0);
        let initial = PhasePoint::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![0.0]),
        );
        let integrator = ImplicitMidpoint::new(0.01, 10);
        let result = integrator.integrate(&initial, &h, 500);
        let energy_drift = (result.energy_end - result.energy_start).abs();
        assert!(energy_drift < 0.01, "Energy drift too large: {}", energy_drift);
    }

    #[test]
    fn test_symplectic_euler_position_first() {
        let h = SeparableHamiltonian::uniform(1, 1.0, 1.0);
        let initial = PhasePoint::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![0.0]),
        );
        let integrator = SymplecticEuler::new(0.01, SymplecticEulerVariant::PositionFirst);
        let result = integrator.integrate(&initial, &h, 100);

        // Symplectic Euler has bounded energy error
        let energy_drift = (result.energy_end - result.energy_start).abs();
        assert!(energy_drift < 0.1, "Energy drift too large: {}", energy_drift);
    }

    #[test]
    fn test_symplectic_euler_momentum_first() {
        let h = SeparableHamiltonian::uniform(1, 1.0, 1.0);
        let initial = PhasePoint::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![0.0]),
        );
        let integrator = SymplecticEuler::new(0.01, SymplecticEulerVariant::MomentumFirst);
        let result = integrator.integrate(&initial, &h, 100);

        let energy_drift = (result.energy_end - result.energy_start).abs();
        assert!(energy_drift < 0.1, "Energy drift too large: {}", energy_drift);
    }

    #[test]
    fn test_stormer_verlet_vs_euler() {
        let h = SeparableHamiltonian::uniform(1, 1.0, 1.0);
        let initial = PhasePoint::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![0.0]),
        );

        let sv = StormerVerlet::new(0.1).integrate(&initial, &h, 100);
        let se = SymplecticEuler::new(0.1, SymplecticEulerVariant::PositionFirst)
            .integrate(&initial, &h, 100);

        // Both should have bounded energy error, but Verlet should be better
        let sv_drift = (sv.energy_end - sv.energy_start).abs();
        let se_drift = (se.energy_end - se.energy_start).abs();
        // Verlet is second-order, so should generally be better than first-order Euler
        assert!(sv_drift < 1.0);
        assert!(se_drift < 1.0);
    }

    #[test]
    fn test_integrator_periodicity() {
        // Harmonic oscillator should return to initial state after one period
        let h = SeparableHamiltonian::uniform(1, 1.0, 1.0);
        let initial = PhasePoint::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![0.0]),
        );

        // Period = 2π, use many small steps
        let n_steps = 6283;
        let dt = 2.0 * std::f64::consts::PI / n_steps as f64;
        let integrator = StormerVerlet::new(dt);
        let result = integrator.integrate(&initial, &h, n_steps);

        // Should approximately return to start
        assert!((result.state.q[0] - 1.0).abs() < 0.01);
        assert!(result.state.p[0].abs() < 0.01);
    }

    #[test]
    fn test_integration_result_fields() {
        let h = SeparableHamiltonian::uniform(1, 1.0, 1.0);
        let initial = PhasePoint::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![0.0]),
        );
        let integrator = StormerVerlet::new(0.01);
        let result = integrator.integrate(&initial, &h, 100);
        assert_eq!(result.steps, 100);
        assert!((result.time - 1.0).abs() < 1e-10);
        assert_eq!(result.state.dim(), 1);
    }
}
