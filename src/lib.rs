//! # lau-symplectic-agent
//!
//! Agents as Hamiltonian systems — symplectic phase space, structure-preserving decisions,
//! guaranteed behavioral invariants.
//!
//! This crate models agents as Hamiltonian systems where:
//! - Phase space positions represent beliefs
//! - Phase space momenta represent action tendencies
//! - The symplectic form ω governs the geometry of agent decisions
//! - Symplectic integrators (Störmer-Verlet) preserve behavioral structure

pub mod phase_space;
pub mod hamiltonian;
pub mod decision;
pub mod integrator;
pub mod liouville;
pub mod poisson;
pub mod contact;
pub mod generating;
pub mod reduction;

pub use phase_space::*;
pub use hamiltonian::*;
pub use decision::*;
pub use integrator::*;
pub use liouville::*;
pub use poisson::*;
pub use contact::*;
pub use generating::*;
pub use reduction::*;
