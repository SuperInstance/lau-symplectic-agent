# lau-symplectic-agent

A Rust crate that models autonomous agents as **Hamiltonian systems** on symplectic manifolds. Agent beliefs are positions (q), action tendencies are momenta (p), and decisions are canonical transformations that preserve the symplectic form ω. The result: energy conservation, phase-space volume preservation (Liouville's theorem), and structure-preserving dynamics guaranteed by geometry rather than heuristic tricks.

---

## What This Does

| Module | What It Gives You |
|---|---|
| `phase_space` | Symplectic manifolds, phase points (q, p), the canonical symplectic form J |
| `hamiltonian` | `Hamiltonian` trait, separable H = T(p) + V(q), closure-based Hamiltonians, `HamiltonianAgent` |
| `integrator` | Störmer-Verlet, implicit midpoint, symplectic Euler — all structure-preserving |
| `liouville` | Phase-space volume computation, `AgentDiversity` as volume, Liouville verification |
| `poisson` | Numerical Poisson brackets, coupled agents, multi-agent systems with pairwise coupling |
| `decision` | Canonical transformations, symplectic decisions, decision batches, Hamiltonian-flow decisions |

All types derive `Serialize`/`Deserialize`. ~60 tests verify energy conservation, symplecticity, Poisson bracket identities, and Liouville's theorem numerically.

---

## Key Idea

Standard RL agents update state with no geometric constraints — information about the shape of the decision space is lost at every step. A **symplectic agent** lives on a phase space equipped with a non-degenerate 2-form ω. Hamilton's equations

```
dq/dt =  ∂H/∂p,    dp/dt = −∂H/∂q
```

generate a flow that automatically preserves ω, H, and the Liouville volume element. This means:

- **Energy is bounded** (no runaway behaviour).
- **Diversity is preserved** (agent populations don't collapse).
- **Decisions are reversible** (canonical transformations are invertible).

---

## Install

```toml
[dependencies]
lau-symplectic-agent = "0.1"
```

Requires Rust **2021 edition**. Depends on `nalgebra` and `serde`.

---

## Quick Start

### Create an agent and integrate its dynamics

```rust
use lau_symplectic_agent::{
    PhaseSpace, PhasePoint, SeparableHamiltonian, StormerVerlet,
};

// 2-DOF phase space (beliefs × actions)
let ps = PhaseSpace::new(2);

// Harmonic oscillator: H = ½|p|² + ½|q|²
let h = SeparableHamiltonian::uniform(2, 1.0, 1.0);

// Start at position (1,0), momentum (0,1)
let initial = PhasePoint::new(
    nalgebra::dvector![1.0, 0.0],
    nalgebra::dvector![0.0, 1.0],
);

// Integrate with Störmer-Verlet (symplectic, 2nd order)
let integrator = StormerVerlet::new(0.01);
let result = integrator.integrate(&initial, &h, 1000);

println!("Energy before: {:.6}", result.energy_start);
println!("Energy after:  {:.6}", result.energy_end);
println!("Drift:         {:.2e}", (result.energy_end - result.energy_start).abs());
```

### Coupled agents

```rust
use lau_symplectic_agent::{CoupledAgents, PhaseSpace, SeparableHamiltonian, PhasePoint};

let coupled = CoupledAgents::new(
    PhaseSpace::new(2),
    PhaseSpace::new(2),
    0.5, // coupling strength
);

let h = SeparableHamiltonian::uniform(2, 1.0, 1.0);
let state1 = PhasePoint::new(nalgebra::dvector![1.0, 0.0], nalgebra::dvector![0.0, 0.0]);
let state2 = PhasePoint::new(nalgebra::dvector![0.0, 0.0], nalgebra::dvector![0.0, 1.0]);

let energy = coupled.coupled_energy(&h, &h, &state1, &state2);
let ((dq1, dp1), (dq2, dp2)) = coupled.coupled_equations(&h, &h, &state1, &state2);
```

### Symplectic decisions

```rust
use lau_symplectic_agent::{CanonicalTransformation, SymplecticDecision};

let transform = CanonicalTransformation::identity(2);
let decision = SymplecticDecision {
    transformation: transform,
    confidence: 0.95,
    label: "rotate_belief".into(),
};

let new_state = decision.execute(&current_state);
```

---

## API Reference

### `phase_space` module

| Type | Description |
|---|---|
| `PhasePoint` | Point (q, p) on the phase space — creation, addition, scaling, norm, inner product, vector round-trip |
| `SymplecticForm` | Canonical ω with matrix J = [[0, I], [−I, 0]] — apply to tangent vectors, check symplecticity |
| `PhaseSpace` | Manifold of dimension n — origin, symplectic product, volume element |

### `hamiltonian` module

| Type / Trait | Description |
|---|---|
| `Hamiltonian` (trait) | `evaluate`, `grad_q`, `grad_p`, `equations_of_motion`, `clone_box` |
| `SeparableHamiltonian` | H = ½ Σ p²/m + ½ Σ k q² — analytic gradients |
| `FnHamiltonian` | Wrap a closure `Fn(&PhasePoint) -> f64` as a Hamiltonian |
| `HamiltonianAgent` | Agent with phase space, current state, time — `energy`, `equations_of_motion`, `euler_step` |

### `integrator` module

| Integrator | Order | Notes |
|---|---|---|
| `StormerVerlet` | 2 | Leapfrog; best for separable H; excellent energy conservation |
| `ImplicitMidpoint` | 2 | Fixed-point iteration; works for non-separable H |
| `SymplecticEuler` | 1 | Position-first or momentum-first variants |

Each provides `.step()` and `.integrate()` returning an `IntegrationResult` (state, time, energy before/after, step count). Störmer-Verlet also has `.integrate_with_trajectory()`.

### `liouville` module

| Function / Type | Description |
|---|---|
| `parallelepiped_volume(vectors)` | √det(VᵀV) for basis vectors |
| `PhaseRegion` | Centre + basis vectors — volume, vertices, transform |
| `AgentDiversity` | Phase-space volume of an agent population; `preservation_ratio` |
| `verify_liouville(v₀, v₁, tol)` | Check |v₁ − v₀| / v₀ < tol |

### `poisson` module

| Function / Type | Description |
|---|---|
| `poisson_bracket(&f, &g, point)` | Numerical {f, g} via central differences |
| `CoupledAgents` | Two agents with coupling strength — coupled energy, equations of motion |
| `MultiAgentSystem` | N agents + coupling matrix — total energy, pairwise Poisson brackets, centre of mass |

### `decision` module

| Type | Description |
|---|---|
| `CanonicalTransformation` | 2n×2n matrix (optionally affine) — `apply`, `compose`, `inverse`, `validate` |
| `SymplecticDecision` | Transformation + confidence + label — `execute`, `from_hamiltonian_flow` |
| `DecisionBatch` | Sequence of decisions — `execute` (chained), `all_symplectic` |

---

## How It Works

### Phase Space Structure
An n-dimensional agent lives in a 2n-dimensional phase space with coordinates (q₁…qₙ, p₁…pₙ). The **symplectic form** ω is represented by the canonical matrix:

```
J = [[0, I], [-I, 0]]
```

ω(u, v) = uᵀJv. This is skew-symmetric (ω(v,u) = −ω(u,v)) and non-degenerate (det(J) = 1).

### Hamilton's Equations
Given a Hamiltonian H(q, p), the equations of motion are:

```
dq/dt =  ∂H/∂p    (belief evolves along action gradient)
dp/dt = −∂H/∂q    (action evolves against belief gradient)
```

For the built-in `SeparableHamiltonian`, the analytic gradients are:
- ∂H/∂qᵢ = kᵢ qᵢ (spring force)
- ∂H/∂pᵢ = pᵢ / mᵢ (velocity)

### Symplectic Integrators
The Störmer-Verlet (leapfrog) scheme splits each time step into three sub-steps:

```
p_{½} = pₙ − (dt/2) ∂V/∂q(qₙ)
q_{n+1} = qₙ + dt ∂T/∂p(p_{½})
p_{n+1} = p_{½} − (dt/2) ∂V/∂q(q_{n+1})
```

This is second-order accurate and **symplectic**: the flow map exactly preserves ω (to machine precision). Energy oscillates but never drifts.

### Liouville's Theorem
The phase-space flow Φₜ generated by a Hamiltonian preserves the Liouville volume element: for any region Ω,

```
vol(Φₜ(Ω)) = vol(Ω)
```

In agent terms: **diversity is conserved**. The crate verifies this numerically by tracking the volume of an agent ensemble through Störmer-Verlet integration.

### Poisson Brackets
The Poisson bracket {f, g} measures the interaction between two observables:

```
{f, g} = Σᵢ (∂f/∂qᵢ · ∂g/∂pᵢ − ∂f/∂pᵢ · ∂g/∂qᵢ)
```

Key identities: {f, f} = 0, {f, g} = −{g, f}, {qᵢ, pⱼ} = δᵢⱼ.

### Canonical Transformations & Decisions
A canonical transformation M satisfies MᵀJM = J. These form a group (composable, invertible). A **symplectic decision** is an application of such a transformation to the agent's phase point. Because the transformation preserves ω, the decision doesn't distort the geometry of the agent's state space.

---

## The Math

**Symplectic form:**
$$\omega(u, v) = u^\top J v, \qquad J = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix}$$

**Hamilton's equations:**
$$\dot{q}_i = \frac{\partial H}{\partial p_i}, \qquad \dot{p}_i = -\frac{\partial H}{\partial q_i}$$

**Separable Hamiltonian:**
$$H(q, p) = \underbrace{\frac{1}{2}\sum_i \frac{p_i^2}{m_i}}_{T(p)} + \underbrace{\frac{1}{2}\sum_i k_i\, q_i^2}_{V(q)}$$

**Störmer-Verlet update:**
$$p_{n+\frac{1}{2}} = p_n - \frac{\Delta t}{2}\nabla_q V(q_n), \quad q_{n+1} = q_n + \Delta t\,\nabla_p T(p_{n+\frac{1}{2}}), \quad p_{n+1} = p_{n+\frac{1}{2}} - \frac{\Delta t}{2}\nabla_q V(q_{n+1})$$

**Poisson bracket:**
$$\{f, g\} = \sum_i \left(\frac{\partial f}{\partial q_i}\frac{\partial g}{\partial p_i} - \frac{\partial f}{\partial p_i}\frac{\partial g}{\partial q_i}\right)$$

**Liouville's theorem:**
$$\frac{d}{dt}\int_\Omega d\mu = 0 \qquad \text{(phase space volume is constant)}$$

**Canonical transformation condition:**
$$M^\top J M = J$$

---

## Note

The `lib.rs` declares modules `contact`, `generating`, and `reduction` which are not yet implemented. Importing those will cause a compile error until they are added.

---

## Tests

**~60 tests** covering:

- Phase-point arithmetic and vector round-trips
- Symplectic form: skew-symmetry, J² = −I, symplecticity checks
- Hamiltonian evaluation and gradient correctness (analytic vs numerical)
- Störmer-Verlet energy conservation (< 0.01 drift over 1000 steps)
- Periodicity of the harmonic oscillator
- Liouville volume preservation (numerical verification)
- Poisson bracket identities: {f,f} = 0, antisymmetry, Jacobi identity
- Coupled agent energy and equations of motion
- Canonical transformation validation, composition, inversion
- Symplectic decision execution and Hamiltonian-flow construction

```bash
cargo test
```

---

## License

MIT
