## Introduction
Finding the ground state of a physical system is a frequent task of paramount importance in physics and also physical chemistry. The ground state of a given Hamiltonian corresponds to the state of lowest energy and is thus frequently used to deduce the low-energy properties of systems. In this project, we focus on determining the ground state of molecular systems in Euclidian space $\mathbb{E}_3$. Starting with most simple and trivial examples, our method is demonstrated to work. Passing on to more complicated problems, our method quickly runs into issues which will be discussed in detail.

## Theoretical Background
Given a Hamiltonian $H$ which is a linear operator on a suitable function space over $\mathbb{R}^3$, one tries to solve the following eigenvalue equation, also known as the time-independent Schrödinger equation:

$$
    H \Psi = E \Psi
$$

Here $\Psi \in \mathcal{H}$ is a so-called wavefunction, which must be square integrable. $\mathcal{H}$ is the corresponding Hilbert space of square-integrable functions. For our physical systems, we assume that the spectrum of $H$ is bounded from below and there exists a ground state energy $E_0$.

## Variational Principle
To find this energy, along with its eigenfunction $\Psi_0$, a variational principle can be applied: $\Psi_0$ and $E_0$ follow from the minimisation of the expected energy functional:

$$
    E[\Psi] = \frac{\langle \Psi | H | \Psi \rangle}{\langle \Psi | \Psi \rangle} \\
    = \frac{\int_{\mathbb{R}^3} d^3 r\, \Psi^\ast(r) H \Psi(r)}{\int_{\mathbb{R}^3} d^3 r\, |\Psi(r)|^2}
$$

Thus we are posed with an optimisation problem:

$$
    \Psi_0 \in \underset{\Psi \in \mathcal{H}}{\mathrm{argmin}}  \frac{\langle \Psi | H | \Psi \rangle}{\langle \Psi | \Psi \rangle} \quad \text{and} \quad E_0 = E[\Psi_0]
$$

This optimisation problem is hard to tackle, as the Hilbert space $\mathcal{H}$ is infinite dimensional. However, restricting the optimisation to a simpler subset $M \subset \mathcal{H}$ is helpful, as long as the ground state lies still within it. There are plenty of choices for such a subset $M$, but particularly useful are the ones which make $M$ isomorphic to some $\mathbb{R}^p$ with a given number $p$, thus allowing to make $M$ a differential manifold of dimension $p$. The perks of this are that traditional minimisation algorithms can now be employed to minimise the above energy functional.
