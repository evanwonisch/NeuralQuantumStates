## Introduction
Finding the ground state of a physical system is a frequent task of paramount importance in physics and also physical chemistry. The ground state of a given Hamiltonian corresponds to the state of lowest energy and is thus frequently used to deduce the low-energy properties of systems. In this project, we focus on determining the ground state of molecular systems in Euclidian space $\mathbb{E}_3$. Starting with most simple and trivial examples, our method is demonstrated to work. Passing on to more complicated problems, our method quickly runs into issues which will be discussed in detail.

## Theoretical Background
Given a Hamiltonian $H$ which is a linear operator on a suitable function space over $\mathbb{R}^3$ (in the most simple one-electron case I describe here), one tries to solve the following eigenvalue equation, also known as the time-independent Schrödinger equation:

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

## Use Neural Networks
As discussed above, one has to choose a nice subset of the Hilbert space $\mathcal{H}$. This is nicely done by assuming some explicit form of the wavefunction, and then introducing parameters. Of course, the symmetries of a given system can help us guess the form of the wavefunction, and will be incorporated into our algorithm. Still, one does not avoid the question of how to parametrise an arbitrary square-integrable function. Here we use neural networks, as they are universal function approximators, to build such functions.
A neural network can be constructed by concatenation of very simple, yet non-linear functions. Such a building block is called layer and shall be denoted $f_{m,n,\theta}$:

$$
    f_{m,n,\theta}: \mathbb{R}^m \to \mathbb{R}^n
$$

$$
                    x \mapsto \sigma(Ax + b)
$$

Here, $\sigma$ may be a nonlinear function you pick, $A$ a linear map from $\mathbb{R}^m$ to $\mathbb{R}^n$ and $b \in \mathbb{R}^n$. We call the parameters of this layer $\theta = \{A, b\}$. One obtains a neural network by concatenation of many of these layers and collecting all parameters in one $\theta \in \mathbb{R}^p$, where $p$ is the number of values we need, to store all the matrices $A$ and vectors $b$.


$$
    N_\theta : \mathbb{R}^m \to \mathbb{R}^n
$$

$$
                x \mapsto f_1 \circ \dots \circ f_N
$$

And the choice of $f_1$ to $f_N$ has of course to be such, that the concatenation of all of them works and results in the signature indicated for $N_\theta$. We call $N_\theta$ a neural network.

### How to build a wavefunction?

To build a wavefunction, we simply build a neural network with the corresponding signature:

$$
    N_\theta: \mathbb{R}^3 \to \mathbb{R}^2
$$

Here, the choice of how many building blocks and of what size is suppressed, this is to be figured out later. We convert the $\mathbb{R}^2$ output of the neural network to a complex number by simply introducing the imaginary unit to one of its components:

$$
    \Psi(r) = N_\theta^{(1)}(r) + i N_\theta^{(2)}(r)
$$

If one uses a $C^{\infty}$ non-linear function $\sigma$, also $\Psi$ will be $C^{\infty}$ with respect to $r$, and also $\theta$. Yet, when using many layers, $\Psi$ can be very general and approximate a wide variety of functions.
Alternatives are numerous, the above might not be normalisable and thus not a good wavefunction, one can mitigate this: Making sure, that the non-linearity $\sigma$ is always positive, one can have a better candidate by exponentiating it with a minus sign:

$$
    \Psi(r) = \exp[-N_\theta^{(1)}(r) + i N_\theta^{(2)}(r)]
$$

Here, we allow for an arbitrary complex phase, and if the neural network goes to infinity quick enough when $r\to\infty$ (linearly is already enough), then the wavefunction is nicely normalisable.
