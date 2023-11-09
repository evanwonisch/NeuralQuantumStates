# NeuralQuantumStates
Research Project concerning Quantum States represented by Neural Networks

## Concerning Kato-Cusps

Using an Ansatz like

$$
\log \Psi = -\sum_{n} k_n |r - r_n|
$$

results in the correct asymptotic behaviour but does not satisfy Kato-Cusp condition.

An alternative Ansatz

$$
\Psi = \sum_{n} \exp[-k_n |r - r_n|]
$$

satsfies the the cusp condition much better but fails to provide correct asymptotic behavior. A tradeoff between short-range and long-range accuracy must be taken. As the MC-sampler rarely samples low-likelihood points, focussing on the correct cusp condition is more reasonable.


## TODO

- very long run N = 10000
- chapter 6.4 stochastic reconfig
- try R optimisation (set different initial R)
- try two elongated hydrogen nuclei