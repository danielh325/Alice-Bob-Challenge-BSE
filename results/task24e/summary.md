# Task 2.4(c-d): time-domain proof of the proposed readouts

Full-model stochastic trajectories (eta = 0.6, 2000-4000 per codeword, boxcar filter) and
master-equation read cycles. No surrogate model or Liouvillian rate is used here.

## Records

| Run | alpha | mu^2 trajectories / record model (1/us) | T for error <= 1e-3 (us) | error at T_end (counted / Gaussian) |
|---|---|---|---|---|
| conversion, κ_ro=0.1κ₂ | 1.4 | 2.213 / 2.647 | - | 6.6e-03 / 5.5e-03 at 3.0 |
| conversion, κ_ro=0.1κ₂ | 2.1 | 5.147 / 5.956 | 2.04 | 0.0e+00 / 3.7e-05 at 3.0 |
| ε_Z readout | 1.4 | 0.342 / 0.450 | - | 9.8e-02 / 9.2e-02 at 12.0 |
| ε_Z readout | 2.1 | 0.209 / 0.200 | - | 4.0e-03 / 2.9e-03 at 40.0 |
| inflate-then-read (→2.5) | 2.1 | 0.211 / 0.200 | - | 2.0e-03 / 2.3e-03 at 40.0 |

## Drive-induced flips over a full read cycle (master equation)

| Run | alpha | read time (us) | p_flip from read drive | p_flip from alpha ramps |
|---|---|---|---|---|
| ε_Z readout | 1.4 | 12.0 | 1.78e-01 | - |
| conversion, κ_ro=0.1κ₂ | 1.4 | 2.0 | 4.41e-03 | - |
| ε_Z readout | 2.1 | 31.0 | 4.07e-03 | - |
| conversion, κ_ro=0.1κ₂ | 2.1 | 1.2 | 3.29e-05 | - |
| inflate-then-read (→2.5) | 2.1 | 31.0 | 2.83e-04 | -1.3e-06 |
