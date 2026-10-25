# Task 2.4(c-d) follow-up: readouts with computed flip rates

eta = 0.6, HMM + SPRT + freeze + erasure protocol of studies/task24c_protocol.py.
Gamma_X from Liouvillian gaps (no illustrative rates).

| Readout | Gamma_X (rad/us) | mu^2 (1/us) | Gamma/mu^2 | target | error | erasure | mean T (us) | 99% T (us) |
|---|---|---|---|---|---|---|---|---|
| eps_Z readout, alpha=2.1 | 2.05e-04 | 0.200 | 1.0e-03 | 1.0e-03 | 2.1e-03 | 0.0e+00 | 30.86 | 65.87 |
| eps_Z readout, alpha=2.1 | 2.05e-04 | 0.200 | 1.0e-03 | 3.1e-03 | 1.3e-03 | 1.3e-02 | 20.86 | 59.67 |
| inflate-then-read, alpha_r=2.5 | 7.05e-06 | 0.200 | 3.5e-05 | 1.0e-03 | 4.9e-04 | 0.0e+00 | 19.19 | 55.19 |
| inflate-then-read, alpha_r=2.5 | 7.05e-06 | 0.200 | 3.5e-05 | 2.2e-04 | 6.0e-05 | 5.3e-04 | 36.98 | 74.99 |
| conversion, kappa_ro=0.03 kappa2 | 6.25e-06 | 1.787 | 3.5e-06 | 1.0e-03 | 8.8e-04 | 0.0e+00 | 1.97 | 5.51 |
| conversion, kappa_ro=0.03 kappa2 | 6.25e-06 | 1.787 | 3.5e-06 | 2.1e-05 | 4.5e-05 | 0.0e+00 | 4.62 | 8.39 |
| conversion, kappa_ro=0.1 kappa2 | 5.72e-05 | 5.956 | 9.6e-06 | 1.0e-03 | 8.6e-04 | 0.0e+00 | 0.59 | 1.68 |
| conversion, kappa_ro=0.1 kappa2 | 5.72e-05 | 5.956 | 9.6e-06 | 7.0e-05 | 1.5e-05 | 2.4e-04 | 1.23 | 2.52 |
