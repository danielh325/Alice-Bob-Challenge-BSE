# Alice & Bob Challenge — Cat Qubits — DuQuantum 2026

---

## 0. Read this first

### 0.1 Which track should I pick?

| | Introductory Track | Standard Track |
|---|---|---|
| Who it's for | You've seen basic quantum mechanics (states, operators, the Schrödinger equation) but not necessarily second quantization or open quantum systems. | You're comfortable with creation/annihilation operators and the Lindblad master equation (or at least the Heisenberg picture). |
| What you'll do | Build intuition for a dissipative cat qubit step by step, then prepare a cat qubit and a logical basis state. | Measure the qubit's error rates, design a readout, and optimize it at the circuit level. |
| Section | Section 1 | Section 2 |

**Standard Track teams:** you are judged on Section 2 only, but Section 2 reuses the code you build in Tasks 1.0(c) and 1.1. We recommend working through those two quickly (they also serve as a check that your setup is correct) before starting Section 2.

### 0.2 Resources

- **Tutorials** ([challenge/resources](https://github.com/YQuantum-2026/Alice-Bob-Challenge/tree/main/challenge/resources)). Work through them in this order:
  1. `Tutorial__Dynamiqs.ipynb` — how to use `dynamiqs`: building operators, `sesolve`/`mesolve`, Wigner plots, gradients, and batching (running parameter sweeps in one call).
  2. `Tutorial__OpenQuantumSystems.ipynb` — density matrices and the Lindblad master equation.
  3. `Tutorial__Introduction_to_Cats.ipynb` — why cat qubits, coherent and cat states, two-photon dissipation, and the memory–buffer mechanism.
- **Background in this document:** Section B below fills in what the tutorials assume you already know (what a mode is, creation and annihilation operators, what a Wigner function shows, what dissipation means physically, Hilbert-space truncation) and connects the tutorial notation to the notation used here. You do not need to read any papers to complete the Introductory Track.
- **Optional further reading:**
  - Review of bosonic qubits (cat qubits, Wigner functions, and more): A. Joshi, K. Noh, Y. Y. Gao, "Quantum information processing with bosonic qubits in circuit QED," [arXiv:2008.13471](https://arxiv.org/abs/2008.13471).
  - Pedagogical introduction to the Lindblad equation: D. Manzano, "A short introduction to the Lindblad master equation," [arXiv:1906.04478](https://arxiv.org/abs/1906.04478).
  - Circuit QED (how photons live in superconducting circuits): A. Blais *et al.*, "Circuit quantum electrodynamics," [arXiv:2005.12667](https://arxiv.org/abs/2005.12667).
  - The experiment our parameters come from: U. Réglade, A. Bocquet *et al.*, "Quantum control of a cat qubit with bit-flip times exceeding ten seconds," [arXiv:2307.06617](https://arxiv.org/abs/2307.06617).
  - Why cat qubits make error correction cheaper: J. Guillaud, M. Mirrahimi, "Repetition cat qubits for fault-tolerant quantum computation," [arXiv:1904.09474](https://arxiv.org/abs/1904.09474).

### 0.3 AI policy

You **may** use AI assistants (ChatGPT, Claude, Copilot, etc.) to help you understand this challenge, the physics behind it, and the tools.

**WR: Organizers, is this your intention?**
You may also use AI tools to help write code, provided that (i) you state in your README where and how you used them, and (ii) your team can explain and defend every result you present — judges will ask. AI tools make confident mistakes on exactly this kind of physics (missing factors of $2\pi$, wrong signs, unconverged simulations presented as final). Verifying results is your responsibility, and it is part of what we judge.

### 0.4 Deliverables and judging

- **Code:** a notebook or scripts that reproduce every figure and number you report.
- **Write-up:** a short README (or a markdown section in your notebook) explaining what you did, what you found, and how you checked it.
- **Live demo:** present your results to Alice & Bob judges.
- **Judging criteria:** correctness of the physics and the numerics (including convergence checks), clarity of code and documentation, quality of the presentation, and ingenuity on the open-ended tasks.
- **Prizes:** 1st place, 2nd place, and Best Introductory Track submission.


### 0.5 Units and conventions

- We set $\hbar = 1$, so Hamiltonians are written in units of angular frequency.
- **Time is in microseconds ($\mu$s). Rates and couplings are angular frequencies in rad/$\mu$s.**
- Parameters are quoted as ordinary frequencies $f$ (in MHz or kHz). Convert with $\omega = 2\pi f$. Example: $g_2/2\pi = 0.763$ MHz means `g2 = 2 * jnp.pi * 0.763` in rad/$\mu$s.
- If your results are off by a factor of about 6, you probably forgot a $2\pi$.
- **Numerical precision:** the `dynamiqs` tutorial uses `dq.set_precision('simple')` (32-bit floats), which is fast but can be too coarse for this challenge. Use `dq.set_precision('double')`, especially for the slow decays fitted in the Standard Track.

---

## B. Background

This section is shared by both tracks. Introductory Track participants should read it all; Standard Track participants can skim. Sections B.1, B.4, B.6, and B.8 overlap with the tutorials and are kept short here; the others cover material the tutorials don't.

### B.1 Why cat qubits?

A qubit can suffer two kinds of errors:

- a **bit flip** ($X$ error): $|0\rangle \leftrightarrow |1\rangle$;
- a **phase flip** ($Z$ error): the relative sign of a superposition flips, $|0\rangle + |1\rangle \to |0\rangle - |1\rangle$.

Most qubits suffer both at comparable rates, so quantum error correction must fight both. This costs many physical qubits per protected ("logical") qubit.

A **cat qubit** stores $|0\rangle$ and $|1\rangle$ in two states of a single microwave oscillator that are far apart from each other (Section B.4). Ordinary noise has a very hard time turning one into the other, so **bit flips are suppressed exponentially** as the qubit is made "bigger"; experiments have reached bit-flip times above ten seconds. Phase flips are *not* suppressed — they grow (linearly) with the size of the cat. Because only one type of error is left to correct, error correction becomes much cheaper (see the figure in `Tutorial__Introduction_to_Cats.ipynb`). This trade-off — bit flips down exponentially, phase flips up linearly — is the central theme of the Standard Track.

### B.2 What is a "mode", and why are we talking about photons?

Alice & Bob's qubits are made from **superconducting circuits** cooled to about 10 mK. A superconducting resonator (think of a tiny LC circuit, or a metal box for microwaves) can hold an oscillating electromagnetic field.

In this challenge, **a "mode" means one specific pattern in which that field can oscillate, at one well-defined frequency** — like a single note on a guitar string. Mathematically, each mode is exactly a **quantum harmonic oscillator**. Its energy comes in discrete packets ("quanta") of size $\hbar\omega$; for electromagnetic fields these quanta are **photons**, here microwave photons at a few GHz. "The memory has 4 photons" just means "the memory oscillator is (on average) in its 4th excited level."

We use two modes:

- the **memory** (operator $\hat a$): long-lived; it stores the qubit;
- the **buffer** (operator $\hat b$): deliberately leaky; it is a tool used to stabilize the memory (Section B.8).

### B.3 Creation and annihilation operators, and $\langle \hat a^\dagger \hat a\rangle$

The state with exactly $n$ photons in a mode is the **Fock state** $|n\rangle$ ($n = 0, 1, 2, \dots$). Two operators move between them:

$$\hat a|n\rangle = \sqrt{n}\,|n-1\rangle \quad\text{(annihilation: removes one photon)}, \qquad \hat a^\dagger|n\rangle = \sqrt{n+1}\,|n+1\rangle \quad\text{(creation: adds one photon)}.$$

The **number operator** $\hat n = \hat a^\dagger \hat a$ satisfies $\hat n|n\rangle = n|n\rangle$: it counts photons. Its expectation value $\langle \hat a^\dagger\hat a\rangle$ is the **average number of photons** you would find if you prepared the same state many times and counted photons each time. Similarly, $\langle\hat a\rangle$ is a complex number that plays the role of the oscillation's classical **amplitude and phase**.

With two modes, the state space is a tensor product: $|n_a\rangle\otimes|n_b\rangle$ means $n_a$ photons in the memory and $n_b$ in the buffer. The operator $\hat a$ acts only on the memory, and $\hat b$ only on the buffer.

### B.4 Coherent states and cat states

A **coherent state** $|\alpha\rangle$ (with $\alpha$ a complex number) is the quantum state closest to a classical oscillating field with a definite amplitude and phase — it is what a microwave generator or a laser produces. Mathematically,

$$|\alpha\rangle = e^{-|\alpha|^2/2}\sum_{n=0}^{\infty}\frac{\alpha^n}{\sqrt{n!}}|n\rangle, \qquad \hat a|\alpha\rangle = \alpha|\alpha\rangle, \qquad \langle\hat a^\dagger\hat a\rangle = |\alpha|^2.$$

Physically: the photon number is not definite (it is Poisson-distributed with mean $|\alpha|^2$), but the amplitude and phase are as sharp as quantum mechanics allows.

The states $|+\alpha\rangle$ and $|-\alpha\rangle$ have the same amplitude but **opposite phase**. For $|\alpha| \gtrsim 1.5$ they are almost perfectly distinguishable (their overlap is $|\langle\alpha|{-\alpha}\rangle|^2 = e^{-4|\alpha|^2}$). They are our two logical states:

$$|0_L\rangle \approx |+\alpha\rangle, \qquad |1_L\rangle \approx |-\alpha\rangle.$$

Superpositions of them are **cat states** (after Schrödinger's cat, "alive and dead at once"):

$$|\mathcal C_\pm\rangle = \frac{|\alpha\rangle \pm |-\alpha\rangle}{\mathcal N_\pm},$$

with normalization $\mathcal N_\pm = \sqrt{2\left(1 \pm e^{-2|\alpha|^2}\right)}$. $|\mathcal C_+\rangle$ contains only even photon numbers and $|\mathcal C_-\rangle$ only odd ones (the tutorial plots both Fock distributions). The **parity operator** $\hat P = e^{i\pi\hat a^\dagger\hat a}$ gives $+1$ on even Fock states and $-1$ on odd ones, so $\langle\hat P\rangle = \pm 1$ for $|\mathcal C_\pm\rangle$. These are the logical $|\pm_L\rangle$ states; the tutorial writes them as $\lvert\pm\rangle_\alpha$.

### B.5 Phase space and Wigner functions

A classical oscillator is fully described by a point in **phase space** (position $x$ and momentum $p$, or equivalently the complex amplitude $\alpha$). A quantum state can't be a point (uncertainty principle), but it can be represented by a function over phase space: the **Wigner function** $W(x,p)$.

- It behaves like a probability distribution (it integrates to 1, and integrating over $p$ gives the probability distribution of $x$), **but it can be negative**. Negative regions are a fingerprint of genuinely quantum behavior.
- A coherent state $|\alpha\rangle$ looks like a round Gaussian blob centered at $(\mathrm{Re}\,\alpha, \mathrm{Im}\,\alpha)$. The vacuum $|0\rangle$ is a blob at the origin.
- A cat state looks like **two blobs** at $\pm\alpha$ **plus interference fringes** (alternating positive and negative stripes) between them. The fringes encode the superposition; their sign pattern tells you the parity.
- A **classical mixture** of $|+\alpha\rangle$ and $|-\alpha\rangle$ ("one or the other, we don't know which") shows the two blobs but **no fringes**. This distinction matters in Task 1.3.

`dynamiqs` computes and plots Wigner functions for you (see the tutorials; e.g. `dq.plot.wigner`, `dq.plot.wigner_mosaic`, `dq.plot.wigner_gif`). Check the axis convention of whichever function you use.

### B.6 From the Schrödinger equation to the Lindblad equation

**Closed systems.** An isolated system evolves by the Schrödinger equation, $i\,\frac{d}{dt}|\psi\rangle = \hat H|\psi\rangle$. The Hamiltonian $\hat H$ generates *reversible* evolution.

**Density matrices.** If we only know that the system is in $|\psi_k\rangle$ with probability $p_k$, we describe it with the density matrix $\hat\rho = \sum_k p_k|\psi_k\rangle\langle\psi_k|$. This is also unavoidable when the system is entangled with an environment we don't keep track of. The closed-system equation becomes $\frac{d\hat\rho}{dt} = -i[\hat H,\hat\rho]$. Expectation values are $\langle\hat O\rangle = \mathrm{Tr}(\hat O\hat\rho)$.

**Open systems.** Real devices leak energy and information into their environment. Under standard assumptions (weak coupling, memoryless environment), this adds **dissipators** to the equation of motion — the **Lindblad master equation**:

$$\frac{d\hat\rho}{dt} = -i[\hat H,\hat\rho] + \sum_k \mathcal D_{\hat L_k}[\hat\rho], \qquad \mathcal D_{\hat L}[\hat\rho] = \hat L\hat\rho\hat L^\dagger - \tfrac12 \\{ \hat L^\dagger \hat L , \hat\rho \\}.$$

**Notation note.** `Tutorial__OpenQuantumSystems.ipynb` puts the rate $\gamma_i$ *inside* its definition of $\mathcal D$. Here, $\mathcal D$ has no rate; the rate either multiplies it ($\kappa_a\mathcal D_{\hat a}$) or sits inside the jump operator ($\sqrt{\kappa_a}\,\hat a$, which is what you pass to `dq.mesolve`). All three describe the same physics.

**How to read a dissipator.** Each **jump operator** $\hat L$ describes one elementary irreversible event in which the environment takes something from the system. If $\hat L = \sqrt{\kappa}\,\hat a$, the event is "one photon leaks out," and in a short time $dt$ it happens with probability $\kappa\langle\hat a^\dagger\hat a\rangle\,dt$. The term $\hat L\hat\rho\hat L^\dagger$ describes the jump itself; the anticommutator term describes how the state changes when *no* jump happens (and keeps $\mathrm{Tr}\,\hat\rho = 1$). In short: **the Hamiltonian is the reversible part, the dissipators are the irreversible part.**

### B.7 What "photon dissipation" means physically

No resonator is perfectly isolated: it is coupled to cables, to its packaging, and to microscopic defects in its materials. A photon in the resonator eventually escapes into one of these and never returns. This is **single-photon loss**, described by the jump operator $\sqrt{\kappa}\,\hat a$ at rate $\kappa$. The average photon number decays as $e^{-\kappa t}$.

Two useful facts, both of which you will see in your simulations:

- A coherent state *stays* coherent under single-photon loss; it just shrinks: $|\alpha\rangle \to |\alpha e^{-\kappa t/2}\rangle$. (Because $\hat a|\alpha\rangle = \alpha|\alpha\rangle$, losing a photon doesn't change a coherent state.)
- A cat state does *not*: $\hat a|\mathcal C_\pm\rangle \propto |\mathcal C_\mp\rangle$. **Each lost photon flips the parity, which is a phase flip.** This is why single-photon loss in the memory ($\kappa_a$) causes phase flips but not bit flips.

### B.8 Memory + buffer: engineering two-photon dissipation

To *stabilize* a cat qubit, we engineer a very particular kind of dissipation in which the memory exchanges photons with its environment **only in pairs**.

1. The memory and buffer are coupled through a nonlinear superconducting element (an asymmetrically threaded SQUID, or ATS), whose flux is modulated by a **pump** at frequency $\omega_p = 2\omega_a - \omega_b$. Energy conservation then allows exactly one process: **two memory photons ↔ one buffer photon (+ one pump photon)**. This is the term $g_2\hat a^2\hat b^\dagger + g_2\hat a^{\dagger 2}\hat b$ in the Hamiltonian below: $\hat a^2\hat b^\dagger$ destroys two memory photons and creates one buffer photon; its conjugate does the reverse.
2. The buffer is **very lossy** ($\kappa_b \gg \kappa_a$). A buffer photon usually leaks out before it can convert back. Net effect: **the memory loses photons in pairs.**
3. A microwave **drive on the buffer** ($\epsilon_d$) continuously injects buffer photons, which convert into pairs of memory photons. Net effect: **the memory gains photons in pairs.**
4. Gain and loss balance at two points: the coherent states $|\pm\alpha\rangle$ with $\alpha^2 = \epsilon_d/g_2$. Any state nearby is pushed back toward one of them. Think of the memory's phase space as having two "valleys" at $\pm\alpha$.

**Why this suppresses bit flips:** going from $+\alpha$ to $-\alpha$ requires crossing the "ridge" between the valleys. Single-photon loss keeps a coherent state coherent and at the same phase (B.7), so it doesn't push you across; the restoring dynamics push you back into your valley before you get far. The bigger $\alpha$, the farther apart the valleys and the rarer a crossing.

**Why parity matters:** the stabilization adds and removes photons only in pairs, so by itself it never changes photon-number parity. Keep this in mind for Task 1.3.

**The effective single-mode picture.** When the buffer is fast enough, it can be eliminated from the description entirely ("adiabatic elimination"). The memory then sees a single dissipator with jump operator

$$\hat L_2 = \sqrt{\kappa_2}\,\big(\hat a^2 - \alpha^2\big), \qquad \kappa_2 = \frac{4 g_2^2}{\kappa_b}.$$

Its two dark states (states that are left unchanged) are $|\pm\alpha\rangle$, since $\hat a^2|\pm\alpha\rangle = \alpha^2|\pm\alpha\rangle$. You'll use this simpler model in Task 1.0 and compare it to the full two-mode model in Task 1.1. The tutorial states the condition for this approximation as $\kappa_b \gg g_2, \epsilon_d$. Check that condition against the provided parameters before trusting the effective model (Task 1.1).

**Matching the tutorial's notation.** The tutorial writes the drive as $+\epsilon_d^*\hat b + \epsilon_d\hat b^\dagger$ and gets $\alpha^2 = -\epsilon_d/g_2^*$. This challenge writes it as $-\epsilon_d(\hat b + \hat b^\dagger)$ with real $\epsilon_d$ and $g_2$, which gives $\alpha^2 = +\epsilon_d/g_2$. These are the same physics: $\epsilon_d^{\text{here}} = -\epsilon_d^{\text{tutorial}}$. Use the convention in this document.

**Rotating frame.** You won't see the mode frequencies $\omega_a\hat a^\dagger\hat a$ or $\omega_b\hat b^\dagger\hat b$ in $\hat H_{2\mathrm{ph}}$ below. That's because we describe the system in a frame rotating along with each mode, which removes the fast GHz oscillation and keeps only the slow (MHz) dynamics we care about.

### B.9 Simulating oscillators on a computer: truncation ($N_a$, $N_b$)

An oscillator has infinitely many levels; a computer can only keep a finite number. We keep the lowest **$N_a$ Fock states of the memory** and the lowest **$N_b$ Fock states of the buffer**. The two-mode state space then has dimension $N_a N_b$, and the density matrix has $(N_a N_b)^2$ entries, so cost grows quickly.

A truncated simulation is only trustworthy if **the population in the highest few Fock levels of each mode stays negligible at all times** — not just at the end. Check this explicitly and monitor it for both modes. To look at one mode on its own, take the partial trace over the other (`dq.ptrace` in `dynamiqs`; the tutorials don't cover this, so see the [`dynamiqs` documentation](https://www.dynamiqs.org)). The standard test is a **convergence study**: increase $N_a$ (or $N_b$), rerun, and check that the quantities you care about don't change.

### B.10 Glossary

| Symbol / term | Meaning |
|---|---|
| $\hat a$, $\hat a^\dagger$ | Annihilation / creation operator of the memory mode |
| $\hat b$, $\hat b^\dagger$ | Annihilation / creation operator of the buffer mode |
| $\langle\hat a^\dagger\hat a\rangle$ | Mean photon number in the memory |
| $\alpha$ | Cat amplitude; $\alpha^2$ is the target mean photon number |
| $g_2$ | Two-photon exchange rate between memory and buffer |
| $\epsilon_d$ | Buffer drive amplitude; sets $\alpha^2 = \epsilon_d/g_2$ |
| $\kappa_a$ | Memory single-photon loss rate (causes phase flips). Called $\kappa_1$ in the slide deck |
| $\kappa_b$ | Buffer single-photon loss rate (part of the engineered stabilization) |
| $\kappa_2$ | Effective two-photon dissipation rate, $4g_2^2/\kappa_b$ |
| $N_a$, $N_b$ | Number of Fock states kept for memory / buffer in the simulation |
| $\hat P$ | Photon-number parity, $e^{i\pi\hat a^\dagger\hat a}$ |
| $\Gamma_X$, $T_X$ | Bit-flip rate and time, $T_X = 1/\Gamma_X$ |
| $\Gamma_Z$ | Phase-flip rate |

---

## The model and parameters (both tracks)

The full two-mode model (Réglade, Bocquet *et al.*, [arXiv:2307.06617](https://arxiv.org/abs/2307.06617), with $g_2$ taken real) is

$$\hat H_{2\mathrm{ph}} = g_2\hat a^2 \hat b^\dagger + g_2\hat a^{\dagger 2}\hat b - \epsilon_d\hat b - \epsilon_d\hat b^\dagger$$

with the Lindblad master equation

$$\frac{d\hat\rho}{dt} = -i\left[\hat H_{2\mathrm{ph}}, \hat\rho\right] + \kappa_b\mathcal D_{\hat b}[\hat\rho] + \kappa_a\mathcal D_{\hat a}[\hat\rho],$$

i.e. jump operators $\sqrt{\kappa_b}\,\hat b$ and $\sqrt{\kappa_a}\,\hat a$. Reading it with Section B.8 in mind: the first two terms of $\hat H_{2\mathrm{ph}}$ convert photon pairs; the last two are the buffer drive; $\kappa_b$ empties the buffer; $\kappa_a$ is the unwanted memory loss.

**Provided parameters** (fitted experimental values, Réglade *et al.*, Table S2 — quoted as ordinary frequencies; convert with $\omega = 2\pi f$, see 0.5):

| Quantity | Value |
|---|---|
| $g_2/2\pi$ | $0.763$ MHz |
| $\kappa_b/2\pi$ | $2.6$ MHz |
| $\kappa_a/2\pi$ | $9.3$ kHz |
| target $\alpha$ | $2.1$ (so $\epsilon_d = \alpha^2 g_2$) |

**Suggested starting truncations:** $N_a = 24$ (memory), $N_b = 6$ (buffer). These are starting points, *not* verified values. At least one of them is too small for part of the dynamics in Task 1.1 — finding out which, and how you know, is part of Task 1.2.

**Starter code** (see the tutorials for details; API names may differ slightly between `dynamiqs` versions):

```python
import dynamiqs as dq
import jax.numpy as jnp

dq.set_precision('double')         # see Section 0.5

Na, Nb = 24, 6
a, b = dq.destroy(Na, Nb)          # a acts on the memory, b on the buffer

g2      = 2 * jnp.pi * 0.763       # rad/us
kappa_b = 2 * jnp.pi * 2.6         # rad/us
kappa_a = 2 * jnp.pi * 9.3e-3      # rad/us
alpha   = 2.1
eps_d   = alpha**2 * g2

H = g2 * (a @ a @ b.dag()) + g2 * (a.dag() @ a.dag() @ b) - eps_d * (b + b.dag())
jump_ops = [jnp.sqrt(kappa_b) * b, jnp.sqrt(kappa_a) * a]

psi0  = dq.tensor(dq.fock(Na, 0), dq.fock(Nb, 0))   # vacuum in both modes
tsave = jnp.linspace(0.0, 2.0, 201)                  # us
res = dq.mesolve(H, jump_ops, psi0, tsave, exp_ops=[a.dag() @ a, b.dag() @ b])
```

> 🔧 **REVIEW:** kept because no tutorial builds a two-mode system. API usage matches the tutorials except `dq.destroy(Na, Nb)` returning a tuple and `dq.ptrace`, which should be run once before release.

---

## 1. Introductory Track: Stabilizing and Preparing a Cat Qubit

You'll build up the cat qubit one ingredient at a time. Each task has a **"What you should see"** box describing the expected behavior, so you can tell whether your simulation is working before you move on. If your result disagrees with the box, that's worth investigating — it's either a bug or something interesting.

### Task 1.0: Warm-up — adding dissipation one ingredient at a time

Use a **single mode** (just the memory, $N = 30$) for this task. Times in $\mu$s, rates in rad/$\mu$s.

**(a) Single-photon loss, in physical units.** Section 2 of `Tutorial__Dynamiqs.ipynb` simulates a coherent state rotating and losing photons in dimensionless units. Redo it in this challenge's units: $\hat H = \Delta\,\hat a^\dagger\hat a$ with $\Delta/2\pi = 1$ MHz, jump operator $\sqrt{\kappa}\,\hat a$ with an exaggerated $\kappa/2\pi = 1$ MHz, initial state $|\alpha_0 = 2\rangle$, and $0 \le t \le 1\ \mu$s. Overlay the analytic prediction $\langle\hat a^\dagger\hat a\rangle = |\alpha_0|^2 e^{-\kappa t}$ on your simulation.

> **What you should see.** The simulation and the analytic curve agree. The Wigner blob spirals toward the origin and *stays round*: the state remains a coherent state, just a smaller one (Section B.7). With $\kappa = 0$, it would circle the origin once per microsecond with $\langle\hat a^\dagger\hat a\rangle = 4$ constant.

**(b) Pure two-photon loss.** Set $\hat H = 0$ and use the single jump operator $\sqrt{\kappa_2}\,\hat a^2$, where $\kappa_2 = 4g_2^2/\kappa_b$ from the provided parameters. Same initial state. Simulate long enough that $\langle\hat a^\dagger\hat a\rangle$ stops changing.

> **What you should see.** The photon number drops quickly at first, then **stalls at a value well above zero** (close to 0.5). Question to think about: which Fock states does $\hat a^2$ leave untouched? What fraction of the initial coherent state was in them? (You'll come back to this in Task 1.3.)

**(c) Stabilization: two-photon loss + two-photon drive.** The last cell of `Tutorial__Introduction_to_Cats.ipynb` does this in dimensionless units ($\kappa_2 = 1$, $\alpha = 2$). Redo it with the provided parameters: jump operators $\sqrt{\kappa_2}\,(\hat a^2 - \alpha^2\,\hat{\mathbb 1})$ with $\alpha = 2.1$ and $\sqrt{\kappa_a}\,\hat a$, starting from vacuum $|0\rangle$, for $0\le t\le 2\ \mu$s. Then **gradually change the strength of the dissipation**: repeat with $\kappa_2 \to \kappa_2/4$ and $\kappa_2 \to 4\kappa_2$, and plot all three $\langle\hat a^\dagger\hat a\rangle(t)$ curves together. (The tutorial's batching section shows how to run all three in one call.)

> **What you should see.** $\langle\hat a^\dagger\hat a\rangle$ rises smoothly and monotonically from 0 and settles at $\approx\alpha^2 = 4.41$ within a few hundred nanoseconds at the nominal $\kappa_2$. The Wigner function goes from one blob at the origin to two blobs at $\pm\alpha$ with fringes between them. Changing $\kappa_2$ changes *how fast* the state settles (roughly $\propto 1/\kappa_2$ — can you argue why this would be exact if $\kappa_a = 0$?) but not *where* it settles. **This is the "ideal" behavior of a stabilized cat qubit**, and your reference for Task 1.1.

### Task 1.1: The full two-mode model — inflating a cat

Now use the full two-mode model (memory + buffer) with the provided parameters and the starter code above. Starting from vacuum in both modes, simulate the "inflation" of the memory into the cat manifold. Simulate for long enough to reach steady state ($2\ \mu$s is a reasonable starting guess; confirm it actually covers the transient).

- Plot $\langle\hat a^\dagger\hat a\rangle(t)$ and $\langle\hat b^\dagger\hat b\rangle(t)$.
- **Monitor your truncation:** also plot the population in the top two Fock levels of the memory and the top Fock level of the buffer versus time (Section B.9). Hint: build projectors onto those levels, or partial-trace to each mode and read off the diagonal of its density matrix.
- Produce Wigner-function snapshots of the memory mode (partial-trace out the buffer) through the inflation transient — a mosaic or a GIF both work.
- Overlay your $\langle\hat a^\dagger\hat a\rangle(t)$ from Task 1.0(c) (nominal $\kappa_2$). Describe how the two-mode evolution differs from the effective model, and propose a reason. Hint: the tutorial says the effective model needs $\kappa_b \gg g_2, \epsilon_d$. Compute $g_2/\kappa_b$ and $\epsilon_d/\kappa_b$ for the provided parameters. Is the condition satisfied?

> **What you should see (with a converged truncation).** Unlike Task 1.0(c), $\langle\hat a^\dagger\hat a\rangle$ **overshoots** well above $\alpha^2$ within the first few hundred nanoseconds, then rings down and settles within about 1% of $\alpha^2$ after roughly $1\ \mu$s. $\langle\hat b^\dagger\hat b\rangle$ jumps up early in the transient (to a peak of order a couple of photons — compare that to $N_b$!) and then decays to nearly zero in steady state. The Wigner snapshots show the blob at the origin stretching out and splitting into two blobs at $\pm\alpha$ with interference fringes between them. Over several microseconds the fringes slowly fade — that's single-photon loss $\kappa_a$ causing phase flips (Section B.7), which the Standard Track measures in Task 2.2.

### Task 1.2: Semiclassical sanity check and convergence study

Compare your simulated steady-state $\langle\hat a^\dagger\hat a\rangle$ to the semiclassical prediction $\alpha^2 = \epsilon_d/g_2$. If they disagree by more than a few percent, determine whether that's a real physical effect or a numerical artifact, and support your conclusion with an explicit convergence study.

Your convergence study should:

- vary $N_a$ and $N_b$ **separately** (e.g. $N_a \in \\{16, 20, 24, 32\\}$ at fixed $N_b$, then $N_b \in \\{4, 6, 8, 10, 12\\}$ at fixed $N_a$);
- check **two** quantities: the steady-state $\langle\hat a^\dagger\hat a\rangle$ *and* a transient feature, such as the peak value of $\langle\hat a^\dagger\hat a\rangle(t)$ during the overshoot;
- report, for each, the smallest $N_a$ and $N_b$ at which the result stops changing (to a tolerance you state).

Are the steady state and the transient equally sensitive to truncation? Which truncation matters for which, and why physically?

### Task 1.3 — Open-ended: preparing a logical state

The state you prepared in Task 1.1 is *not* a single logical codeword $|+\alpha\rangle$ or $|-\alpha\rangle$ — starting from vacuum, you should find that (on timescales short compared with the phase-flip time, roughly $1/(2\kappa_a\alpha^2)$) it is the symmetric superposition of both.

a) Identify what state you actually prepared, and justify it. The cats tutorial tells you the answer for the *effective* model; here you must **demonstrate it quantitatively in the two-mode model**:
   - Explain *why*, analytically (which quantity is conserved? recall Task 1.0(b); what is $\langle\hat P\rangle$ for the vacuum?).
   - Track at least two quantities over time from your Task 1.1 simulation: e.g. $\langle\hat P\rangle(t)$, the fidelity of the memory state with $|\mathcal C_+\rangle$, and the purity $\mathrm{Tr}(\hat\rho_a^2)$ of the memory state.
   - Show that it is a *superposition* and not a *mixture* of $|\pm\alpha\rangle$ (Section B.5), and estimate from your data how long it stays one.

b) Propose a method to prepare a definite computational basis state instead (e.g. $|0_L\rangle \equiv |+\alpha\rangle$), and demonstrate it numerically. (A starting idea: what if the memory doesn't start in vacuum? Or: what could break the $\alpha\to-\alpha$ symmetry during inflation?) Quantify how well you did, e.g. via $\mathrm{Re}\langle\hat a\rangle$ or the fidelity with $|+\alpha\rangle$.

c) What does your method cost — extra control lines, calibration, interaction with the stabilization mechanism itself? Discuss briefly.

---

## 2. Standard Track: Characterizing Errors and Optimizing Readout

Cat qubits are noise-biased: the two-photon dissipation channel exponentially suppresses bit-flip ($X$) errors as $\alpha$ grows, while phase-flip ($Z$) errors grow only linearly — the physical origin of a $10^7$ bias at typical operating points (Section B.1). This track asks you to (1) verify that scaling numerically, then (2) characterize and optimize a readout scheme built on the same physics.

Reuse $\hat H_{2\mathrm{ph}}$, the jump operators, and `dynamiqs` from the Introductory Track (we recommend running Tasks 1.0(c) and 1.1 first as a setup check — see 0.1). Note a notational reconciliation: the reference slide deck's "$\kappa_1$" (single-photon memory loss) is this document's $\kappa_a$. Truncations must be re-checked for each $\alpha$ you use: the memory needs more Fock states as $\alpha^2$ grows, and the buffer transient can be larger than its steady state suggests (Task 1.2).

**Task 2.1: Extracting the bit-flip rate $\Gamma_X$**

Initialize the memory in $|\alpha\rangle \approx |0_L\rangle$ (buffer in vacuum) and evolve under $\hat H_{2\mathrm{ph}}$ plus the same jump operators as Track 1. Coherent states are quasi-eigenstates of single-photon loss, so a bit flip isn't a discrete "click" — it shows up as the slow, ensemble-averaged decay of $\mathrm{Re}\langle\hat a\rangle(t)$ toward zero. Fit this decay to extract $\Gamma_X = 1/T_X$.

$T_X$ grows exponentially with $\alpha^2$, so this is only numerically tractable at small $\alpha$ (roughly $\alpha \lesssim 1.4$) — pushing much higher makes the simulation intractable within reasonable wall-clock time. That's the physics showing up as a numerical wall, not a bug.

- Verify $\Gamma_X \propto e^{-\gamma\alpha^2}$ with a log-linear fit across a few $\alpha$ values. Compare your fitted $\gamma$ to the reference deck's $\gamma \lesssim 2$ bound — treat this as a coarse consistency check, not a precision target.
- **Discussion (not separately graded, but worth writing up):** work out the scaling you'd get from a naive Fermi's-Golden-Rule estimate of the direct matrix element connecting $|+\alpha\rangle$ and $|-\alpha\rangle$. Does it match your simulation? If not, why not?

> 🔧 **REVIEW (David):** this model has no thermal photons ($n_{\mathrm{th}} = 0$). Earlier notebook work found that $n_{\mathrm{th}} = 0$ gives $\gamma > 2$, while the deck's $\gamma \lesssim 2$ reflects thermal population in the experiment. If that holds, participants who do everything right will "fail" this consistency check. Consider rewording to "compare to $\gamma \approx 2$ and explain any discrepancy," or adding a thermal jump operator $\sqrt{\kappa_a n_{\mathrm{th}}}\,\hat a^\dagger$.

**Task 2.2: Extracting the phase-flip rate $\Gamma_Z$**

Initialize in the even-parity superposition $|+\rangle_L = (|\alpha\rangle + |-\alpha\rangle)/\mathcal N$ and track the parity $\langle \hat P(t)\rangle = \langle e^{i\pi \hat a^\dagger \hat a}\rangle$; fit its decay to get $\Gamma_Z$. Unlike $\Gamma_X$, this does *not* require small $\alpha$ — run directly at the physical target $\alpha = 2.1$ from Track 1.

Verify the predicted linear scaling $\Gamma_Z = 2\kappa_a\alpha^2$ by sweeping $\alpha$, and compare your fitted slope to $2\kappa_a$.

**Task 2.3: Transmon-free memory-bit readout**

Physical picture: displacing the memory slightly imprints information about which well ($\pm\alpha$) it occupies onto the buffer, which can then be read out without a transmon-based dispersive measurement.

Add a memory drive $\hat H_Z = \epsilon_Z\hat a^\dagger + \epsilon_Z\hat a$ (real $\epsilon_Z$) to $\hat H_{2\mathrm{ph}}$.

- Simulate $\langle\hat b\rangle(t)$ starting from each codeword ($\pm\alpha$) and show the two produce distinguishable, opposite-sign steady-state $\langle\hat b\rangle$ — your readout contrast. You may use the linearized estimate $\langle\hat b\rangle_{\rm ss} \approx \mp\epsilon_Z/(2\alpha g_2)$ as a sanity check.
- Choose $\epsilon_Z$ conservatively (well below $g_2$) for now — you'll revisit this choice in Task 2.4.
- Determine how long you need to evolve before $\langle\hat b\rangle$ settles, and whether it approaches steady state monotonically or with oscillation. This matters for how you'd design an actual readout protocol.

**Task 2.4 — Open-ended / capstone: Optimizing the readout**

So far, $\hat H_Z$ was added to $\hat H_{2\mathrm{ph}}$ by hand. In a real device, the two-photon exchange, the buffer drive, and any readout drive all originate from one circuit-level Hamiltonian:

$$\hat H_{\rm circuit} = \omega_a\hat a^\dagger\hat a + \omega_b \hat b^\dagger \hat b - 2E_J\cos(\varphi_\Sigma)\cos(\hat\varphi+\varphi_\Delta), \qquad \hat\varphi = \varphi_a(\hat a+\hat a^\dagger)+\varphi_b(\hat b+\hat b^\dagger)$$

with the ATS biased at $\varphi_\Delta = \pi/2$ and a parametric flux pump $\varphi_\Sigma \to -\pi/2 + \epsilon_p\cos(\omega_p t)$. At this bias point, $\cos(\varphi_\Sigma)\cos(\hat\varphi + \varphi_\Delta) \approx -\epsilon_p\cos(\omega_p t)\sin\hat\varphi$, so the ATS term is (to first order in $\epsilon_p$) $2E_J\epsilon_p\cos(\omega_p t)\sin\hat\varphi$. Expanding $\sin\hat\varphi$ and moving to the frame rotating at $\omega_a$, $\omega_b$, with $\omega_p = 2\omega_a - \omega_b$, generates $g_2\hat a^2\hat b^\dagger + \mathrm{h.c.}$ as the leading resonant term, with

$$g_2 = \tfrac12 E_J\,\epsilon_p\,\varphi_a^2\,\varphi_b.$$

The buffer drive $\epsilon_d$ enters as a separate microwave drive on the buffer at $\omega_b$. The full derivation is in the slide deck.

> 🔧 **REVIEW:** the cats tutorial lists "basic circuit QED" and "how to obtain an approximate Hamiltonian" as learning goals but contains neither. Add this to slide deck.

**Illustrative circuit parameters.** These are chosen to be self-consistent with $g_2$ above and to avoid accidental resonances between low-order processes. They are *not* the parameters of a specific device.

| Quantity | Value |
|---|---|
| $\omega_a/2\pi$ | $4.0$ GHz |
| $\omega_b/2\pi$ | $4.8$ GHz |
| $\omega_p/2\pi = (2\omega_a - \omega_b)/2\pi$ | $3.2$ GHz |
| $E_J/2\pi$ | $40$ GHz |
| $\varphi_a$ (memory zero-point phase fluctuation) | $0.06$ |
| $\varphi_b$ (buffer zero-point phase fluctuation) | $0.3$ |
| $\epsilon_p$ (pump amplitude, rad) | $\approx 0.035$, fixed by $g_2/2\pi = 0.763$ MHz via the formula above |
| $\kappa_a$, $\kappa_b$, $\alpha$ | As in the provided-parameters table |

**A note on tractability.** Simulating $\hat H_{\rm circuit}$ directly means resolving GHz oscillations over microsecond timescales, which is expensive. Reasonable approaches include: simulating short time windows; working in the rotating frame while keeping the time-dependent non-resonant terms up to a chosen order in $\hat\varphi$ (see the `dynamiqs` documentation on time-dependent Hamiltonians, e.g. `dq.modulated`); reducing truncations; or deriving effective corrections analytically (e.g. to second order in the non-resonant terms) and simulating those. State which approach you use and why it is justified.

This task is intentionally open-ended. There is no single expected answer; we want to see how far you push it.

a) Work out how a memory readout drive actually enters this circuit-level Hamiltonian, and what other terms beyond the two-photon term you've used so far survive the same expansion and rotating-wave step.

b) Numerically investigate what happens to your readout contrast (Task 2.3) and to $\Gamma_X$ (Task 2.1) as you increase $\epsilon_Z$. Is there a trade-off? Where does it come from physically?

c) Propose and numerically demonstrate a strategy to improve readout contrast and/or readout speed without significantly degrading $\Gamma_X$ — a control-level idea (e.g., shaping the drive in time or frequency) or a hardware/parameter-level one are both fair game.

d) Justify why your strategy works, using your simulations as evidence.
