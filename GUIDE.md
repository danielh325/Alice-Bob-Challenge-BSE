# A plain-language guide to our cat-qubit project

This guide explains the whole project in simple words. It follows the same order as our slides, so
you can read it while you present. Each section says what we did, what we found, and why it
happens. A short glossary is at the end.

---

## 1. The big idea

A normal computer stores information in **bits**. Each bit is 0 or 1. A quantum computer uses
**qubits**. A qubit can be 0, 1, or a mix of both at the same time. That mix is called a
**superposition**.

Qubits make two kinds of mistakes:

- A **bit flip** turns 0 into 1, or 1 into 0.
- A **phase flip** keeps the 0 and the 1, but it changes the sign between them. It turns
  "0 plus 1" into "0 minus 1".

Most qubits make both kinds of mistakes, and fixing both is expensive. A **cat qubit** is designed
so that bit flips almost never happen. Then the computer only has to fix phase flips, which is
much cheaper. That is why the company Alice & Bob builds cat qubits.

## 2. The machine: two boxes of microwave light

A cat qubit lives in a tiny superconducting circuit, cooled close to absolute zero. You can think
of it as **two boxes that hold microwave light**:

- The **memory** stores the qubit. It holds light for a long time.
- The **buffer** is a helper. It leaks light quickly on purpose.

Light comes in small energy packets called **photons**. "The memory holds 4 photons" means the
memory contains about 4 of these packets.

A machine part couples the two boxes so that they trade photons in a special way. **Two** memory
photons can turn into **one** buffer photon, and one buffer photon can turn back into two memory
photons. The buffer then leaks its photon away. We also pump new photons into the buffer with a
microwave drive.

The rule for the energy of the system is called the **Hamiltonian**, $\hat H$. For our two boxes it is

$$
\hat H = g_2\left(\hat a^2\hat b^\dagger + \hat a^{\dagger 2}\hat b\right) - \epsilon_d\left(\hat b + \hat b^\dagger\right).
$$

You can read it piece by piece:

- $\hat a$ removes one photon from the memory, and $\hat a^\dagger$ adds one.
- $\hat b$ and $\hat b^\dagger$ do the same for the buffer.
- $\hat a^2\hat b^\dagger$ removes **two** memory photons and adds **one** buffer photon. The
  number $g_2$ says how fast this trade happens.
- $\epsilon_d(\hat b + \hat b^\dagger)$ is the drive that feeds the buffer. The number $\epsilon_d$
  says how strong the drive is.

Two kinds of leaks are also present. The buffer leaks photons at rate $\kappa_b$, which is what we
want. The memory leaks photons at a much smaller rate, $\kappa_a$, which we do not want.

The challenge gives these values:

| Quantity | Meaning | Value |
|---|---|---|
| $g_2/2\pi$ | speed of the two-for-one photon trade | 0.763 MHz |
| $\kappa_b/2\pi$ | how fast the buffer leaks | 2.6 MHz |
| $\kappa_a/2\pi$ | how fast the memory leaks | 9.3 kHz |
| $\alpha$ | size of the cat | 2.1 |

The drive strength is chosen so that the memory settles at about $\alpha^2 = 2.1^2 = 4.41$ photons:

$$
\epsilon_d = \alpha^2 g_2 .
$$

**One number matters a lot.** The textbook shortcut, called the *effective model*, treats the buffer as
if it emptied instantly. That shortcut only works if the buffer leaks much faster than the drive
pushes, which means $\epsilon_d/\kappa_b$ must be much smaller than 1. For our values,

$$
\frac{\epsilon_d}{\kappa_b} = \frac{4.41 \times 0.763}{2.6} \approx 1.29 .
$$

This is **not** small. So we expect the real two-box system to behave differently from the shortcut.
It does, as Section 5 shows.

## 3. What our simulation does

We use a Python library called `dynamiqs`. It computes how the state of the boxes changes in time.

A computer cannot keep infinitely many photon levels. So we keep only the levels from 0 up to some
maximum. This is called **truncation**. We write $N_a$ for the number of levels kept in the memory, and
$N_b$ for the buffer. A simulation is trustworthy only if the highest levels stay almost empty. If
they fill up, the computer is cutting off real physics. We check this throughout with a
**truncation monitor**: a plot of how much probability sits in the top levels.

We also wrote **114 automatic tests**. Each test compares the code with a result we can work out
by hand. If the code ever changes in a wrong way, a test fails.

## 4. Task 1.0: three warm-ups

We first used only the memory, and we added one effect at a time.

**(a) Losing single photons.** The memory leaks one photon at a time at rate $\kappa$. Theory says the
average photon number shrinks exponentially:

$$
\langle n \rangle(t) = |\alpha_0|^2 e^{-\kappa t}.
$$

Our simulation matches this formula to within $10^{-8}$ photons. The light wave also keeps its
shape while it shrinks.

**(b) Losing photons only in pairs.** Now the memory can only lose two photons at a time. You might
expect it to empty completely. It does not: it stops at about **0.5 photons**.

The reason is **parity**, which means whether the photon number is even or odd. Removing photons
two at a time never changes parity. So an even number of photons ends at 0, and an odd number
ends at 1. Our starting light had odd photon numbers with probability

$$
\frac{1 - e^{-8}}{2} \approx 0.49983 ,
$$

and the simulation stops at exactly 0.49983. We use this "parity cannot change" rule again in Task 1.3.

**(c) Stabilization.** Now photons are added in pairs as well as removed in pairs. The memory fills
up and settles at about $\alpha^2 = 4.41$ photons. We tried three strengths of the pair process. A
stronger process fills the memory faster, but every run settles at the same value.

## 5. Task 1.1: the full two-box model

Now we simulate both boxes together, starting with no photons at all.

**What we see.** The memory does not fill up smoothly. It **overshoots** to 6.89 photons after
0.23 microseconds, which is 56% too many. It then swings up and down and settles near 4.41 after
about 1 microsecond. The buffer briefly holds about 2 photons and then empties.

**Why it happens.** The buffer is not fast enough to empty instantly, because $\epsilon_d/\kappa_b \approx 1.29$.
So the buffer stores photons for a short time and then hands them back to the memory. This works like
pushing a child on a swing: stored energy goes back and forth before it calms down. The textbook
shortcut ignores this storage, so it predicts a smooth rise and misses the overshoot.

**The picture.** A **Wigner function** is a map of the light in the memory. A single blob means one
light wave. During the inflation, the blob at the center stretches sideways and splits into two blobs,
one at $+\alpha$ and one at $-\alpha$. Striped "fringes" appear between them. The stripes are the
sign of a quantum superposition.

## 6. Task 1.2: checking that the computer is right

### Enough photon levels?

We repeated the simulation with more and more levels and watched two numbers: the final photon
number, and the height of the overshoot.

| What we measured | Memory levels needed | Buffer levels needed |
|---|---|---|
| final photon number | 16 | 4 |
| height of the overshoot | **28** | **12** |

The final value is easy to get right, because at the end the buffer is nearly empty. The overshoot is
harder, because for a short time the buffer holds about 2 photons, with a small chance of holding many
more. The challenge suggested 6 buffer levels. With 6 levels the overshoot comes out **11.5% too
low**. So 6 buffer levels are too few.

### Is the computer's step size accurate enough?

The simulation moves forward in small time steps. A setting called the **tolerance** controls how
accurate each step must be. With the default tolerance, $10^{-6}$, our most detailed run showed
*negative* probabilities in the top levels. A probability can never be negative, so this was a
computer error, not physics. With a stricter tolerance, $10^{-9}$, the error went away, and the run took
no longer. We now always use $10^{-9}$.

### Why the final value is 4.4026 instead of 4.41

The final photon number is 0.17% below 4.41. This difference is real physics, and two effects cause it:

1. **The cat is not exactly two waves.** A cat with only even photon numbers has, on average,
   $$\langle n\rangle = \alpha^2 \tanh(\alpha^2) = 4.40870$$
   photons. When we switched off memory leaks, the simulation gave exactly 4.40870.
2. **Memory leaks pull it down a little more.** A simple estimate predicts a drop of
   $\kappa_a/(2\kappa_2) \approx 0.0052$. The simulation gives 0.0061, which is close.

## 7. Task 1.3(a): what state did we make?

**The answer is the "even cat".** The even cat is the superposition of the two waves $+\alpha$ and $-\alpha$:

$$
|\mathcal C_+\rangle \propto |{+\alpha}\rangle + |{-\alpha}\rangle .
$$

**Why.** Every part of the machine adds or removes memory photons only in pairs. So the parity cannot
change, except through the small memory leak. We started with zero photons, and zero is even. So the
memory must end up with only even photon numbers, and the even cat is the only state like that. When we
turned off the memory leak, the parity stayed even to within $10^{-14}$, which is computer rounding error.

**It is not one definite answer.** The chance of finding the memory in $+\alpha$ is exactly 50%.

**It is a real superposition, not a coin toss.** There is an important difference between two situations:

- A **superposition** is genuinely both waves at once.
- A **mixture** is like a coin under a cup. It is heads or tails, and we just do not know which.

Both give a 50% chance of $+\alpha$, so that number cannot tell them apart. The Wigner map can. Our
state has stripes that go *below zero*, and a mixture has no stripes. No ordinary probability can be
negative, so the stripes prove that the state is quantum.

**How long it lasts.** Each photon that leaks out of the memory flips the parity, which is a phase flip.
So the stripes slowly fade. Theory predicts that they fade at the rate

$$
\Gamma = 2\kappa_a\alpha^2 \approx 0.515 \text{ per microsecond}.
$$

We measured 0.513. So the superposition lasts about $1/\Gamma \approx 1.95$ microseconds.

## 8. Task 1.3(b): choosing one definite state

We want to make $|{+\alpha}\rangle$ alone, not the 50/50 cat. To do that, we must tip the balance toward one side.

**An analogy.** Imagine a ball on top of a hill with a valley on each side. If the ball sits exactly at
the top, it can roll either way. If we nudge it a little to the right, it rolls into the right valley.
The two valleys are $+\alpha$ and $-\alpha$.

There is a quantum twist. The ball is not a point. It is a fuzzy blob, because quantum light always
has a little random noise. So even after a nudge, part of the blob can still sit on the wrong side.

**Method A: nudge first, then start.** We first give the memory a small push of size $\beta$, and then
turn on the machine. The chance of landing in the right valley is the fraction of the blob that sits on
the right side. The blob is a bell curve centered at $\beta$ with a width of $\tfrac12$, so

$$
F \approx 1 - \tfrac12\,\mathrm{erfc}\!\left(\sqrt{2}\,\beta\right).
$$

Here $F$ is the **fidelity**, the chance that we made the right state, and erfc is the standard
function for the area in the tail of a bell curve. This formula has no adjustable numbers, and it
matches every simulation to within 0.003. A push of $\beta = 1.5$ gives $F = 0.9988$, or 99.88%.

**Method B: nudge while it starts.** We turn on the machine and give a very short push in the first
50 nanoseconds. This gives $F = 0.9985$. A very short push works exactly like Method A. A slower push
works less well, because by then the machine has already enlarged the random noise.

**Pushing too hard is bad too.** A push of size 4 lowers $F$ to about 0.955. The memory starts with
about 16 photons, which overloads the buffer, and the energy swings the state back across the hill. We
checked that this result is real. It does not change when we keep more photon levels, and it
disappears in the textbook shortcut model. So it comes from the buffer, just like the overshoot in
Task 1.1. The best push is about the size of $\alpha$.

## 9. Task 1.3(c): what does it cost?

- **An extra control wire.** Both methods need a way to push the memory directly. That wire must not
  make the memory leak more.
- **Aiming the push.** The direction of the push decides which state you get. If the aim is off by
  45°, the error grows from 0.1% to 4.3%. The aim must be correct to within about 20°.
- **Method A:** the machine must be off during the push, so the sequence takes extra time.
- **Method B:** the machine stays on, but the push must be about 10 times stronger than $g_2$. A push
  that strong could cause side effects that our model does not include.

## 10. How we know we are right

- **Hand-calculated checks.** Our 114 tests compare the code with results that we can derive on paper.
- **Control runs.** To test whether an effect is real, we switch off one ingredient and see whether
  the effect goes away.
- **Three convergence checks.** We checked the memory levels, the buffer levels, and the step-size tolerance.
- **Honest corrections.** We found and fixed our own mistakes: the default tolerance, a timing
  measurement that was 12% off, and two claims about Task 1.3 that the data did not support.

## Where these ideas come from

These are the papers behind the slides and animations.

- **Mirrahimi et al., *New Journal of Physics* (2014):** the idea of protecting a cat qubit with two-photon processes.
- **Leghtas et al., *Science* (2015):** the first experiment using a memory and a buffer to trade photons in pairs. Our "two boxes" animation shows this scheme.
- **Guillaud and Mirrahimi, arXiv:1904.09474 (2019):** why removing bit flips makes error correction much cheaper.
- **Réglade, Bocquet et al., arXiv:2307.06617 (2024):** the Alice & Bob experiment that measured bit-flip times above 10 seconds. All of our parameter values come from this paper.

The "bigger cats almost never flip" animation uses the rule that the bit-flip rate falls roughly as
$e^{-2\alpha^2}$ as the cat grows. The challenge describes this scaling as $e^{-\gamma\alpha^2}$ with
$\gamma$ about 2 or a little less, so the animation's numbers are illustrations, not measurements.

## Glossary

| Word | Meaning |
|---|---|
| Qubit | The quantum version of a bit. It can be 0, 1, or a superposition of both. |
| Photon | One packet of light energy. |
| Memory, buffer | The two boxes of microwave light. The memory stores the qubit, and the buffer helps. |
| Coherent state $\lvert\alpha\rangle$ | A clean light wave. Its average photon number is $\lvert\alpha\rvert^2$. |
| Cat state | A superposition of two opposite light waves, $+\alpha$ and $-\alpha$. |
| Parity | Whether the photon number is even or odd. |
| Superposition | A state that is genuinely several possibilities at once. |
| Mixture | A state that is one possibility, but we do not know which. |
| Wigner function | A map of the light. Negative regions show quantum behavior. |
| Fidelity $F$ | The chance that we made the state we wanted. 1 means perfect. |
| Truncation | Keeping only a limited number of photon levels in the computer. |
| Tolerance | How accurate each time step of the simulation must be. |
| Effective model | A simpler model that assumes the buffer empties instantly. |
