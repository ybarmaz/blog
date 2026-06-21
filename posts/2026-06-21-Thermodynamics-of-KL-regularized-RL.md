---
aliases:
- /2026/06/21/Thermodynamics-of-KL-regularized-RL
author: Yves Barmaz
badges: true
branch: master
categories:
- Reinforcement Learning
- thermodynamics
- geometry
date: '2026-06-21'
description: A thermodynamic view of KL-regularized RL, from Gibbs equilibria and entropy mirror descent to TRPO, PPO, GRPO, and support-driven phase transitions.
output-file: 2026-06-21-Thermodynamics-of-KL-regularized-RL.html
title: "The thermodynamics of KL-regularized RL: From mirror descent to phase transitions"
toc: false
echo: true
comments:
  utterances:
    repo: ybarmaz/blog
---

Recently, I have been trying to cast some physical intuition on the zoology of reinforcement learning algorithms used in LLM post-training. Who has never been frustrated that while gradient descent sounds like powder skiing, TRPO, PPO, and GRPO sound more like the bush-whacking part where you have to somehow get back to the car after a ski tour? Well, it turns out the snow metaphor runs deeper than the type 1/type 2 fun duality: these algorithms are all trying to melt a fixed pretrained distribution toward higher reward, and "how hard you melt it" is a temperature — one whose equilibrium can exhibit a genuine phase transition in the large-task limit. This post is about taking that analogy literally and seeing how far it goes.

## Why does RL post-training look like statistical physics?

When we fine-tune a language model with reinforcement learning under a KL penalty to the pretrained reference, we can write down the closed-form optimum as

$$\pi^*(y\mid x) \propto \pi_\text{ref}(y\mid x)\exp(r(x,y)/\tau).$$

This is a Boltzmann distribution. The KL coefficient $\tau$ sits in the denominator of the exponent, exactly where temperature sits in $p \propto e^{-E/kT}$.

Where does the closed form come from? Hold the prompt $x$ fixed and maximize the objective $\mathbb{E}_\pi[r] - \tau\,\mathrm{KL}(\pi\,\|\,\pi_\text{ref})$ over the distribution $\pi(\cdot\mid x)$, with a Lagrange multiplier $\lambda$ for the normalization $\sum_y \pi(y\mid x) = 1$. Applying the calculus of variations, the optimal policy satisfies

$$\frac{\delta}{\delta \pi(y|x)} \left[ \sum_y \pi(y|x) r(x,y) - \tau \sum_y \pi(y|x) \log \frac{\pi(y|x)}{\pi_\text{ref}(y|x)} - \lambda \left(\sum_y \pi(y|x) - 1\right) \right] = 0,$$

yielding 

$$r(x,y) - \tau\log\frac{\pi(y\mid x)}{\pi_\text{ref}(y\mid x)} - \tau - \lambda = 0,$$

and solving for $\pi$ gives the Gibbs optimum

$$\pi^*(y\mid x) = \frac{1}{Z}\,\pi_\text{ref}(y\mid x)\exp\!\bigl(r(x,y)/\tau\bigr),\qquad Z = \sum_y \pi_\text{ref}(y\mid x)\exp\!\bigl(r(x,y)/\tau\bigr),$$

where the multiplier $\lambda$ is fixed by normalization, leaving the partition function $Z$ as the proportionality constant.

The analogy is not just decorative. Assume a binary reward (each trajectory earns $r(x, y) = \Delta$ or $0$) and write $p = \pi_\text{ref}(\text{correct})$ for the reference's base success rate. Under $\pi^*_\tau$ the probability of a correct trajectory is sigmoidal in $1/\tau$, rising from $\approx p$ (regularization-dominated) to $\approx 1$ (reward-dominated) around a support-dependent critical temperature

$$\tau_c(p) = \frac{\Delta}{\log\!\bigl((1-p)/p\bigr)} \qquad (p < 1/2).$$

For fixed $p$ this is a smooth crossover, not a true singularity; §10 derives it in full and shows how task-size scaling sharpens the crossover into a genuine phase boundary.

The closed form is specific to the two-level structure of binary reward. With richer reward shapes, the same Gibbs reweighting still produces a τ-driven crossover, just smoother and without a clean midpoint. What remains in general is the phenomenology: temperature controls how sharply $\pi^*_\tau$ concentrates on high-reward trajectories, and in large structured problems this concentration can sharpen into a phase-transition-like crossover.

That equilibrium picture — a Boltzmann distribution with $\tau$ as temperature — is the foundation the rest of the post builds on. The aim is to bring some mathematical clarity to the common RL post-training algorithms (TRPO, PPO, GRPO), using concepts borrowed from thermodynamics (free energy, Boltzmann distributions, phase transitions) and from optimization geometry (mirror descent, natural gradient, Bregman divergences), organized around a simple separation of roles:

1. **Equilibrium.** The KL-regularized RL objective is a free-energy minimization on policy distributions; its minimizer is the Gibbs distribution $\pi^*_\tau$, and the support-dependent phase transition above is a direct consequence.
2. **Ideal dynamics.** Exact entropy mirror descent on the free energy $\mathcal{F}_\tau$ gives the corresponding gradient flow in policy-distribution space — an exponentiated-advantage update at each step.
3. **Practical algorithms.** TRPO, PPO, and GRPO are best read as practical approximations to that ideal flow, trading off geometric fidelity, estimator design, and iteration cost — the trade-offs the post mostly tracks.
4. **Diagnostics.** In controlled settings where $\pi_\text{ref}$, $r$, and the action space are explicit, $\pi^*_\tau$ and the ideal mirror-descent flow are computable, so the gap between an optimizer's iterates and the thermodynamic target is *measurable*, not just rhetorical. In full LLM training, the same picture becomes a guide to empirical signatures such as crossings in success rate, entropy, reference KL, or pass@k as the KL strength is swept.

These four roles are conceptual threads, not section numbers: they are developed across §§3–11 (equilibrium in §5 and §10, ideal dynamics in §§3–4 and §9, practical algorithms in §§6–8, diagnostics in §11), after the geometry background of §§1–2. Along the way the post pulls apart two distinctions that the usual expositions tend to compress (§5 — two KLs; §6 — region vs penalty), and culminates in §11 with the takeaway that the equilibrium picture is directly testable in synthetic benchmarks, while in real LLM training it appears only through empirical crossing signatures.

---

## 1. Gradient descent, viewed sideways

Most readers know gradient descent as the explicit update

$$x_{t+1} = x_t - \eta\,\nabla f(x_t).$$

A more revealing form is implicit. The update is the minimizer of a *local* problem,

$$x_{t+1} = \arg\min_x\left\{\eta\,\langle \nabla f(x_t), x\rangle + \tfrac{1}{2}\|x - x_t\|^2\right\}.$$

The first term linearizes the loss, the second term keeps the candidate near the current iterate. Gradient descent is an instance of *proximal* optimization, and its proximal penalty happens to be the squared Euclidean distance.

That choice is not innocent. Squared Euclidean distance is the right way to measure a "small movement" in flat space. It's the wrong choice on non-flat geometries, like the simplex, the manifold of probability distributions, or positive-definite cones, where a more general algorithm like [mirror descent](https://en.wikipedia.org/wiki/Mirror_descent) is needed.

## 2. Mirror descent: change the geometry

First we pick a strictly convex potential $\psi$, called the *mirror map*. Its [**Bregman divergence**](https://en.wikipedia.org/wiki/Bregman_divergence)

$$D_\psi(x, y) = \psi(x) - \psi(y) - \langle \nabla \psi(y), x - y\rangle$$

replaces the squared distance. It is nonnegative, vanishes when $x=y$, and (importantly) is generally asymmetric, $D_\psi(x,y) \ne D_\psi(y,x)$. This is a feature, not a bug, when the underlying space has direction-dependent geometry.

The mirror descent update is

$$x_{t+1} = \arg\min_x\left\{\langle \nabla f(x_t), x\rangle + \tfrac{1}{\eta}D_\psi(x, x_t)\right\}.$$

With $\psi(x) = \tfrac{1}{2}\|x\|^2$, we recover gradient descent. With $\psi$ chosen to match the geometry of the problem, we get something better.

## 3. Entropy mirror descent: multiplicative updates on the simplex

For probability vectors, the natural mirror map is the *negative entropy*

$$\psi(x) = \sum_i x_i \log x_i.$$

Its Bregman divergence is the KL divergence

$$D_\psi(x, y) = \sum_i x_i \log(x_i/y_i) = \mathrm{KL}(x\,\|\,y).$$

The mirror descent update on the simplex with this $\psi$ has a closed form

$$x_{t+1, i} \propto x_{t, i}\,\exp(-\eta\, g_{t,i}),$$

where $g_t = \nabla f(x_t)$. This is the [**multiplicative weights**](https://en.wikipedia.org/wiki/Multiplicative_weight_update_method) or **exponentiated gradient** update. The crucial difference from Euclidean gradient descent is that updates are *multiplicative*, not additive. Positivity and normalization are preserved automatically. Coordinates with negative gradient are upweighted exponentially; coordinates with positive gradient are suppressed exponentially.

This is one of the cleanest examples of why "geometry-aware" optimization is more than a metaphor. The KL geometry of the simplex turns gradient descent into something that respects its constraints.

## 4. Policies are distributions

A policy $\pi(a\mid s)$ is a conditional distribution. As a result, the search space of a policy optimization algorithm is a family of distributions, and it can be framed as a mirror-descent problem with KL as the Bregman divergence.

Concretely, we consider a per-state objective of maximizing the expected advantage of the new policy, but penalizing how far it moves from the old policy in KL.

$$\pi_{t+1}(\cdot\mid s) = \arg\max_{\pi}\left\{\mathbb{E}_{a\sim \pi}[A_t(s,a)] - \tfrac{1}{\eta}\,\mathrm{KL}\bigl(\pi(\cdot\mid s)\,\|\,\pi_t(\cdot\mid s)\bigr)\right\}.$$

The closed-form solution is the **exponentiated-advantage update**,

$$\pi_{t+1}(a\mid s) \propto \pi_t(a\mid s)\,\exp\bigl(\eta\,A_t(s, a)\bigr).$$

This is the policy-space analogue of multiplicative weights. The old policy is the reference measure, the advantage $A_t$ reshapes it, $\eta$ controls how aggressively. *Bad actions are exponentially suppressed and good actions are exponentially amplified.*

This looks superficially similar to the Gibbs formula $\pi^* \propto \pi_\text{ref}\exp(r/\tau)$, but the roles are different: $r$ defines the final equilibrium, while $A_t$ is the local slope used for one update. §5 makes that distinction precise.

So far this is standard textbook material. The next two sections pull apart distinctions that textbook presentations tend to compress.

---

## 5. Two KLs: landscape versus timestep

There is a subtle trap in the phrase "KL-regularized RL." It sounds like there is one KL term doing one job. In practice there are two, and they live at different levels of the story.

One KL is part of the **objective**. It says what policy we ultimately want.

The other KL is part of the **algorithm**. It says how far we are willing to move in one update.

Conflating them is the fastest way to turn the thermodynamic analogy into mush.

### The reference KL: the landscape

The first KL is the reference penalty,

$$\mathrm{KL}(\pi\,\|\,\pi_\text{ref}).$$

It compares the trained policy to a fixed pretrained reference. It enters the global objective

$$\mathcal{F}_\tau(\pi) = -\mathbb{E}_\pi[r] + \tau\,\mathrm{KL}(\pi\,\|\,\pi_\text{ref}).$$

This is the free energy. The reward term wants to concentrate probability on high-reward trajectories; the KL term resists moving too far from the pretrained model. The coefficient $\tau$ is the thermodynamic temperature. It is not a step size — it determines the equilibrium.

Minimizing this free energy gives

$$\pi^*_\tau(y\mid x) \propto \pi_\text{ref}(y\mid x)\exp(r(x,y)/\tau).$$

So changing $\tau$ changes the policy you are trying to reach. Large $\tau$ keeps the optimum close to the reference; small $\tau$ lets reward dominate and concentrates mass on high-reward trajectories. This is the object behind the phase-transition picture.

### The proximal KL: the integrator

The second KL is the proximal penalty,

$$\mathrm{KL}(\pi\,\|\,\pi_k).$$

It compares the candidate next policy to the current policy. It is not part of the final objective; it is part of the numerical method used to move through policy space.

A mirror-descent step takes a local linear approximation to the objective and regularizes the step by KL distance to the current iterate,

$$\mathcal{G}_\eta(\pi;\pi_k) = -\mathbb{E}_\pi[\hat A_k] + \frac{1}{\eta}\,\mathrm{KL}(\pi\,\|\,\pi_k).$$

Here $\eta$ is a step size. The coefficient $1/\eta$ looks temperature-like, but it is only a *per-step* temperature: it controls how much policy mass is allowed to move in one update, not where the trajectory ends up.

That gives the clean separation:

|  | **Reference KL** | **Proximal KL** |
|---|---|---|
| Form | $\mathrm{KL}(\pi\,\|\,\pi_\text{ref})$ | $\mathrm{KL}(\pi\,\|\,\pi_k)$ |
| Reference policy | Fixed pretrained model | Previous iterate |
| Lives in | Global objective | Local update rule |
| Coefficient | $\tau$ | $1/\eta$ |
| Meaning | Thermodynamic temperature | Algorithmic step scale |
| Controls | Where the flow should end | How the flow is discretized |

In short, $\tau$ chooses the landscape, and $\eta$ chooses the timestep.

### The missing link: what is the advantage?

The two KLs are connected only if the local slope $\hat A_k$ is the slope of the reference-KL free energy, obtained by differentiating

$$\mathcal{F}_\tau(\pi) = -\sum_a \pi(a)\,r(a) + \tau\sum_a \pi(a)\log\frac{\pi(a)}{\pi_\text{ref}(a)}$$

with respect to $\pi(a)$. Up to constants that vanish after normalization, we obtain

$$-\nabla_\pi \mathcal{F}_\tau\big|_a = r(a) - \tau\log\frac{\pi(a)}{\pi_\text{ref}(a)} - \tau.$$

So the free-energy gradient is not the bare reward. It is the reward corrected by the current KL pressure away from the reference,

$$r(a) - \tau\log\frac{\pi_k(a)}{\pi_\text{ref}(a)}.$$

This is the advantage the proximal step must see if it is to integrate the gradient flow of $\mathcal{F}_\tau$.

With the right slope, the mirror-descent step is

$$\pi_{k+1} = \arg\max_\pi \left\{ \mathbb{E}_\pi[\hat A^\tau_k] - \frac{1}{\eta}\,\mathrm{KL}(\pi\,\|\,\pi_k) \right\},$$

where

$$\hat A^\tau_k \;\sim\; r - \tau\log\frac{\pi_k}{\pi_\text{ref}} - \text{baseline}.$$

This yields the exponentiated update

$$\pi_{k+1}(a) \propto \pi_k(a)\exp\!\bigl(\eta\,\hat A^\tau_k(a)\bigr).$$

In the per-state bandit picture, this is why subtracting a baseline is harmless. Adding or subtracting a constant from every action advantage multiplies all exponentiated weights by the same factor, and that factor disappears when the policy is normalized. What matters is not the absolute level of the free-energy slope, but which actions are above or below the current average.

In sequential RL, the familiar value baseline plays the analogous role for the policy-gradient estimator: it removes an action-independent component of the return, reducing variance without changing the expected policy-gradient direction.

Now the roles are clean:

- $\tau$ appears *inside* the advantage, because it defines the free-energy landscape.
- $\eta$ appears *outside* the advantage, because it defines the step size through that landscape.

Drop the reference-KL correction from $\hat A_k$, and the same mirror-descent machinery no longer flows toward the Gibbs policy $\pi^*_\tau$, but toward high bare reward, eventually concentrating on $\arg\max r$ as aggressively as the proximal KL allows.

So the two KLs do different jobs:

- **Reference KL shapes the equilibrium.**
- **Proximal KL shapes the trajectory.**

We need the first to define the thermodynamic target, and the second to define a stable numerical path toward it. Mixing them up makes it look as if the trust-region step size $\eta$ and the thermodynamic temperature $\tau$ are the same parameter, which they are not.

## 6. TRPO: from a region to a multiplier

Sections §6–§8 now move from the ideal flow to the algorithms people actually run, from TRPO, which keeps the KL geometry but only locally, to PPO, which keeps the trust-region intuition but drops the constrained solve, and finally GRPO, which keeps PPO's surrogate but changes the advantage estimator. This section starts with [TRPO](https://arxiv.org/abs/1502.05477), because it is the closest practical descendant of entropy mirror descent.

The instructions behind TRPO are "improve the policy, but do not move too far."

In distribution space, the cleanest version of that idea is the trust-region problem

$$
\max_{\pi}\,\mathbb{E}_{a\sim\pi}[\hat A(a)]
\quad\text{s.t.}\quad
\mathrm{KL}(\pi\,\|\,\pi_\text{old}) \le \delta.
$$

The KL is in the forward direction, $\mathrm{KL}(\pi\,\|\,\pi_\text{old})$, because this is the direction that matches the entropy mirror-descent update of §4. The new policy is allowed to improve the local surrogate, but only within a KL ball around the old one.

At first glance this is a constrained optimization problem, not a mirror-descent step. The bridge is the Lagrange multiplier. For a nontrivial advantage direction, the unconstrained linearized objective would put all mass on the best action, typically far outside the KL ball. The constraint therefore binds. By the [KKT conditions](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions), the constrained optimum can be written equivalently as the solution of a penalized problem

$$
\max_\pi \left\{ \mathbb{E}_{a\sim\pi}[\hat A(a)] - \lambda^*\,\mathrm{KL}(\pi\,\|\,\pi_\text{old}) \right\},
$$

where $\lambda^*>0$ is chosen so that the KL constraint is active. This equivalence is not for an arbitrary penalty coefficient; $\lambda^*$ is the particular multiplier selected by the active constraint.

Setting

$$
\eta = \frac{1}{\lambda^*},
$$

we recover exactly the proximal objective from §5,

$$
\max_\pi \left\{ \mathbb{E}_{a\sim\pi}[\hat A(a)] - \frac{1}{\eta}\,\mathrm{KL}(\pi\,\|\,\pi_\text{old}) \right\}.
$$

The solution is the exponentiated-advantage update

$$
\pi_\text{new}(a)
\propto
\pi_\text{old}(a)\exp(\eta\,\hat A(a)).
$$

So, in the clean distribution-space derivation, TRPO is mirror descent. It first linearizes the objective, then takes the largest safe step in KL geometry.

Practical TRPO implements a local version of this idea in neural-network parameter space. Instead of solving the exact distribution-space problem, it linearizes the surrogate in parameters and approximates the KL constraint to second order,

$$
L(\theta)
\approx
L(\theta_k) + g^\top(\theta-\theta_k),
$$

$$
\mathrm{KL}(\pi_{\theta_k}\,\|\,\pi_\theta)
\approx
\frac{1}{2}(\theta-\theta_k)^\top F_k(\theta-\theta_k),
$$

where $F_k$ is the Fisher information matrix. The resulting step is the natural-gradient direction

$$
\Delta\theta \propto F_k^{-1}g,
\qquad
\|\Delta\theta\|_{F_k}^2 = 2\delta.
$$

In this sense, TRPO is the practical algorithm closest to the mirror-descent story. It replaces the exact KL geometry of policy space by its local Fisher approximation in parameter space.

There is one technical subtlety, though. Practical TRPO is often implemented with the reverse KL,

$$
\mathrm{KL}(\pi_\text{old}\,\|\,\pi),
$$

because expectations under the old policy are directly estimable from old-policy rollouts. Forward and reverse KL give different finite-step optima, the reverse-KL solution is not the exponentiated tilt above. But infinitesimally they agree, both KL directions have the same second-order expansion, namely the Fisher metric. Therefore the clean mirror-descent derivation should be read as the small-step, distribution-space idealization of TRPO, not as the exact finite-step implementation.

This also connects back to the two-KL distinction from §5. The multiplier $\lambda^*$, or equivalently $\eta = 1/\lambda^*$, is an algorithmic step scale: it is chosen by the trust-region constraint and can change from update to update. It is not the thermodynamic temperature $\tau$, which belongs to the fixed reference-KL objective and determines the Gibbs target.

Likewise, TRPO follows the Gibbs/free-energy flow only when the surrogate advantage is the free-energy advantage from §5,

$$r(a) - \tau\log\frac{\pi_k(a)}{\pi_\text{ref}(a)} - \text{baseline}.$$

With a bare-reward advantage, TRPO remains a trust-region policy optimizer, but it is not integrating the KL-regularized free-energy flow whose minimizer is $\pi^*_\tau$.

There is one more practical point to note, because it recurs below. The advantage $\hat A$ is not handed to us, it is *estimated*. TRPO, like the policy-gradient methods it descends from, is **actor-critic**. In other words, a learned value function (the critic) supplies $\hat A$, usually through generalized advantage estimation (GAE). This is a second approximation stacked on the small-step/Fisher one. Even when the advantage is the right (KL-regularized) one in expectation, the critic is biased and noisy, so TRPO tracks the ideal free-energy gradient only as well as the critic estimates it. PPO inherits this same critic, and GRPO's defining move (§8) is to throw it away.

## 7. PPO: a clipped actor-critic surrogate

TRPO's per-update step has two geometric ingredients, namely a constraint (maximize the surrogate subject to a KL trust region) and a natural-gradient solve (use the Fisher geometry to turn that constrained policy-space problem into a parameter-space update).

[PPO](https://arxiv.org/abs/1707.06347) keeps the goal but changes the machinery. It replaces the hard KL constraint with a clipped likelihood-ratio objective, and it replaces the natural-gradient solve with ordinary first-order optimization, usually Adam over several epochs on the same rollout batch.

Concretely, given rollout samples from the old policy, the importance ratio is defined as

$$\rho_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}.$$

PPO's clipped actor objective is

$$L^\text{CLIP}(\theta) = \mathbb{E}_t\left[ \min\left( \rho_t(\theta)\hat A_t,\; \operatorname{clip}\bigl(\rho_t(\theta),\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A_t \right) \right].$$

The advantage $\hat A_t$ is the same critic estimate as in TRPO (§6). PPO inherits that estimation unchanged and changes only the geometry.

PPO then has two distinct approximation gaps:
- The first is geometric. The clip is not a true KL constraint. Once $\rho_t$ leaves $[1-\epsilon, 1+\epsilon]$ in the direction that would improve the objective, the clipped term stops rewarding further movement. This discourages large likelihood-ratio changes, but it does not enforce a global trust region, and it does not produce a natural-gradient step. The underlying parameter-space direction is whatever Adam produces on the clipped surrogate.
- The second is statistical. The critic-estimation gap is inherited from §6 rather than introduced. In ordinary RL, it is mostly harmless, but for the Gibbs-flow reading it matters. PPO follows the flow of $\mathcal{F}_\tau$ only as well as $\hat A_t$ estimates the KL-regularized advantage, and only if the reference-KL correction is actually present in the reward or loss.

So PPO shares TRPO's intent — improve the policy without moving too far — but drops both of TRPO's geometric ingredients, i.e. the formal KL constraint and the Fisher/natural-gradient solve, while keeping TRPO's actor-critic estimation: a learned critic whose job is not to define the geometry, but to estimate the actor's local direction. This makes PPO cheap, stable, and effective in many settings, but it is not mirror descent in any clean mathematical sense. The mirror-descent picture tells us what PPO is *aiming at*; whether its clipped actor update, critic-based advantage estimate, and Adam step actually track the policy-space mirror-descent direction is an empirical question, addressed in §11.

## 8. GRPO: critic-free policy gradient for LLM RL

LLM RL has a structural simplification that PPO doesn't exploit: it's a **contextual bandit**. Each prompt typically produces one response and a terminal reward, with no within-trajectory bootstrapping needed. PPO carries over the full apparatus of sequential RL (a learned value head $V_\phi$, GAE for advantage estimation) that's overkill for this structure. [**Group Relative Policy Optimization**](https://arxiv.org/abs/2402.03300) (GRPO), introduced by DeepSeek and used to train DeepSeek-R1, drops the critic and replaces it with a per-prompt Monte Carlo baseline.

For each prompt $x$, it samples a *group* of $G$ completions $\{y_i\}_{i=1}^G$ from the current policy and defines the **group-relative advantage**

$$\hat A_i = \frac{r(x, y_i) - \bar r_G}{\sigma_G + \varepsilon},\qquad \bar r_G = \tfrac{1}{G}\sum_j r(x, y_j),\quad \sigma_G^2 = \tfrac{1}{G}\sum_j (r(x, y_j) - \bar r_G)^2.$$

The mean-centering $\bar r_G$ is a Monte Carlo estimate of the prompt-conditional value

$$V^\pi(x) = \mathbb{E}_{y\sim\pi(\cdot\mid x)}[r(x, y)].$$

This is a Monte Carlo prompt-level baseline. Two finite-sample subtleties are worth flagging. First, because the same sample $r_i$ contributes to the group mean $\bar r_G$, the resulting estimator is shrunk by a factor $(1 - 1/G)$ relative to the ideal $(r_i - V^\pi(x))$ baseline; a leave-one-out baseline removes the factor exactly, but in practice the rescaling is absorbed into the learning rate. Second, the group-mean variance scales as $1/G$, so groups of size $G \in [4, 16]$ are typical.

The full GRPO loss reuses PPO's clipped likelihood-ratio surrogate, with the group-relative $\hat A$ in place of GAE-from-critic, plus an explicit reference-KL penalty,

$$L_\text{GRPO}(\theta) = \mathbb{E}\!\left[\min\bigl(\rho_i(\theta)\hat A_i,\,\mathrm{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon)\hat A_i\bigr)\right] - \beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_\text{ref}).$$

At the level of the actor objective, GRPO is PPO with a different advantage estimate: the same clipped likelihood-ratio surrogate and the same reference-KL penalty (GRPO's $\beta$ plays the role of the reference-KL temperature/penalty coefficient $\tau$), but with group-relative advantages instead of critic/GAE advantages. What that shared structure implies thermodynamically (same equilibrium, same phase boundary) will be discussed in §§10–11, once the equilibrium and its dynamics are made explicit.

**The standard normalization** brings an added benefit of scale robustness. Dividing the centered reward by the within-group standard deviation $\sigma_G$ makes $\hat A_i$ invariant to reward scale: multiplying all rewards by a constant leaves the gradient unchanged. It also makes the per-prompt advantage scale comparable across prompts with different intrinsic reward variances — a noisy-reward prompt doesn't get systematically larger updates than a sharp-reward one. Edge cases warrant care: an all-tie group (e.g. binary reward with all completions correct or all wrong) has $\sigma_G = 0$ and centered differences also zero, so it produces no learning signal (the $\varepsilon$ in $\sigma_G + \varepsilon$ keeps the division finite, but the numerator already vanishes). This standardization is loosely in the spirit of an adaptive trust region — the local reward landscape sets the per-prompt advantage scale rather than a global hyperparameter — but it isn't a strict equivalent of TRPO's $\lambda^*$. It changes the *trajectory* (dynamics of approach), not the *equilibrium*; the Gibbs distribution $\pi^*_\tau$ is unaffected.

So GRPO is not just an aside. It is arguably a *cleaner* fit to the mirror-descent picture for LLM RL than PPO. The bandit structure makes the Monte Carlo baseline natural, and within-group normalization standardizes the advantage scale across prompts.

### The approximation ladder

Summarizing §§3–8 in one table:

| | preserves | drops | price |
|---|---|---|---|
| **Entropy mirror descent** (§§3–4) | exact gradient flow of $\mathcal{F}_\tau$ on the simplex | nothing — it's the ideal | tabular only; not implementable on neural policies |
| **TRPO** (§6) | trust-region budget + KL geometry to second order via Fisher | exact policy-space KL (kept only to quadratic order) | per-step CG solve + Fisher–vector products + line search |
| **PPO** (§7) | trust-region *intent* — "don't move too far per update" | the Fisher solve; the formal constraint | realized parameter-space direction can misalign with mirror descent (§11) |
| **GRPO** (§8) | same equilibrium and surrogate as PPO | the learned value critic | higher-variance advantage; $\sim(1-1/G)$ shrinkage in the baseline |

Each row trades a different part of the ideal picture for a runnable training loop. The §11 diagnostic pairs naturally with this ladder: it asks how much fidelity is actually retained at each step, in practice.

## 9. From mirror-descent steps to probability flows

So far we have treated mirror descent as a discrete update: linearize the objective, penalize movement in KL, and take an exponentiated-advantage step. But if the step size becomes infinitesimal, the same object becomes a flow of probability mass.

Starting from the exponentiated update

$$\pi_{k+1}(a) \propto \pi_k(a)\exp(\eta A(a)),$$

expanding $\exp(\eta A)\approx 1+\eta A$ for small $\eta$ and renormalizing yields

$$\pi_{k+1}(a)-\pi_k(a) \approx \eta\,\pi_k(a)\bigl(A(a)-\mathbb{E}_{\pi_k}[A]\bigr),$$

or in continuous time,

$$\partial_t\pi_t(a) = \pi_t(a)\bigl(A(a)-\mathbb{E}_{\pi_t}[A]\bigr).$$

This is the [**replicator equation**](https://en.wikipedia.org/wiki/Replicator_equation). Actions whose advantage is above average gain mass; actions whose advantage is below average lose mass. In this setting it is the natural-gradient flow of the free energy $\mathcal{F}_\tau$ under information geometry, provided $A$ is the KL-regularized advantage. With a bare reward, the same dynamics concentrate on $\arg\max r$; with the KL-shaped advantage, the fixed point is the Gibbs policy $\pi^*_\tau$.

PPO complicates this ideal picture. In a small-step caricature, clipping replaces the exponential response to advantage by a saturating response, a kind of flux-limited distortion of the clean flow. But this is only an analogy. In a neural policy, the realized PPO step is produced by Adam on a clipped surrogate in parameter space, and it need not align with the exact distribution-space flow. That gap is what §11 discusses: in synthetic benchmarks it can be measured directly, while in full LLM training it appears only through indirect signatures.

A different geometry on the policy space, Wasserstein rather than Fisher, would lead to a Fokker–Planck drift–diffusion equation with the same Gibbs stationary law. That picture is natural for continuous action spaces with a meaningful ground metric, but not for language-model completions, and it is not the geometry implemented by TRPO, PPO, or GRPO. This emphasizes that the Gibbs equilibrium belongs to the free energy, while the transient flow depends on the chosen geometry.

## 10. The attractor: Gibbs equilibrium and support thresholds

The previous section was about dynamics. This one is about the attractor.

For the KL-regularized free energy

$$\mathcal{F}_\tau(\pi) = -\mathbb{E}_\pi[r] + \tau\,\mathrm{KL}(\pi\,\|\,\pi_\text{ref}),$$

the equilibrium policy is

$$\pi^*_\tau(a) \propto \pi_\text{ref}(a)\exp(r(a)/\tau).$$

This is the thermodynamic core of the post. The pretrained policy is the reference measure, reward is negative energy, and $\tau$ is temperature. Large $\tau$ keeps the equilibrium close to the reference; small $\tau$ lets reward dominate.

The simplest support-threshold calculation is the binary-reward case. Suppose a trajectory is either correct, with reward $\Delta$, or incorrect, with reward $0$. Let

$$p = \pi_\text{ref}(\text{correct})$$

be the reference probability of a correct trajectory. Under the Gibbs policy,

$$P^*(\text{correct}) = \frac{p\,e^{\Delta/\tau}}{p\,e^{\Delta/\tau} + 1 - p}.$$

This is a logistic function of $1/\tau$. Its midpoint occurs when the tilted correct mass equals the incorrect mass,

$$p\,e^{\Delta/\tau} = 1 - p,$$

so

$$\tau_c(p) = \frac{\Delta}{\log((1-p)/p)}, \qquad p < 1/2.$$

For fixed finite $p$, this is a smooth crossover, not a singularity. The phase-transition language enters through task-size scaling. If correct behavior requires $K$ independent steps, and the reference gets each one right with probability $q$, then

$$p_K = q^K.$$

If the reward gap also scales extensively, $\Delta_K = K\delta$, then

$$\tau_c \approx \frac{K\delta}{K|\log q|} = \frac{\delta}{|\log q|}.$$

The critical scale remains finite as $K\to\infty$, and the crossover sharpens. If $\Delta$ is held fixed instead, $\tau_c\to 0$, so the boundary collapses onto zero temperature.

This gives the support-limited reading of RL post-training. Here "support" should be read not merely as nonzero probability, but as usable reference probability mass on the relevant trajectories. In the finite binary problem, larger reference support $p$ increases the critical temperature: the correct trajectory already has more mass, so a weaker reward tilt is enough to make it dominate.

In the large-task scaling the more meaningful parameter is the per-step support $q$, since the total support is $p_K = q^K$ and the extensive-reward limit above gives $\tau_c \to \delta/|\log q|$. A model with larger per-step competence $q$ has smaller $|\log q|$, hence larger $\tau_c$: it can enter the reward-dominated regime even under a stronger KL penalty. A model with poor per-step support has small $q$, large $|\log q|$, and therefore a much smaller $\tau_c$. If the relevant trajectory is effectively outside the reference support, Gibbs reweighting cannot manufacture it: RL can amplify probability mass, but it cannot amplify mass that is not there.

This is only the equilibrium story. It tells us what exact KL-regularized optimization would target. It does not tell us whether PPO, GRPO, or any finite neural optimizer actually reaches that target. That distinction is the point of the diagnostic section that follows.

## 11. From equilibrium to empirical crossings

The thermodynamic picture gives a clean equilibrium and an ideal flow, but it does not guarantee that a practical optimizer actually follows them. The Gibbs policy,

$$\pi^*_\tau \propto \pi_\text{ref}\exp(r/\tau),$$

is the target of the KL-regularized objective; entropy mirror descent is the ideal policy-space flow toward it. TRPO, PPO, and GRPO are practical approximations to that picture, each preserving different pieces of the geometry.

In a synthetic benchmark, this distinction can be tested directly. If the reward, reference policy, and action space are simple enough, $\pi^*_\tau$ is computable. One can then compare exact mirror descent, TRPO-like natural-gradient updates, PPO, and GRPO-style estimators against the known Gibbs target. The relevant questions are no longer rhetorical: does the optimizer reach $\pi^*_\tau$? How much proximal KL does it spend? Is the realized update direction aligned with the exact mirror-descent direction? Does the observed transition occur at the predicted $\tau_c(p)$?

In full LLM post-training, those diagnostics are not directly available. The action space is the space of completions, the reward may be learned or noisy, and the exact Gibbs distribution over all outputs is inaccessible. There, the thermodynamic picture is better read as a guide to what signatures to look for rather than as a directly computable target.

The main empirical signature is a crossing. As the effective temperature is lowered — equivalently, as the reference-KL penalty is weakened relative to reward — the theory predicts a transition from a reference-dominated regime to a reward-dominated regime. One should therefore look for sharp changes in success rate, entropy, reference KL, pass@k, or related order parameters as the KL strength is swept. If the support-threshold picture is right, the crossing should depend strongly on base-model support: models with better pretrained support should cross at a less extreme KL penalty, while weaker models should require a colder objective or fail to cross at all.

This is also where optimizer effects enter. The phase boundary belongs to the free-energy landscape, but the trajectory belongs to the optimizer. TRPO, PPO, and GRPO may share the same equilibrium target when they use the same reference-KL objective, but they need not approach it in the same way. PPO may improve reward while drifting away from the ideal KL/Fisher flow; GRPO may change the finite-time trajectory through group-baseline variance and normalization; TRPO should be closest to the local natural-gradient picture, modulo critic error, finite steps, and parameterization.

So the honest status is:

* in synthetic environments, the thermodynamic picture gives a target that can be tested directly;
* in real LLM RL, it gives a phase-transition hypothesis and a set of observable signatures;
* the boundary is an equilibrium claim, while the observed crossing is filtered through optimizer dynamics, sampling, reward noise, and finite compute.

That does not weaken the analogy. It clarifies what kind of claim it is. The Boltzmann formula defines the equilibrium; the phase-transition picture predicts how that equilibrium should change with temperature and support; practical training runs then tell us whether a given optimizer and model actually exhibit the predicted crossing.

## 12. What the analogy does not claim

The thermodynamic language is useful because the KL-regularized optimum is literally a Gibbs reweighting. But the analogy should not be read as a proof that practical RL algorithms implement equilibrium statistical mechanics.

First, the clean derivations live in distribution space. Entropy mirror descent acts directly on policies, and the Fisher metric is the local geometry induced by KL. TRPO approximates this geometry in neural-network parameter space; PPO drops the Fisher solve altogether and optimizes a clipped surrogate with Adam. Far from the current iterate, parameter geometry and distribution geometry can decouple.

Second, the KL direction matters at finite step size. The mirror-descent derivation uses $\mathrm{KL}(\pi\,\|\,\pi_\text{old})$, while practical trust-region methods often estimate $\mathrm{KL}(\pi_\text{old}\,\|\,\pi)$ because old-policy samples are available. Infinitesimally the two agree through the Fisher metric, but their finite-step optima are different.

Third, the equilibrium picture is not the same as the training trajectory. State-distribution shift, finite batches, critic error, clipping, Adam dynamics, reward noise, and limited sampling all affect what happens in practice. PPO’s clipping, in particular, discourages large likelihood-ratio changes but does not impose a true global KL constraint.

So the claim is not that TRPO, PPO, or GRPO exactly implement the Gibbs flow. The claim is narrower: KL-regularized RL defines a Gibbs equilibrium and an ideal KL/Fisher flow, and this gives a useful reference geometry for understanding where practical algorithms agree with the theory and where they depart from it.


## 13. Target, flow, trajectory

KL-regularized RL has a clean thermodynamic core. Once a reference policy, reward, and temperature are fixed, the equilibrium is a Gibbs reweighting,

$$\pi^*_\tau \propto \pi_\text{ref}\exp(r/\tau).$$

That equilibrium can exhibit a support-dependent transition: when pretrained mass on good trajectories is large enough, reward amplification takes over; when it is too small, the KL penalty keeps the policy reference-dominated unless the system is run much colder. Entropy mirror descent gives the ideal KL/Fisher flow toward this equilibrium. TRPO, PPO, and GRPO are practical approximations to that geometry, preserving different pieces of it and losing others.

The main lesson is not that PPO or GRPO literally implement statistical mechanics. They do not. The lesson is that the thermodynamic picture separates three things that are often conflated: the equilibrium target, the ideal policy-space flow, and the finite-time trajectory of the optimizer. The boundary belongs to the free energy; the trajectory belongs to the algorithm.

The thermodynamic vocabulary is not just a metaphor. It is the right language for a problem whose closed-form optimum is literally a Boltzmann distribution — and the right language for asking, precisely, when the optimizer you ran is actually following the dynamics that vocabulary names.
