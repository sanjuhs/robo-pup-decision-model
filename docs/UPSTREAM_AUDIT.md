# Upstream audit

Inspected 2026-10-16 (Asia/Kolkata). Licenses and READMEs were read from upstream;
the revisions below were resolved with read-only `git ls-remote`. Upstream performance
claims have **not** been independently reproduced here.

| Project | Inspected HEAD | License | Decision |
| --- | --- | --- | --- |
| [robo-jev](https://github.com/STEERIX-home/robo-jev) | `235a0e824675bd0dbf23436543e231f2bda1f6df` | Apache-2.0 | Inspiration for separating physical judgment from motor control. No source or weights copied. |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | `0d1406da61670d085bb0bcec47b502899b188a03` | Apache-2.0 | Native headless physics for the first prototype. |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) | `d59156e099a8785dd58fbce2e25a6ee1f7b5800c` | Apache-2.0 | Audited benchmark environment option. The prototype uses a custom MJX adapter instead; Playground is not integrated. |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | `b2878f4b83e21f16a53c54a62def628e05fd2e96` | MIT | PPO, checkpointing and reproducible policy initialization. |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | `57c1610e80214db72c1c7410d6b65bb1a989e233` | MIT | Environment API and validation. |
| [Three.js](https://github.com/mrdoob/three.js) | `c94d808a4c773375bf221b13e61938e0aef9aea0` | MIT | Browser geometry and camera; bundled locally. |
| [runpodctl](https://github.com/runpod/runpodctl) | `551d6a4310767ad9671961bea9f64cb2c365dc1e` | GPL-3.0 | Separately installed operator CLI; no CLI code linked or vendored into the application. |

License URLs: each project's `LICENSE` at the recorded revision. Installed package
versions are pinned separately in `uv.lock` and `web/package-lock.json`; upstream
HEADs above are audit references, not dependency pins. Distributing dependency
binaries or browser bundles requires retaining their license notices. `npm`/Vite
preserves Three.js's license banner; the source tree includes third-party notices.

## Why the first path uses CPU physics

The available cloud instance has no GPU. Native MuJoCo + SB3 is a small stack that
can be tested here without paid provisioning. It does **not** provide GPU physics.
CUDA can accelerate neural-network optimization, but small MLP PPO is often faster
on CPU; an RTX 5090 alone does not make this simulator vectorize on GPU.

An optional custom `mujoco-mjx==3.3.7` / `jax==0.7.2` adapter was subsequently added
and exercised on an authorized RTX 5090. Both installed distributions' LICENSE
files were read and declare Apache-2.0. The GPU path batches physics in JAX and
keeps PPO optimization on CPU. See the measured results and failed-attempt record
in `BENCHMARKS.md`; this does not imply Playground or Warp integration.

The custom dog is an original primitive-geometry model, not a Unitree model or a
validated hardware digital twin. PPO learns residual joint targets around an analytic
trot. Each policy starts with fresh network weights; the gait prior is shared.
This is not end-to-end torque locomotion or sim-to-real training.

robo-jev's README explicitly limits out-of-simulator risk calibration. A 2B model,
adapter training, paid model API, and its reported benchmark scores are not needed
for this proof of concept. The local decision layer remains inspectable rules;
there is no claim that a Jev model has been trained or integrated.
