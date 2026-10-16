# Third-party notices

Original RoboPup application code and primitive robot geometry use the root MIT
license. The procedural golden-puppy skin and generated fur texture are original
artwork covered by that license; no external character files are downloaded.
No robo-jev source, model weights, Unitree assets, or MuJoCo Menagerie
meshes are redistributed.

Dependencies retain their own licenses:

- MuJoCo — Apache-2.0, Google DeepMind.
- Optional MuJoCo MJX and JAX — Apache-2.0. NVIDIA CUDA libraries installed for
  GPU use retain their own NVIDIA license terms; they are not vendored in this repository.
- Stable-Baselines3 — MIT, Antonin Raffin and contributors.
- Gymnasium — MIT, OpenAI and Farama Foundation.
- Three.js — MIT, three.js authors (full notice in `web/THREE_LICENSE.txt`).
- NumPy — BSD-3-Clause; PyTorch — BSD-style; FastAPI — MIT;
  Uvicorn — BSD-3-Clause; Pydantic — MIT. See installed package metadata for
  full notices and transitive dependencies.
- runpodctl — GPL-3.0; installed separately as an operator tool, not bundled.

See `docs/UPSTREAM_AUDIT.md` for source references and scope of verification.

Local vision/speech additions: Pillow 12.3.0 — MIT-CMU; PocketSphinx 5.0.4 —
BSD-2-Clause, Carnegie Mellon University; sounddevice 0.5.6 — MIT; cffi 2.1.1 —
MIT-0; pycparser 3.11 — BSD-style. Full distribution notices accompany installed
packages. Principal speech and synthetic-fixture notices are retained in
`licenses/`. The bundled arena planner was trained from scratch in this project
and is released under the root MIT license. No Hugging Face or Jev weights are
redistributed. The synthetic fixture says “fetch the ball” and was generated
with FFmpeg/Flite `slt`, not recorded from a human speaker.
