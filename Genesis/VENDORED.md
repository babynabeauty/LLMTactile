# Vendored Genesis

Upstream [Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis) at tag
`v1.4.1`, commit `a68b97b9363aa0be9ccb0c8291b021f3afdd807d` — the revision
`dexterous-hands/src/fresh_campaign.py` asserts at runtime, and the one every `fresh_v141` recipe
was trained against.

Two things are dropped from the upstream checkout, neither used by this project:

- `genesis/assets/{meshes,urdf,xml,tower}` (201 MB of demo assets). `genesis/assets/textures` is kept.
- The `doc`, `genesis/ext/LuisaRender` and `genesis/ext/ParticleMesher` submodules, which are empty
  mount points in a plain clone.

Nothing else is modified. To refresh, re-clone at the pinned commit and reapply those two removals.
