## HoDEL: Homotopy-inspired Differentiable Energy Learning from Equilibrium Shapes

### Getting Started

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager.
2. In the root folder of this repository, execute `uv sync`.

> If you have CUDA device, `uv sync --extra cuda` installs `jax[cuda13]`.

### Figures

Within `figures`, we have three folders with additional `README.md` on how to run all relevant benchmark and visualizations. If you are installing from GitHub (or supplemental materials), various `.npz` will not exist for various `.ipynb`. All upstream datasets will be included.

- [x] 1D Arruda Boyce deformed in FEA.
- [x] 3D rod deformed in Dismech.
- [x] 3D Wide ribbon deformed in FEA.

