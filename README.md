# TorchFEA — A PyTorch Finite-Element Analysis Framework

**Developed by Zenan Song**

TorchFEA is a research-oriented finite-element framework built on PyTorch. It
provides a single object model for parts, assemblies, materials, loads,
constraints, solvers, and results, with automatic differentiation available
through the same tensor-based formulation. Models can be authored from Python,
imported from Abaqus-style `.inp` files, or built interactively in the
Gmsh/OCC-based CAE interface.

## Features

- Nonlinear three-dimensional solid mechanics with PyTorch tensors and
  float64 arithmetic
- Static implicit Newton–Raphson solving with line search and sparse
  Pardiso/Conjugate-Gradient linear algebra
- Dynamic implicit (Newmark-β) and dynamic explicit (central-difference)
  solvers
- Neo-Hookean, Mooney–Rivlin, Yeoh, Gent, Arruda–Boyce, Ogden, and
  Saint Venant–Kirchhoff/linear-elastic material models
- Pressure, body force, concentrated force, moment, springs, penalty degrees
  of freedom, self-contact, and body-to-body contact
- Reference points, node coupling, displacement boundaries, named node/element
  sets, and surface sets
- PyTorch automatic differentiation for Jacobians, sensitivities, and
  optimization workflows
- Native Gmsh OCC CAD histories with boxes, wedges, cylinders, spheres, cones,
  tori, STEP import, transforms, and Boolean operations
- Abaqus-style CAE preprocessing and postprocessing UI with BREP/mesh views,
  selection, contour plots, deformation-gradient, stress, strain, and result
  export
- Native `.npz` model/result serialization with model check codes for result
  validation

## Installation

TorchFEA requires Python ≥ 3.12. A PyTorch **float64** build is recommended;
CUDA is optional and can be selected through PyTorch's normal device settings.

Install a released package:

```bash
pip install torchfea
```

Install the current checkout while developing:

```bash
pip install -e .
```

The GUI entry point is installed with the package:

```bash
torchfea-ui
```

When running directly from the source tree, use:

```bash
python -m torchfea.ui
```

## Quick start: INP model and static solve

The following is the same construction pattern used by the pressure examples:

```python
from pathlib import Path

import torch
import torchfea

torch.set_default_dtype(torch.float64)

inp = torchfea.FEA_INP()
inp.read_inp(Path("examples/models/C3D4.inp"))

fe = torchfea.from_inp(inp)
fe.solver = torchfea.solver.StaticImplicitSolver(tol_error=1e-8)

fe.assembly.add_load(
    torchfea.loads.Pressure(
        instance_name="final_model",
        surface_set="surface_1_All",
        pressure=0.06,
    ),
    name="pressure-1",
)
fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(
        instance_name="final_model",
        set_nodes_name="surface_0_Bottom",
        indexDoF=[0, 1, 2],
    ),
    name="fixed-bottom",
)

result = fe.solve()
print(result.converged, result.GC.shape)

fe.save_model("model.npz")
result.save("result.npz")
```

The same `Assembly` API is used for reference points, coupling constraints,
springs, contact, additional instances, materials, and boundary conditions.
See [`docs/usage.md`](docs/usage.md) for complete examples.

## CAD construction from Python

CAD is represented as a serializable feature history owned by
`FEAController.cad`. The history is replayable and independent of the UI:

```python
from torchfea import FEAController
from torchfea.cad import Box, Cylinder, Cut

fe = FEAController()
history = fe.cad.add_part("Part-1")

history.add(Box("Box-1", dx=20, dy=10, dz=5))
history.add(Cylinder("Cylinder-1", x=10, y=5, radius=2, dz=5))
history.add(Cut("Cut-1", object_id="Box-1", tool_id="Cylinder-1"))

# Generate a C3D4 Part with one global mesh size.
part = fe.cad.mesh_part("Part-1", mesh_size=1.5)
```

Every feature is an explicit serializable class with a discoverable Python
signature. The available operations are `Box`, `Wedge`, `Cylinder`, `Sphere`,
`Cone`, `Torus`, `StepImport`, `Fuse`, `Cut`, `Intersect`, `Fragment`,
`Translate`, `Rotate`, `Dilate`, and `Mirror`. `CADHistory` also provides typed
convenience methods such as `add_box`, `cut`, and `rotate`.
After meshing, attach the returned native `Part` to an `Assembly`, assign
materials and density to its element blocks, add an `Instance`, and define the
analysis conditions through the normal model API. The CAE UI performs these
same kernel operations and saves the history through `FEAController.save_model`.

## CAE workflow

The graphical preprocessor and postprocessor are launched with
`torchfea-ui`. The preprocessor supports:

1. Gmsh/OCC geometry histories and Boolean features
2. Part materials, density, node sets, element sets, and surfaces
3. Reference points, instances, boundaries, couplings, and loads
4. One project-wide mesh size and FE mesh generation
5. Part/assembly geometry and mesh views with BREP picking
6. Native model save, INP export, and STEP import/export

The postprocessor can inspect all instances or one instance, display
displacement, deformation-gradient, Green–Lagrange strain, Cauchy stress,
Mises values, volume ratio, and strain energy, and export `.npz` or `.csv`
results. Both CAD and mesh views provide standard camera views and section
clipping.

An `.npz` created from an INP model is treated as a discrete-mesh model: it
retains the existing nodes, elements, sets, materials, loads, and constraints
without inventing a CAD history or remeshing it. A CAD-created model retains
its OCC history and can be rebuilt in the UI.

See [`docs/cae_ui.md`](docs/cae_ui.md) for the complete CAE workflow and
[`docs/materials.md`](docs/materials.md) for constitutive equations and
small-strain parameter matching.

## Persistence and result validation

Use the controller serializer for a complete native model:

```python
import torchfea

model = torchfea.load_model("model.npz")
result = model.solve()
```

`FEAController.save_model` stores the assembly, discrete parts, CAD histories,
materials, sets, loads, constraints, boundaries, and solver settings in the
native torchfea representation. `StaticResult.save/load` and
`DynamicResult.save/load` store solver results separately. Model and result
archives carry a compact model check code; the UI rejects a result whose code
or degree-of-freedom layout does not match the loaded model.

## Repository layout

```text
src/torchfea/
├── controller.py       # model construction, solve, and serialization
├── cad/                # Gmsh OCC history and meshing
├── model/              # Assembly, Part, elements, materials, loads, BCs
├── solver/             # static and dynamic solvers/results
└── ui/                 # PySide6 + VTK CAE interface

docs/                   # usage, architecture, solver, loads, and CAE guides
examples/              # element, pressure, contact, dynamics, and sensitivity cases
```

Useful references:

- [`docs/usage.md`](docs/usage.md) — Python model definition and solving
- [`docs/cae_ui.md`](docs/cae_ui.md) — CAE pre/postprocessor workflow
- [`docs/structure.md`](docs/structure.md) — package architecture and data flow
- [`docs/solver.md`](docs/solver.md) — nonlinear and dynamic solver details
- [`docs/loads.md`](docs/loads.md) — load interface and examples
- [`docs/materials.md`](docs/materials.md) — material models, theory, and parameters
- [`docs/elements/elements_types_CHN.md`](docs/elements/elements_types_CHN.md) — element formulas and integration rules
- [`docs/elements/isoparametric_and_quadrature_CHN.md`](docs/elements/isoparametric_and_quadrature_CHN.md) — mapping and quadrature details

## License and scope

TorchFEA is research software. For production or safety-critical engineering
use, validate the formulation, material parameters, mesh convergence, and
solver settings for the target problem before relying on the results.
