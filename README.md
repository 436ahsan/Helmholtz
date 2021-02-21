# Helmholtz
AMG Helmholtz project with Achi, Oren, James, Jacob and Ahsan.

## Contents
- `src`: source code.
- `src/helmholtz`: Python Helmholtz code.
- `src/test`: unit tests.
- `notebooks`: Juypter notebooks.

## Installation
- Install conda.
- Create a conda environment from the attached environment.yml: `conda env create -f environment. yml.`
- Add `src` to your PYTHONPATH.

## Testing
The project contains Pytest unit tests for the main modules. To run all tests, run `cd src; pytest test`.

## TODO List
* Replace many x columns by few x's and many windows to fit the restriction.
* Add Ritz projection. For now save only the smallest lam and enough x's that match x's original shape.
* Calculate multiple eigenpairs.

### Done
* Replace GlobalParams by passing the lambda array with x from level to level within the cycle.
