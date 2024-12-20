[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/joshloyal/multidynet/blob/master/LICENSE)

## A Latent Space Approach to Inferring Distance-Dependent Reciprocity 

*Package Authors: [Joshua D. Loyal](https://joshloyal.github.io/) and Xiangyu Wu*

This package provides an interface for the model described in
"A Latent Space Approach to Inferring Distance-Dependent Reciprocity in Directed Networks." Inference is performed using
Hamiltonian Monte Carlo. For more details, see [Loyal et. al. (2024)](https://arxiv.org/abs/2411.18433).

Dependencies
------------
``rlsm`` requires:

- Python (>= 3.10)

and the requirements highlighted in [requirements.txt](requirements.txt). To install the requirements, run

```python
pip install -r requirements.txt
```

Installation
------------

You need a working installation of numpy, scipy, and jax to install ``rlsm``. Install these required dependencies before proceeding.  Use the following commands to get the copy from GitHub and install all the dependencies:

```
>>> git clone https://github.com/joshloyal/rlsm.git
>>> cd rlsm
>>> pip install -r requirements.txt
>>> python setup.py develop
```

Example
-------

```python

```

<img src="images/ls_n50.png" width="100%" />


Simulation Studies and Real-Data Applications
---------------------------------------------

The [scripts](scripts) directory includes the simulation studies and real-data application found in the article.
