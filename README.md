[![DOI](https://zenodo.org/badge/399821666.svg)](https://zenodo.org/badge/latestdoi/399821666)

![monster truck jumping over a crevasse](crevasserator.png)
logo made by [Seth Olinger](http://setholinger.github.io)

# crevasserator
finite element codes to calculate glacier crevasse stress intensity factors

Most of the important stuff is in crevasse.py

The notebooks contain:

- An example single-parameter run crevasse-model-single-parameter-set.ipynb
- An example parameter space study, crevasse-model-parameter-space.ipynb
- An effort to debug the non-smooth results, test-swell-phase-picker.ipynb

The crevasse.py module makes heavy use of [FeNiCS](fenicsproject.org). I prefer running FeNiCS out of a docker image with Jupyer Hub enabled, as described [here](https://fenics.readthedocs.io/projects/containers/en/latest/jupyter.html).
