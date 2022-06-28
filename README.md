[![DOI](https://zenodo.org/badge/399821666.svg)](https://zenodo.org/badge/latestdoi/399821666)

![monster truck jumping over a crevasse](crevasserator.png)
logo made by [Seth Olinger](http://setholinger.github.io)

# crevasserator
finite element method (FEM) codes to calculate glacier crevasse stress intensity factors

Most of the important stuff is in crevasse.py

The notebooks contain:

- An example single-parameter run crevasse-model-single-parameter-set.ipynb
- An example parameter space study, crevasse-model-parameter-space.ipynb
- An effort to debug the non-smooth results, test-swell-phase-picker.ipynb

# Install and Run

1. Clone this repository.
2. Run a [Docker](https://www.docker.com/get-started/) container with [FEniCS](https://fenicsproject.org/) (the FEM backend). Details follow.

The easiest option to get started is to use Jupyter. The following docker command will launch a Jupyter Notebook server where you can run the notebooks.  Make sure
```
docker run --name crevasserator -w /home/fenics -v (pwd):/home/fenics/shared -d -p 127.0.0.1:8888:8888 quay.io/fenicsproject/stable 'jupyter-notebook --ip=0.0.0.0'
```
