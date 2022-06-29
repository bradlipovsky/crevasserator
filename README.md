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

## Install and Run

1. Clone this repository.
2. Run a [Docker](https://www.docker.com/get-started/) container with [FEniCS](https://fenicsproject.org/) (the FEM backend). Details follow.

### Jupyter Notebooks
The easiest option to get started is to use Jupyter. The following docker command will launch a Jupyter Notebook server where you can run the notebooks.  *For the following docker run commands, run these from the directory that contains the cloned repo.*
```
docker run --name crevasserator -w /home/fenics -v (pwd):/home/fenics/shared -d -p 127.0.0.1:8888:8888 quay.io/fenicsproject/stable 'jupyter-notebook --ip=0.0.0.0'
```

### Interactive Terminal
We have run parameter studies with up to 1e4 individual runs. Scaling in this way can't be achieved using Jupyter since the overhead is too high and the sessions don't persist.  Instead, for this application, we want to create a container with an interactive terminal:
```
docker run --name crevasserator_it -it -w /home/fenics -v (pwd):/home/fenics/shared quay.io/fenicsproject/stable
```

After creating the container, one useful workflow is to start screen (to ensure persistence) and then start and attach the container,
```
screen
docker start crevasserator_it
docker attach crevasserator_it
```
