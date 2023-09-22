# 3D Memory In Memory Networks

3D-MIM is a neural network for predicting the evolution of SN simulations. It is based on the paper [3D-Spatiotemporal Forecasting the Expansion of Supernova Shells Using Deep Learning toward High-Resolution Galaxy Simulations
](https://arxiv.org/abs/2302.00026).

```
docker pull kyafuk/tensorflow:mim-3d
```

```
singularity build mim-3d.file docker://kyafuk/tensorflow:mim-3d
```

## Abstract

Supernova (SN) plays an important role in galaxy formation and evolution. In high-resolution galaxy simulations using massively parallel computing, short integration timesteps for SNe are serious bottlenecks. This is an urgent issue that needs to be resolved for future higher-resolution galaxy simulations. One possible solution would be to use the Hamiltonian splitting method, in which regions requiring short timesteps are integrated separately from the entire system. To apply this method to the particles affected by SNe in a smoothed-particle hydrodynamics simulation, we need to detect the shape of the shell on and within which such SN-affected particles 
reside during the subsequent global step in advance. In this paper, we develop a deep learning model, 3D-MIM, to predict a shell expansion after a SN explosion. Trained on turbulent cloud simulations with particle mass $m_{\rm gas}=1 \mathrm{M}_\odot$, the model accurately reproduces the anisotropic shell shape, where densities decrease by over 10 per cent by the explosion. We also demonstrate that the model properly predicts the shell radius in the uniform medium beyond the training dataset of inhomogeneous turbulent clouds. We conclude that our model enables the forecast of the shell and its interior where SN-affected particles will be present.

![model](https://github.com/ZJianjin/mim_images/blob/master/readme_structure.png)


## How to run

```
singularity build mim-3d.file docker://kyafuk/tensorflow:mim-3d
```

## Generation Results

#### SN simulations

![mnist1](https://github.com/ZJianjin/mim_images/blob/master/mnist1.gif)

![mnist2](https://github.com/ZJianjin/mim_images/blob/master/mnist4.gif)

![mnist2](https://github.com/ZJianjin/mim_images/blob/master/mnist5.gif)


## BibTeX
```
@ARTICLE{2023arXiv230200026H,
       author = {{Hirashima}, Keiya and {Moriwaki}, Kana and {Fujii}, Michiko S. and {Hirai}, Yutaka and {Saitoh}, Takayuki R. and {Makino}, Junichiro},
        title = "{3D-Spatiotemporal Forecasting the Expansion of Supernova Shells Using Deep Learning toward High-Resolution Galaxy Simulations}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies, Computer Science - Computational Engineering, Finance, and Science, Computer Science - Machine Learning},
         year = 2023,
        month = jan,
          eid = {arXiv:2302.00026},
        pages = {arXiv:2302.00026},
          doi = {10.48550/arXiv.2302.00026},
archivePrefix = {arXiv},
       eprint = {2302.00026},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230200026H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
