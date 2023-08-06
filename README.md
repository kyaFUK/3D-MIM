# 3D Memory In Memory Networks

3D-MIM is a neural network for predicting the evolution of SN simulations. It is based on the paper [3D-Spatiotemporal Forecasting the Expansion of Supernova Shells Using Deep Learning toward High-Resolution Galaxy Simulations
](https://arxiv.org/abs/2302.00026).

## Abstract

Natural spatiotemporal processes can be highly non-stationary in many ways, e.g. the low-level non-stationarity such as spatial correlations or temporal dependencies of local pixel values; and the high-level non-stationarity such as the accumulation, deformation or dissipation of radar echoes in precipitation forecasting.

We try to stationalize and approximate the non-stationary processes by modeling the differential signals with the MIM recurrent blocks. By stacking multiple MIM blocks, we could potentially handle higher-order non-stationarity. Our model achieves the state-of-the-art results on three spatiotemporal prediction tasks across both synthetic and real-world data.

![model](https://github.com/ZJianjin/mim_images/blob/master/readme_structure.png)



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
