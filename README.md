# 3D Memory In Memory Networks (3D-MIM)

3D-MIM learns time-series of 3D voxels with the sape (timestep, color, width, height, depth). The model is developed to predict the evolution of Supernova (SN) shells on the paper [3D-Spatiotemporal Forecasting the Expansion of Supernova Shells Using Deep Learning toward High-Resolution Galaxy Simulations
](https://arxiv.org/abs/2302.00026). The original Memory In Memory network was proposed by [Yunbo Wang et al. 2018](https://arxiv.org/abs/1811.07490).


## Abstract

Supernova (SN) plays an important role in galaxy formation and evolution. In high-resolution galaxy simulations using massively parallel computing, short integration timesteps for SNe are serious bottlenecks. This is an urgent issue that needs to be resolved for future higher-resolution galaxy simulations. One possible solution would be to use the Hamiltonian splitting method, in which regions requiring short timesteps are integrated separately from the entire system. To apply this method to the particles affected by SNe in a smoothed-particle hydrodynamics simulation, we need to detect the shape of the shell on and within which such SN-affected particles 
reside during the subsequent global step in advance. In this paper, we develop a deep learning model, 3D-MIM, to predict a shell expansion after a SN explosion. Trained on turbulent cloud simulations with particle mass $m_{\rm gas}=1 \mathrm{M}_\odot$, the model accurately reproduces the anisotropic shell shape, where densities decrease by over 10 per cent by the explosion. We also demonstrate that the model properly predicts the shell radius in the uniform medium beyond the training dataset of inhomogeneous turbulent clouds. We conclude that our model enables the forecast of the shell and its interior where SN-affected particles will be present.


# How to run

## Build containers (if necessary)

Two options are prepared to make containers.
If you want to use Docker (e.g., on clusters or supercomputers),

```
docker pull kyafuk/tensorflow:mim-3d
```

If you want to use singularity (e.g., on clusters or supercomputers),

```
singularity build mim-3d.file docker://kyafuk/tensorflow:mim-3d
```

## Parameters you may need to change
The following is `run.sh`.
SampleData.npz has the shape (240,1,32,32,32).

```:run.sh
#!/bin/bash
python -u run.py \
    --is_training=True \
    --dataset_name sn \
    --train_data_paths ./data/sn/SampleData.npz \
    --valid_data_paths ./data/sn/SampleData.npz \
    --save_dir checkpoints/Sample \
    --gen_frm_dir results/Sample \
    --model_name mim \
    --allow_gpu_growth=True \
    --img_channel 1 \
    --img_width 32 \
    --input_length 1 \
    --total_length 20 \
    --max_iterations 1\
    --display_interval 1 \
    --test_interval 12 \
    --snapshot_interval 1 \
    --num_hidden 32,32,32,32 \
    --batch_size 1 \
    --patch_size 1 \
    --num_save_samples 12 \
    |& tee sample.log
```


## Forecast sample
A forecast result by the 3D-MIM.
The following shows the cross section view of 3D voxels.
_Left_: the simulation result (ground truth). _Right_: the 3D-MIM's forecast result.

![SN](https://github.com/kyaFUK/3D-MIM/blob/master/test_sample/SN_video.gif)


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
