# 3D Memory In Memory Networks (3D-MIM)

3D-MIM learns time-series of 3D voxels with the sape (timestep, color, width, height, depth). The model is developed to predict the evolution of Supernova (SN) shells on the paper [3D-Spatiotemporal Forecasting the Expansion of Supernova Shells Using Deep Learning toward High-Resolution Galaxy Simulations
](https://arxiv.org/abs/2302.00026), which is accepted for [MNRAS](https://academic.oup.com/mnras). The original Memory In Memory network was proposed by [Yunbo Wang et al. 2018](https://arxiv.org/abs/1811.07490).

## Forecast sample
A forecast result by the 3D-MIM.
The following shows the cross section view of 3D voxels.
_Left_: the simulation result (ground truth). _Right_: the 3D-MIM's forecast result.

![SN](https://github.com/kyaFUK/3D-MIM/blob/master/test_sample/SN_video.gif)



# How to run

## Build containers (if necessary)

Two options are prepared to make containers.
If you want to use Docker,

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
