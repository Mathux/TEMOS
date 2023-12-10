<div align="center">

# TEMOS: TExt to MOtionS
## Generating diverse human motions from textual descriptions

</div>

## Description
Official PyTorch implementation of the paper [**"TEMOS: Generating diverse human motions from textual descriptions"**](http://arxiv.org/abs/2204.14109), [ECCV 2022 (Oral)](https://eccv2022.ecva.net).

Please visit our [**webpage**](https://mathis.petrovich.fr/temos/) for more details.

![teaser_light](visuals/teaser_white.png#gh-light-mode-only)![teaser_dark](visuals/teaser_black.png#gh-dark-mode-only)


### Bibtex
If you find this code useful in your research, please cite:

```
@inproceedings{petrovich22temos,
  title     = {{TEMOS}: Generating diverse human motions from textual descriptions},
  author    = {Petrovich, Mathis and Black, Michael J. and Varol, G{\"u}l},
  booktitle = {European Conference on Computer Vision ({ECCV})},
  year      = {2022}
}
```

You can also put a star :star:, if the code is useful to you.


## Installation :construction_worker:

<details><summary>Click to expand</summary>

### 1. Create conda environment

<details><summary>Instructions</summary>

```
conda create python=3.9 --name temos
conda activate temos
```

Install [PyTorch 1.10](https://pytorch.org/) inside the conda environment, and install the following packages:
```bash
pip install pytorch_lightning --upgrade
pip install torchmetrics==0.7
pip install hydra-core --upgrade
pip install hydra_colorlog --upgrade
pip install shortuuid
pip install rich
pip install pandas
pip install transformers
pip install psutil
pip install einops
```
The code was tested on Python 3.9.7 and PyTorch 1.10.0.

</details>

### 2. Download the datasets

<details><summary>Instructions</summary>

#### KIT Motion-Language dataset
**Be sure to read and follow their license agreements, and cite accordingly.**

Use the code from [Ghosh et al.](https://github.com/anindita127/Complextext2animation) to download and prepare the kit dataset (extraction of xyz joints coodinates data from axis-angle Master Motor Map). Move or copy all the files which ends with "_meta.json", "_annotations.json" and "_fke.csv" inside the ``datasets/kit`` folder.
"
These motions are process by the Master Motor Map (MMM) framework. To be able to generate motions with SMPL body model, please look at the next section.

#### (Optional) Motion processed with MoSh++ (in AMASS)
**Be sure to read and follow their license agreements, and cite accordingly.**

Create this folder:
```bash
mkdir datasets/AMASS/
```

Go to the [AMASS website](https://amass.is.tuebingen.mpg.de/download.php), register and go to the Download tab. Then download the "SMPL+H G" files corresponding to the datasets [KIT, CMU, EKUT] into the ``datasets/AMASS`` directory and uncompress the archives:

```bash
cd datasets/AMASS/
tar xfv CMU.tar.bz2
tar xfv KIT.tar.bz2
tar xfv EKUT.tar.bz2
cd ../../
```

</details>

### 3. Download text model dependencies

<details><summary>Instructions</summary>

#### Download distilbert from __Hugging Face__
```bash
cd deps/
git lfs install
git clone https://huggingface.co/distilbert-base-uncased
cd ..
```

</details>

### 4. (Optional) SMPL body model

<details><summary>Instructions</summary>

This is only useful if you want to use generate 3D human meshes like in the teaser. In this case, you also need a subset of the AMASS dataset (see instructions below).

Go to the [MANO website](https://mano.is.tue.mpg.de/download.php), register and go to the Download tab.

- Click on "Models & Code" to download ``mano_v1_2.zip`` and place it in the folder ``deps/smplh/``.
- Click on "Extended SMPL+H model" to download ``smplh.tar.xz`` and place it in the folder ``deps/smplh/``.

The next step is to extract the archives, merge the hands from ``mano_v1_2`` into the ``Extended SMPL+H models``, and remove any chumpy dependency.
All of this can be done using with the following commands. (I forked both scripts from this repo [SMPLX repo](https://github.com/vchoutas/smplx/tree/master/tools), updated them to Python 3, merged them, and made it compatible with ``.npz`` files).


```bash
pip install scipy chumpy
bash prepare/smplh.sh
```

This will create ``SMPLH_FEMALE.npz``, ``SMPLH_MALE.npz``, ``SMPLH_NEUTRAL.npz`` inside the ``deps/smplh`` folder.

</details>

### 5. (Optional) Download pre-trained models

<details><summary>Instructions</summary>

Make sure to have gdown installed

```bash
pip install --user gdown
```

Then, please run this command line:

```bash
bash prepare/download_pretrained_models.sh
```

Inside the ``pretrained models`` folder, you will find one for each type of data (see Section [datasets](#datasets) below for more information).
```
pretrained_models
├── kit-amass-rot
│   └── 1cp6dwpa
├── kit-amass-xyz
│   └── 5xp9647f
└── kit-mmm-xyz
    └── 3l49g7hv
```

</details>

</details>

## How to train TEMOS :rocket:

<details><summary>Click to expand</summary>

The command to launch a training experiment is the folowing:
```bash
python train.py [OPTIONS]
```

The parsing is done by using the powerful [Hydra](https://github.com/facebookresearch/hydra) library. You can override anything in the configuration by passing arguments like ``foo=value`` or ``foo.bar=value``.


### Experiment path
Each training will create a unique output directory (referred to as ``FOLDER`` below), where logs, configuations and checkpoints are stored.

By default it is defined as ``outputs/${data.dataname}/${experiment}/${run_id}`` with ``data.dataname`` the name of the dataset (see examples below), ``experiment=baseline`` and ```run_id``` a 8 unique random alpha-numeric identifier for the run (everything can be overridden if needed).

This folder is printed during logging, it should look like ``outputs/kit-mmm-xyz/baseline/3gn7h7v6/``.


### Some optional parameters
#### Datasets
- ``data=kit-mmm-xyz``: KIT-ML motions processed by the [MMM](https://mmm.humanoids.kit.edu/) framework (as in the [original data](https://motion-annotation.humanoids.kit.edu/dataset/)) loaded as xyz joint coordinates (after axis-angle transformation → xyz) (by default)
- ``data=kit-amass-rot``: KIT-ML motions loaded as [SMPL](https://smpl.is.tue.mpg.de/) rotations and translations, from [AMASS](https://amass.is.tue.mpg.de/) (processed with [MoSh++](https://github.com/nghorbani/moshpp))
- ``data=kit-amass-xyz``: KIT-ML motions loaded as xyz joint coordinates, from [AMASS](https://amass.is.tue.mpg.de/) (processed with [MoSh++](https://github.com/nghorbani/moshpp)) after passing through a [SMPL](https://smpl.is.tue.mpg.de/) layer and regressing the correct joints.


#### Training
- ``trainer=gpu``: training with CUDA, on an automatically selected GPU (default)
- ``trainer=cpu``: training on the CPU (not recommended)


</details>

## How to generate motions with TEMOS :walking:

<details><summary>Click to expand</summary>

### Dataset splits
To get results comparable to previous work, we use the same splits as in [Language2Pose](https://github.com/chahuja/language2pose) and [Ghosh et al.](https://github.com/anindita127/Complextext2animation).
To be explicit, and not rely on random seeds, you can find the list of id-files in [datasets/kit-splits/](datasets/kit-splits/) ([train](datasets/kit-splits/train)/[val](datasets/kit-splits/val)/[test](datasets/kit-splits/test)).

When sampling [Ghosh et al.](https://github.com/anindita127/Complextext2animation)'s motions with their code, I noticed that their dataloader is missing some sequences (see the discussion [here](https://github.com/anindita127/Complextext2animation/issues/3#issuecomment-1059566036)).
In order to compare all the methods with the same test set, we use the 520 sequences produced by Ghosh et al. code for the test set (instead of the 587 sequences). This split is refered as [gtest](datasets/kit-splits/gtest) (for "Ghosh test"). It is used per default in the sampling/evaluation/rendering code. You can change this set by specifying ``split=SPLIT`` in each command line.

You can also find in [datasets/kit-splits/](datasets/kit-splits/), the split used for the human-study ([human-study](datasets/kit-splits/human-study)) and the split used for the visuals of the paper ([visu](datasets/kit-splits/visu)).


### Sampling/generating motions
The command line to sample one motion per sequence is the following:
```bash
python sample.py folder=FOLDER [OPTIONS]
```

This command will create the folder ``FOLDER/samples/SPLIT`` and save the motions in the npy format.

### Some optional parameters
- ``mean=false``: Take the mean value for the latent vector, instead of sampling (default is false)
- ``number_of_samples=X``: Generate ``X`` motions (by default it generates only one)
- ``fact=X``: Multiplies sigma by ``X`` during sampling (1.0 by default, diversity can be increased when ``fact>1``)


### Model trained on SMPL rotations
If your model has been trained with ``data=kit-amass-rot``, it produces [SMPL](https://smpl.is.tue.mpg.de/) rotations and translations. In this case, you can specify the type of data you want to save after passing through the [SMPL](https://smpl.is.tue.mpg.de/) layer.
- ``jointstype=mmm``: Generate xyz joints compatible with the [MMM](https://mmm.humanoids.kit.edu/) bodies (by default). This gives skeletons comparable to ``data=kit-mmm-xyz`` (needed for evaluation).
- ``jointstype=vertices``: Generate human body meshes (needed for rendering).

</details>

## Evaluating TEMOS (and prior works) :bar_chart:

<details><summary>Click to expand</summary>

To evaluate TEMOS on the metrics defined in the paper, you must generate motions first (see above), and then run:
```bash
python evaluate.py folder=FOLDER [OPTIONS]
```
This will compute and store the metrics in the file ``FOLDER/samples/metrics_SPLIT`` in a yaml format.

### Some optional parameters
Same parameters as in ``sample.py``, it will choose the right directories for you. In the case of evaluating with ``number_of_samples>1``, the script will compute two metrics ``metrics_gtest_multi_avg`` (the average of single metrics) and ``metrics_gtest_multi_best`` (chosing the best output for each motion). Please check the paper for more details.

### Model trained on SMPL rotations
Currently, evaluation is only implemented on skeletons with [MMM](https://mmm.humanoids.kit.edu/) format. You must therefore use ``jointstype=mmm`` during sampling.


### Evaluating prior works

Please use this command line to download the motions generated from previous work:

```bash
bash prepare/download_previous_works.sh
```

Then, to evaluate a method, you can do for example:

```bash
python evaluate.py folder=previous_work/ghosh
```

or change "ghosh" with "jl2p" or "lin".


To give an overview on how to extract their motions:
1. Generate motions with their code (it is still in the rifke feature space)
2. Save them in xyz format (I "hack" their render script, to save them in xyz npy format instead of rendering)
3. Load them into the evaluation code, as shown above.

</details>

## Rendering motions :high_brightness:

<details><summary>Click to expand</summary>

To get the visuals of the paper, I use [Blender 2.93](https://www.blender.org/download/releases/2-93/). The setup is not trivial (installation + running), I do my best to explain the process but don't hesitate to tell me if you have a problem.


### Instalation
The goal is to be able to install blender so that it can be used with python scripts (so we can use ``import bpy''). There seem to be many different ways to do this, I will explain the one I use and understand (feel free to use other methods or suggest an easier way). The installation of Blender will be done as a standalone package. To use my scripts, we will run blender in the background, and the python executable in blender will run the script.

In any case, after the installation, please do step 5/6. to install the dependencies in the python environment.

1. Please follow the [instructions](https://www.blender.org/download/lts/2-93/) to install blender 2.93 on your operating system. Please install exactly this version.
2. Locate the blender executable if it is not in your path. For the following commands, please replace ``blender`` with the path to your executable (or create a symbolic link or use an alias).
   - On Linux, it could be in ``/usr/bin/blender`` or ``/snap/bin/blender`` (already in your path).
   - On macOS, it could be in ``/Applications/Blender.app/Contents/MacOS/Blender`` (not in your path)
3. Check that the correct version is installed:
   - ``blender --background --version`` should return "Blender 2.93.X".
   - ``blender --background --python-expr "import sys; print('\nThe version of python is '+sys.version.split(' ')[0])"`` should return "3.9.X".
4. Locate the python installation used by blender the following line. I will refer to this path as ``/path/to/blender/python``.
```bash
blender --background --python-expr "import sys; import os; print('\nThe path to the installation of python of blender can be:'); print('\n'.join(['- '+x.replace('/lib/python', '/bin/python') for x in sys.path if 'python' in (file:=os.path.split(x)[-1]) and not file.endswith('.zip')]))"
```

5. Install pip
```bash
/path/to/blender/python -m ensurepip --upgrade
```

6. Install these packages in the python environnement of blender:
```bash
/path/to/blender/python -m pip install --user numpy
/path/to/blender/python -m pip install --user matplotlib
/path/to/blender/python -m pip install --user hydra-core --upgrade
/path/to/blender/python -m pip install --user hydra_colorlog --upgrade
/path/to/blender/python -m pip install --user moviepy
/path/to/blender/python -m pip install --user shortuuid
```

### Launch a python script (with arguments) with blender
Now that blender is installed, if we want to run the script ``script.py`` with the blender API (the ``bpy`` module), we can use:
```bash
blender --background --python script.py
```

If you need to add additional arguments, this will probably fail (as blender will interpret the arguments). Please use the double dash ``--`` to tell blender to ignore the rest of the command.
I then only parse the last part of the command (check [temos/launch/blender.py](temos/launch/blender.py) if you are interested).


### Rendering one sample
To render only one motion, please use this command line:
```bash
blender --background --python render.py -- npy=PATH_TO_DATA.npy [OPTIONS]
```

### Rendering all the npy of a folder
Please use this command line to render all the npy inside a specific folder.
```bash
blender --background --python render.py -- folder=FOLDER_WITH_NPYS [OPTIONS]
```

### SMPL bodies
Don't forget to generate the data with the option ``jointstype=vertices`` before.
The renderer will automatically detect whether the motion is a sequence of joints or meshes.


### Some optional parameters
- ``downsample=true``: Render only 1 frame every 8 frames, to speed up rendering (by default)
- ``canonicalize=true``: Make sure the first pose is oriented canonically (by translating and rotating the entire sequence) (by default)
- ``mode=XXX``: Choose the rendering mode (default is ``mode=sequence``)
  - ``video``: Render all the frames and generate a video (as in the supplementary video)
  - ``sequence``: Render a single frame, with ``num=8`` bodies (sampled equally, as in the figures of the paper)
  - ``frame``: Render a single frame, at a specific point in time (``exact_frame=0.5``, generates the frame at about 50% of the video)
- ``quality=false``: Render to a higher resolution and denoise the output (default to false to speed up))

</details>

## License :books:
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, Hugging Face, Hydra, and uses datasets which each have their own respective licenses that must also be followed.
