<div align="center">

# TEMOS: TExt to MOtionS
## Generating diverse human motions from textual descriptions

</div>

## Description
Official PyTorch implementation of the paper [**"TEMOS: Generating diverse human motions from textual descriptions".**](http://arxiv.org/abs/2204.14109).
Please visit our [**webpage**](https://mathis.petrovich.fr/temos/) for more details.

![teaser_light](visuals/teaser_white.png#gh-light-mode-only)![teaser_dark](visuals/teaser_black.png#gh-dark-mode-only)


## Installation :construction_worker:
### 1. Create conda environment

```
conda create python=3.9 --name temos
conda activate temos
```

Install [PyTorch 1.10](https://pytorch.org/) inside the conda environnement, and install the following packages:
```bash
pip install pytorch_lightning --upgrade
pip install hydra-core --upgrade
pip install shortuuid
pip install hydra_colorlog --upgrade
pip install tqdm
pip install pandas
pip install transformers
pip install psutil
```
The code was tested on Python 3.9.7 and PyTorch 1.10.0.


### 2. Download the datasets
#### KIT Motion-Language dataset
**Be sure to read and follow their license agreements, and cite accordingly.**

Use the code from [Ghosh et al.](https://github.com/anindita127/Complextext2animation) or [JL2P](https://github.com/chahuja/language2pose) to download and prepare the kit dataset (extraction of xyz joints coodinates data from axis-angle Master Motor Map). Move or copy all the files which ends with "_meta.json", "_annotations.json" and "_fke.csv" inside the ``datasets/kit`` folder.
"

#### AMASS dataset
WIP


### 3. Download text model dependencies
#### Download distilbert from __Hugging Face__
```bash
cd deps/
git lfs install
git clone https://huggingface.co/distilbert-base-uncased
cd ..
```

## How to use TEMOS :rocket:
Each training will create a unique output directory, which we will refer by "FOLDER".

### Training on KIT xyz data processed by the MMM framework
```bash
python train.py data=kit-mmm-xyz
```

### Training on KIT xyz data processed by SMPL
```bash
python train.py data=kit-amass-xyz
```

### Training on KIT with SMPL rotations
```bash
python train.py data=kit-amass-rot
```

### Sampling with TEMOS
```bash
python sample.py folder=FOLDER
```

### Evaluating TEMOS
```bash
python eval.py folder=FOLDER
```
