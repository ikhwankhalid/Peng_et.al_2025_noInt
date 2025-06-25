Code used in a manuscript by Jing-Jie Peng†, Beate Throm†, Maryam Najafian Jazi, Ting-Yun Yen, Rocco Pizzarelli, Hannah Monyer*‡, Kevin Allen*‡

## Setting up

Clone this repository

```
cd ~
mkdir repo
cd repo
git clone https://github.com/PJJ19/Peng_et.al_2025_noInt.git
```

## Download the data in Dryad

https://doi.org/10.5061/dryad.f7m0cfz80

Concatenate and merge the different sections

### Combine files

```
cat folder.tar.gz.part.* > folder.tar.gz
```

### Extract archive

```
tar -xzf folder.tar.gz -C /path/to/destination_directory
```

## Setup python Environment

### Python environment

Here's a step by step guide to implement the method:

Make sure you have Anaconda or Miniconda installed

First create your conda environment:

```
conda create -n my_custom_environment python=3.11.5
```

```
conda activate my_custom_environment
```

Install basic packages

```
conda install -c conda-forge pip ipython jupyter nb_conda ffmpeg
```

```
pip install -r requirements.txt
```

```
pip install statannotations==0.6.0
```

```
pip install seaborn==0.12.2
```

Note, you might run into a version dependency issue with statannotations and seaborn, ignore it for now, the functions we used did not have this conflict issue.


```
conda install Cython
```

Install [spikeA](https://github.com/kevin-allen/spikeA/blob/main/docs/main.md)

We recommend creating an empty folder to store the custom packages.

```
cd ~
mkdir repo
```

```
cd ~/repo
git clone https://github.com/kevin-allen/spikeA.git
cd ~/repo/spikeA
pip install -e ~/repo/spikeA
cd ~/repo/spikeA/spikeA/
python setup.py build_ext --inplace
```


Install [autopipy](https://github.com/kevin-allen/autopipy/blob/master/docs/easy_installation.md)

```
cd ~/repo
git clone https://github.com/kevin-allen/autopipy.git
pip install -e ~/repo/autopipy
```

## Data analysis

The starting point of the analysis are in /Data_analysis

change the `PROJECT_DATA_PATH` in `setup_project.py` to the directory you extracted the file from Dryad. The code for each figure is in their corresponding folder.


Note: Some notebooks have `if:` in empty code blocks, I use this as code breakers for chunks of code to run when I press "Run all". You can skip to the code after the `if:` block.

## Grid cell decoding method

For a step-by-step guide to our decoding method, check out: [autopi_grid_cell_decoding_repo](https://github.com/PJJ19/autopi_grid_cell_decoding)