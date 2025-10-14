<div align="center">
  
# C-PLANK
  
</div>

## Description
C-PLANK (Chemi-Proteome Language Attention NetworK) is a deep learning framework that learn from the chemoproteomics datasets of fragment-based ligand-protein interactions in cells, providing interpretable interaction fingerprints of local residues and atoms.
## Architecture

## Setup C-PLANK
C-PLANK is implemented and tested with Python 3.12, pytorch 2.5 and cuda 12.4 on Linux. However, C-PLANK is also supported for additional standard operating systems such as Windows and macOS to run with following depending packages:
```
pandas
numpy
torch
scikit-learn
rdkit
fair-esm
yacs
itertools
tqdm
multiprocessing
```
## Installation Guid
```
# clone the source code of C-PLANK
$ git clone https://github.com/Bin-cc/C-PLANK
$ cd C-PLANK

# create a new conda or miniconda environment
$ conda create --name plank python=3.12
$ conda activate plank

# install requried python dependencies
$ pip install -r requirements.txt
```
## Use C-PLANK
C-PLANK is a virtual screening tool used for 




