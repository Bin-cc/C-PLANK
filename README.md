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
C-PLANK is a virtual screening tool used for deciphering ligand interactome derived from affinity-based proteome profiling (AfBPP). In current version of C-PLANK, only fully functionalized fragment (FFF) ligands with photoreactive group of diazirine are considered during training. For the other types of photoreactive group, due to the insufficient resource and biased labeling preference, C-PLANK did not leverage them in training process but this model framework could be also expanded to the other photoreactive groups.
#### Custom Dataset 
The training dataset utilized by C-PLANK is prepared in ./Dataset/split_dataset.tsv which is generated from Offensperger et al [1]. If you have a real-world case to train and test C-PLANK, the custom dataset is required to include a header row of 'sequence', 'SMILES', and 'label'.
#### Embedding Generation
To reduce the loading time, we recommand to prepare the protein and ligand embeddings in advance with following command
```
$ python getEmbedding.py --outpath ${path} --data ${dataset}
```
The dataset contains the header row of 'ProteinID', 'sequence', 'ligID', and 'SMILES'. If you have not prepared the embedding file, it will be generated during training which would cost much time. 
#### Run C-PLANK
We provide a set of hyperparameters in config.py for running the model and reproducing our result. To train C-PLANK, you can simply run the following command
```
$ python main.py --filepath ${path} --seed 42
```
The file path is the path toward custom dataset and files of protein/ligand embeddings.
#### Interpretation Prediction
With the trained model, you can implement a single enquiry or a large-scale screening of ligand-protein interactions with visualized interpretation (optional). The details can be checked in screening.ipynb.
## Acknowledgements
This implementation is inspired and partially based on the prior works from refer [2] and [3]
## Reference
[1] Offensperger, F. et al. Large-scale chemoproteomics expedites ligand discovery and predicts ligand behavior in cells. Science 384, eadk5864 (2024). https://doi.org/doi:10.1126/science.adk5864
[2] Bai, P., Miljković, F., John, B. & Lu, H. Interpretable bilinear attention network with domain adaptation improves drug–target prediction. Nature Machine Intelligence 5, 126-136 (2023). https://doi.org/10.1038/s42256-022-00605-1
[3] Hadipour, H. et al. GraphBAN: An inductive graph-based approach for enhanced prediction of compound-protein interactions. Nature Communications 16, 2541 (2025). https://doi.org/10.1038/s41467-025-57536-9
## Citation
