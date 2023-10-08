# BiGCN: Leveraging Cell and Gene Similarities for Single-cell Transcriptome Imputation with Bi-Graph Convolutional Networks

This is an official implementation of the paper, "BiGCN: Leveraging Cell and Gene Similarities for Single-cell Transcriptome Imputation with Bi-Graph Convolutional Networks"

scBiGCN is a method that utilizes two GCNs to reconstruct gene expression matrices based on the similarity matrices of cells and genes. Utilizing each similarity enables the recovery of information lost through Dropout.

<img width="1027" alt="Screenshot 2023-09-14 at 15 49 53" src="https://github.com/inoue0426/scBiGCN/assets/8393063/c9d1fbc0-bdf0-49b3-91b3-50181cbe16ec">

scBiGCN has been implemented in Python.

To get started immediately, check out our tutorials:
- [Tutorial](https://github.com/inoue0426/scBiGCN/blob/main/sample%20notebook.ipynb)

## Installation from GitHub
To clone the repository and install manually, run the following from a terminal:
```
git clone git@github.com:inoue0426/scBiGCN.git
cd scBiGCN
conda create --name scBiGCN python=3.10 -y
conda activate scBiGCN
pip install -r requirement.txt
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

## Requirement

```
numpy==1.23.5
pandas==2.0.3
scikit-learn==1.3.0
torch==1.13.1+cu116
torch-geometric==2.3.1
torch-sparse==0.6.17+pt113cu116
tqdm==4.65.0
```

** Note: To utilize GPU acceleration, it is necessary to configure your own CUDA environment for PyTorch and PyTorch-Sparse.

## Environment

Our experiment was conducted on Ubuntu with an RTX 2080, but it also works on an M1 Mac. To use this model, please adjust the requirements to suit your environment, paying special attention to the PyTorch-related libraries. 
**  Please note that as of October 8, 2023, this model doesn't support GPU on M1 Mac due to compatibility issues with PyTorch Geometric. 


## Usage

### Quick Start

The following code runs scBiGCN on test data located in the scBiGCN repository.

```python
import pandas as pd
import bigcn

df = pd.read_csv('sample_data/sample_data.csv.gz', index_col=0)
bigcn.run_model(df, verbose=True)
```

## Help
If you have any questions or require assistance using MAGIC, please feel free to make a issues on https://github.com/inoue0426/scBiGCN/
