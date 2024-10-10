# BAPred: Protein-ligand Binding Affinity Prediction

This repository contains code for predicting the binding affinity between protein-ligand pairs using GNN (Graph Neural Network) models. The model predicts the binding affinity score for each ligand pose relative to the receptor.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/BAPred.git
cd BAPred
```

2. Set up a Python environment and install dependencies:
You can use the provided `env.yaml` to create a conda environment:
```bash
conda env create -f env.yaml
conda activate BAPred
```

## Usage

This repository provides a script to predict the binding affinity of protein-ligand pairs using a pretrained model.

### Running the Inference

To run the inference and predict binding affinity, use the following command:
```bash
python inference.py -r ./example/1KLT.pdb -l ./example/ligands.sdf -o ./result.csv --model_path ./weight --device cuda
```
Where:  
- `-r` specifies the receptor protein PDB file.  
- `-l` specifies the ligand SDF file.  
- `-o` specifies the output CSV file for results.  
- `--model_path` specifies the directory containing the model weight (`BAPred.pth`).  
- `--device` specifies whether to use `cuda` or `cpu`.

### Output

The output will be saved in the specified CSV file and will contain the following columns:  
- **Name**: Name or index of the ligand.  
- **Affinity**: Predicted binding affinity score for the ligand pose.

## File Structure

```
.
├── data
│   ├── atom_feature.py        # Atom feature extraction
│   ├── data.py                # Data loading and preprocessing
│   └── utils.py               # Utility functions
├── example
│   ├── 1KLT.pdb               # Example receptor PDB file
│   ├── ligands.sdf            # Example ligand SDF file
│   └── run.sh                 # Example script to run inference
├── inference.py               # Inference script for binding affinity prediction
├── LICENSE                    # License file
├── model
│   ├── GatedGCNLSPE.py        # GNN model implementation
│   ├── GraphGPS.py            # GraphGPS model architecture
│   ├── MHA.py                 # Multi-Head Attention module
│   ├── model.py               # Model wrapper class
├── README.md                  # This README file
└── weight
    └── BAPred.pth             # Saved weight for the binding affinity prediction model
```

## Example

Below is an example of how to run the code:
```bash
python inference.py     -r ./example/1KLT.pdb     -l ./example/ligands.sdf     -o ./result.csv     --batch_size 128     --model_path ./weight     --device cuda
```

The example receptor `1KLT.pdb` and ligand `ligands.sdf` are provided in the `example/` directory. This command will generate a CSV file named `result.csv` containing the predicted binding affinity values for each ligand pose.

## Models

The prediction model is based on Graph Neural Networks (GNNs) with additional layers such as GraphGPS and Multi-Head Attention. The model takes the protein and ligand graphs as input and outputs the predicted binding affinity for each ligand pose.

- **Binding Affinity Model (`BAPred.pth`)**: Predicts the binding affinity of the ligand pose.

The model architectures are defined in `model/GatedGCNLSPE.py`, `model/GraphGPS.py`, and `model/MHA.py`.

