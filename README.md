<div align="center">

# 🧬 BAPred

**Protein-Ligand Binding Affinity Prediction using Graph Neural Networks**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-orange.svg)](https://pytorch.org/)
[![DGL](https://img.shields.io/badge/DGL-2.4.0-green.svg)](https://www.dgl.ai/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CASP16](https://img.shields.io/badge/CASP16-2nd%20Place-gold.svg)](https://predictioncenter.org/casp16/)
[![GitHub stars](https://img.shields.io/github/stars/eightmm/BAPred.svg?style=social&label=Star)](https://github.com/eightmm/BAPred)

*High-performance protein-ligand binding affinity prediction model - 2nd place in CASP16 ligand affinity challenge*

</div>

## 🌟 Features

- 🏆 **CASP16**: 2nd place in the prestigious CASP16 ligand affinity prediction challenge
- 🎯 **High Accuracy**: Graph Neural Network-based architecture for precise binding affinity prediction
- 🔬 **Research Ready**: Pre-trained models ready for immediate use
- 🛠️ **Easy Integration**: Simple Python API and command-line interface
- 📈 **Scalable**: Batch processing for high-throughput screening

## 🚀 Quick Start

### Installation

Choose your preferred installation method:

<details>
<summary><b>🐍 Option 1: Using Conda (Recommended)</b></summary>

```bash
git clone https://github.com/eightmm/BAPred.git
cd BAPred
conda env create -f env.yaml
conda activate BAPred
```

</details>

<details>
<summary><b>📦 Option 2: Using pip</b></summary>

```bash
git clone https://github.com/eightmm/BAPred.git
cd BAPred
pip install -r requirements.txt
```

</details>

### 🏃‍♂️ Run Your First Prediction

```bash
python run_inference.py -r example/1KLT.pdb -l example/ligands.sdf -o results.csv
```

That's it! 🎉 Your binding affinity predictions will be saved in `results.csv`.

## 📋 Usage Examples

### Basic Usage
```bash
# Predict binding affinities
python run_inference.py -r example/1KLT.pdb -l example/ligands.sdf -o results.csv
```

### Advanced Options
```bash
# Use CPU instead of GPU
python run_inference.py -r protein.pdb -l ligands.sdf -o results.csv --device cpu

# Custom batch size for memory optimization
python run_inference.py -r protein.pdb -l ligands.sdf -o results.csv --batch_size 64

# Specify custom model path
python run_inference.py -r protein.pdb -l ligands.sdf -o results.csv --model_path /path/to/model
```

### Python API
```python
from bapred.inference import inference

# Run prediction programmatically
inference(
    protein_pdb="example/1KLT.pdb",
    ligand_file="example/ligands.sdf",
    output="results.csv",
    batch_size=128,
    model_path="bapred/weight",
    device="cuda"
)
```

## 📁 Project Structure

```
BAPred/
├── 📦 bapred/                 # Main package
│   ├── 🧪 data/               # Data processing modules
│   │   ├── atom_feature.py    # Atomic feature extraction
│   │   ├── data.py           # Dataset handling
│   │   └── utils.py          # Utility functions
│   ├── 🧠 model/              # Neural network models
│   │   ├── GatedGCNLSPE.py   # Gated Graph Convolution
│   │   ├── GraphGPS.py       # Graph GPS architecture
│   │   ├── MHA.py            # Multi-Head Attention
│   │   └── model.py          # Main model wrapper
│   ├── ⚖️ weight/             # Pre-trained weights
│   │   └── BAPred.pth        # Model checkpoint
│   └── 🔮 inference.py       # Inference engine
├── 📝 example/               # Example files
│   ├── 1KLT.pdb             # Sample protein structure
│   └── ligands.sdf          # Sample ligand library
├── 🚀 run_inference.py      # Easy-to-use script
├── 📋 requirements.txt      # Python dependencies
├── 🐍 env.yaml             # Conda environment
└── 📖 README.md            # You are here!
```

## 🎯 Model Architecture

BAPred leverages cutting-edge graph neural network architectures:

- **🔗 Graph Convolution**: Gated GCN with Laplacian Positional Encoding
- **🌐 Graph GPS**: Global attention mechanism for long-range interactions
- **🎭 Multi-Head Attention**: Enhanced feature representation
- **🔄 Complex Interactions**: Protein-ligand interaction modeling

## 📊 Input/Output Formats

### Input
- **Protein**: PDB format (`.pdb`)
- **Ligands**: SDF (`.sdf`), MOL2 (`.mol2`), or text file with paths (`.txt`)

### Output
- **CSV/TSV file** with columns:
  - `Name`: Ligand identifier
  - `pKd`: Predicted binding affinity (pKd scale)
  - `Kcal/mol`: Binding energy in kcal/mol

## 🛠️ System Requirements

- **Python**: 3.11 or higher
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **GPU**: CUDA-compatible GPU (optional, but recommended for speed)
- **Storage**: 2GB free space

## 📊 Performance

| Dataset | Ligands | Processing Time | Performance |
|---------|---------|----------------|-------------|
| **CASP16** | Challenge dataset | Competition | **🥈 2nd Place** |
| Example | 500 | ~3 minutes | High precision |
| Custom | Variable | Scales linearly | Research-grade |

### 🏆 CASP16 Achievement

BAPred achieved **2nd place** in the CASP16 (Critical Assessment of protein Structure Prediction) ligand affinity prediction challenge, demonstrating its state-of-the-art performance in real-world protein-ligand binding affinity prediction tasks.

## 🤝 Contributing

We welcome contributions! Please feel free to:

- 🐛 Report bugs
- 💡 Suggest features
- 📖 Improve documentation
- 🔧 Submit pull requests

## 📄 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use BAPred in your research, please cite:

```bibtex
@software{bapred2024,
  title={BAPred: Protein-Ligand Binding Affinity Prediction using Graph Neural Networks},
  author={Jaemin Sim},
  year={2024},
  url={https://github.com/eightmm/BAPred}
}
```

## 🙋‍♀️ Support

- 📖 **Documentation**: Check this README and code comments
- 🐛 **Issues**: [GitHub Issues](https://github.com/eightmm/BAPred/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/eightmm/BAPred/discussions)

---

<div align="center">

**Made with ❤️ for the scientific community**

⭐ Star us on GitHub if this project helped you!

</div>

