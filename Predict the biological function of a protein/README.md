# CAFA 6 Protein Function Prediction

Deep learning solution for the CAFA 6 (Critical Assessment of Functional Annotation) protein function prediction competition using ESM-2 transformers.

## 🎯 Project Overview

This project implements a state-of-the-art protein function prediction model using the ESM-2 (Evolutionary Scale Modeling) transformer architecture. The model predicts Gene Ontology (GO) terms for protein sequences, addressing one of the fundamental challenges in computational biology.

### Key Features

- **ESM-2 Transformer Model**: Utilizes the 650M parameter `facebook/esm2_t33_650M_UR50D` model for protein representation learning
- **Multi-label Classification**: Predicts up to 1,500 GO terms per protein
- **GO Graph Score Propagation**: Implements ontology-aware score propagation to ensure consistency with GO hierarchy
- **Fine-tuning Strategy**: Full model fine-tuning for optimal performance on Azure compute
- **Reproducible Pipeline**: Seed-controlled experiments with comprehensive data preprocessing

## 📊 Results

- **Model**: ESM-2 (650M parameters)
- **Training Classes**: 1,500 most frequent GO terms
- **Training Epochs**: 10
- **Batch Size**: 2 (optimized for large model)
- **Learning Rate**: 5e-5
- **Max Sequence Length**: 512 tokens

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/KinannAlamir/Kaggle.git
cd "Kaggle/Predict the biological function of a protein"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using uv (Fast)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/KinannAlamir/Kaggle.git
cd "Kaggle/Predict the biological function of a protein"

# Create environment and install dependencies
uv sync
```

### Important: urllib3 Compatibility

The project requires `urllib3<2.0` due to compatibility issues with transformers and botocore. This is automatically handled in the requirements.

## 📁 Project Structure

```
.
├── main.ipynb                          # Main training notebook
├── main.py                             # Python script version (if needed)
├── README.md                           # This file
├── requirements.txt                    # Pip dependencies
├── pyproject.toml                      # UV/pip project configuration
├── .gitignore                          # Git ignore rules
├── LICENSE                             # MIT License
└── cafa-6-protein-function-prediction/ # Data directory (not in repo)
    └── data/
        ├── Train/
        │   ├── go-basic.obo
        │   ├── train_sequences.fasta
        │   ├── train_terms.tsv
        │   └── train_taxonomy.tsv
        └── Test/
            ├── testsuperset.fasta
            └── testsuperset-taxon-list.tsv
```

## 🚀 Usage

### 1. Download Competition Data

Download the CAFA 6 dataset from [Kaggle](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction) and place it in:
```
cafa-6-protein-function-prediction/data/
```

### 2. Run the Notebook

```bash
jupyter notebook main.ipynb
```

Or use VS Code with Jupyter extension.

### 3. Training Pipeline

The notebook follows these steps:

1. **Environment Setup**: Install dependencies and verify urllib3 version
2. **Data Loading**: Load GO ontology, protein sequences, and annotations
3. **EDA**: Analyze sequence lengths and GO term distributions
4. **Preprocessing**: Tokenize sequences with ESM-2 tokenizer
5. **Model Definition**: Create ESM-2 + classification head architecture
6. **Training**: Fine-tune on training set with validation monitoring
7. **Prediction**: Generate predictions for test set
8. **Post-processing**: Apply GO graph score propagation
9. **Submission**: Create competition-format TSV file

### 4. Key Configuration Options

In the notebook, you can adjust:

```python
model_checkpoint = "facebook/esm2_t33_650M_UR50D"  # Model size
NUM_CLASSES = 1500      # Number of GO terms to predict
BATCH_SIZE = 2          # Batch size (reduce if OOM)
EPOCHS = 10             # Training epochs
MAX_LEN = 512           # Max sequence length
```

## 🧬 Model Architecture

```
ESM2ForProteinFunction(
  (esm): EsmModel (650M parameters)
    - 33 transformer layers
    - 20 attention heads
    - 1280 hidden dimension
  (fc): Linear(1280 → 1500)
  (sigmoid): Sigmoid()
)
```

**Key Design Decisions:**
- Uses CLS token representation for sequence-level prediction
- Full model fine-tuning (no frozen layers)
- Binary cross-entropy loss for multi-label classification
- Adam optimizer with low learning rate (5e-5)

## 📈 Performance Optimization

### For Azure/Cloud Environments

- **Batch Size**: Start with 2 and increase if GPU memory allows
- **Epochs**: 10 epochs provides good convergence
- **Gradient Accumulation**: Not implemented but can be added for larger effective batch sizes
- **Mixed Precision**: Can be enabled with `torch.cuda.amp` for faster training

### For Local Development

- Consider using smaller models: `facebook/esm2_t12_35M_UR50D` or `facebook/esm2_t6_8M_UR50D`
- Reduce `NUM_CLASSES` to 500
- Freeze ESM layers (uncomment line in cell 17)

## 🔬 Methodology

### Score Propagation Algorithm

The model implements GO-aware score propagation:

```python
# For each predicted term:
#   propagate score to all ancestor terms
#   P(parent) = max(P(parent), P(child))
```

This ensures predictions respect the GO hierarchy's true-path rule.

### Data Preprocessing

- Sequences truncated to 512 tokens (ESM-2 maximum)
- UniProt IDs extracted from FASTA headers
- Terms grouped by protein for efficient label encoding
- 80/20 train/validation split

## 🐛 Troubleshooting

### urllib3 ImportError

If you see `cannot import name 'DEFAULT_CIPHERS'`:

1. Run the first cell in the notebook to reinstall urllib3
2. **Restart the kernel** (critical!)
3. Re-run cells sequentially

### FileNotFoundError for data files

The notebook uses absolute paths. If data is not found:
- Verify data is in `cafa-6-protein-function-prediction/data/`
- Check the `BASE_DIR` variable in cell 6

### CUDA Out of Memory

- Reduce `BATCH_SIZE` to 1
- Reduce `MAX_LEN` to 256
- Use a smaller model checkpoint

## 📚 References

- **ESM-2 Paper**: [Evolutionary-scale prediction of atomic-level protein structure with a language model](https://www.science.org/doi/10.1126/science.ade2574)
- **CAFA Competition**: [Critical Assessment of Function Annotation](https://biofunctionprediction.org/cafa/)
- **Gene Ontology**: [The Gene Ontology Resource](http://geneontology.org/)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add test-time augmentation
- [ ] Implement ensemble methods
- [ ] Add attention visualization
- [ ] Support for longer sequences (up to 1024)
- [ ] Multi-GPU training with DDP

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Meta AI for the ESM-2 model
- CAFA organizers for the competition and dataset
- HuggingFace for the transformers library

## 📧 Contact

**Kinann Alamir**
- GitHub: [@KinannAlamir](https://github.com/KinannAlamir)
- Repository: [Kaggle Competition Solutions](https://github.com/KinannAlamir/Kaggle)

---

*Last Updated: December 2025*