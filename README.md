
# DetectAI

Official code repository for the paper  
**Can We Trust LLM Detectors?**  
📄 arXiv: https://arxiv.org/abs/2601.15301

This repository contains Jupyter notebooks for evaluating **training-free** and **trained (supervised + contrastive)** AI text detectors.

---

## Repository Structure

```

DetectAI/
│
├── training_free.ipynb      # Training-free detectors (zero-shot)
├── trained_detectors.ipynb  # Supervised & contrastive detectors
└── README.md

````

---

## Models & Datasets

All datasets, trained weights, and embeddings are hosted on Hugging Face:

👉 https://huggingface.co/chhola14bhatoora/detectai

No large files are stored directly in this repository.

---

## Setup

### Requirements
- Python ≥ 3.9
- PyTorch
- HuggingFace `transformers`
- `datasets`
- `scikit-learn`
- `numpy`, `pandas`

Install dependencies:
```bash
pip install torch transformers datasets scikit-learn numpy pandas
````

---

## Usage

⚠️ **Important:**
In **both notebooks**, you only need to modify **paths in the FIRST cell**:

* Dataset locations
* Model checkpoint paths
* (Optional) Hugging Face cache directory

No other code changes are required.

---

### `training_free.ipynb`

* Runs training-free detectors
* Zero-shot evaluation
* Proxy-model sensitivity analysis
* In-domain and OOD evaluation

---

### `trained_detectors.ipynb`

* Supervised detectors (BERT, GAN-BERT)
* Contrastive learning–based detector
* Few-shot adaptation to unseen LLMs
* Robustness and adversarial analysis

---

## Reproducibility

* All experiments are fully reproducible using the provided notebooks
* Pretrained weights and processed datasets are loaded directly from Hugging Face
* Notebooks can be run sequentially after setting paths

---

## Citation

If you use this code or models, please cite:

```bibtex
@misc{sandhan2026trustllmdetectors,
      title={Can We Trust LLM Detectors?}, 
      author={Jivnesh Sandhan and Harshit Jaiswal and Fei Cheng and Yugo Murawaki},
      year={2026},
      eprint={2601.15301},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.15301}, 
}
```


---
