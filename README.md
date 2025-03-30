# Targeted FGSM Attack on MNIST

This project analyzes the behavior of a CNN model under **targeted adversarial attacks** using **FGSM (Fast Gradient Sign Method)** on the MNIST dataset.

## Goal

- Train a CNN on MNIST
- Apply **targeted FGSM** to mislead the model toward specific target classes
- Measure and visualize attack success rates across:
  - Various epsilon values (`ε = 0.1, 0.2, 0.3, 0.4`)
  - All (True class, Target class) combinations

## Files

| File | Description |
|------|-------------|
| `test.py` | Main script to train, attack, and generate heatmaps |
| `requirements.txt` | Required Python packages |
| `README.md` | This documentation |

## How to Run
```bash
#colab 실행시
from google.colab import files
uploaded = files.upload()
!pip install -r requirements.txt
!python test.py
```
```bash
pip install -r requirements.txt
python test.py
