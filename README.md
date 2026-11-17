# PyTorch Image Classifier

> A Streamlit app that classifies and analyses images with PyTorch models (ResNet-50 and Faster R-CNN).

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)


## Features

- Image classification with ResNet-50 (ImageNet labels).
- Object detection with Faster R-CNN.
- Uses GPU when available, CPU otherwise.
- Streamlit UI.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Models | PyTorch, torchvision (ResNet-50, Faster R-CNN) |
| Images | Pillow |

## Quick Start

```bash
git clone https://github.com/dastonbek8747/pytorch-image-classifier.git
cd pytorch-image-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

## License

MIT © 2026 Doston Sultonov.
