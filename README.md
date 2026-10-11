# CanvasFlow — Neural Style Transfer

A neural style transfer web application built using **Adaptive Instance Normalisation (AdaIN)**, implemented from scratch in PyTorch with a Flask front-end.

This project was developed as a final-year deep learning assignment to explore real-time artistic style transfer without per-image optimization.

---

## What it does

Given a **content image** (the structural subject) and a **style reference** (an artwork or painting), the model produces a new image that retains the content's structure while adopting the visual texture and colour palette of the style reference.

The **α (alpha)** parameter controls transfer strength — interpolating linearly between the original content features and the stylized features before decoding.

---

## How it works

The model follows the AdaIN architecture (Huang & Belongie, 2017):

1. A pre-trained **VGG-19 encoder** (truncated at `relu4_1`) extracts deep feature maps from both the content and style images.
2. **Adaptive Instance Normalisation** aligns the channel-wise mean and variance of the content features with those of the style features — a single, fast, feed-forward operation.
3. A **trained convolutional decoder** reconstructs a full-resolution image from the normalized feature representation.

Training minimizes a weighted sum of:
- **Content loss**: MSE between the decoder output's `relu4_1` features and the AdaIN target
- **Style loss**: MSE between channel statistics (mean + std) of the output and style image across `relu1_1` through `relu4_1`

---

## Project structure

```
NST_Code/
├── app.py                  Flask application
├── train.py                Training script
├── vgg_normalised.pth      Pre-trained VGG encoder weights
├── utils/
│   ├── models.py           VGGEncoder + Decoder architecture
│   └── utils.py            AdaIN, dataset, transform helpers
├── templates/
│   └── index.html          Jinja2 template
├── static/
│   ├── css/style.css       Stylesheet
│   ├── js/app.js           Client-side logic
│   └── uploads/            Uploaded and generated images (runtime)
├── examples/               Sample input/output images
├── content_data/           Content training images (MS-COCO)
├── style_data/             Style training images (WikiArt)
└── experiment/             Saved decoder checkpoints
```

---

## Setup

### Requirements

```
Python 3.9+
PyTorch 2.x
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the app

Set the decoder path if it differs from the default (`experiment/final_exp/decoder_final.pth`):

```bash
export DECODER_PATH=/path/to/decoder_final.pth
```

Then start the server:

```bash
cd NST_Code
python app.py
```

Open `http://localhost:5000` in your browser.

### Training from scratch

```bash
cd NST_Code
python train.py \
  --content_dir /path/to/content_data \
  --style_dir   /path/to/style_data \
  --vgg         vgg_normalised.pth \
  --experiment  my_run \
  --epochs      20 \
  --batch_size  4
```

Checkpoints are saved to `experiment/<name>/` every `--save_interval` epochs.

---

## Reference

Huang, X., & Belongie, S. (2017). *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*. ICCV 2017. [arXiv:1703.06868](https://arxiv.org/abs/1703.06868)
