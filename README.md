# Plant Traits Predictor

I’ve always been fascinated by plants and their incredible diversity. This project started as a way to combine my love for plants with my interest in machine learning, aiming to build a tool that can help identify plant species from photos while also considering environmental context.

Plant Traits Predictor is a deep learning project that predicts plant species from images and basic metadata (latitude, longitude, day-of-year). Built using PyTorch, this project combines image features with environmental metadata for more accurate plant identification.

## Features

- Image-based classification: using a CNN backbone (ResNet-based).
- Metadata integration: Incorporates latitude, longitude, and day-of-year for improved predictions.
- Top-k predictions: Returns the most likely species for a given plant image.
- Easy-to-use CLI: Run predictions directly from the command line.
- Dataset-ready: Works with iNaturalist-style CSV metadata and images.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Plant-predictions.git
cd Plant-predictions

