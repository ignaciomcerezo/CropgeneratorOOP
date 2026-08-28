# cropgen

`cropgen` is a Python package for generating OCR training datasets from annotated document images. It provides a PyTorch-compatible `OCRDataset` class that samples variable-length sequences of document lines with optional augmentation transforms.

## Overview

This repository provides dataset generation and augmentation for handwritten document OCR. It is designed to:

- ingest Label Studio annotation exports,
- extract and normalize image crops with automatic layout analysis,
- produce an `OCRDataset` that samples contiguous line sequences at training time,
- apply trainable augmentation transforms (line-wise, intra-paragraph, inter-paragraph),
- handle complex geometry with automatic intersection correction and stroke/background separation.

## Main components

- **`cropgen.shared`**  
  Shared data structures and utilities used across the pipeline.

- **`cropgen.external_interfaces`**  
  Interfaces to external systems: Label Studio for annotations and Oracle Cloud for storage.

- **`cropgen.processing`**  
  Core processing logic for turning annotations into image/text samples with geometric analysis.

- **`cropgen.datasets`**  
  The main `OCRDataset` (and other dataset variants) class for training, with configurable line-sequence sampling and clustering.

- **`cropgen.transforms`**  
  Image and geometry augmentation transforms: linewise (distortion, stretching), intra-paragraph (paragraph layout modifications), and inter-paragraph (multi-line sampling, moving paragraphs).

- **`cropgen.tests`**  
  Test suite for validating the pipeline.

## Package metadata

- **Package name:** `cropgen`
- **Python:** `>=3.10`
- **Primary dependencies:** `numpy`, `scipy`, `datasets`, `fuzzywuzzy`, `pandas`, `pydantic`, `shapely`, `pillow`, `requests`, `tqdm`, `label-studio-sdk`
- **Training extras:** `torch`, `transformers`, `trl`, `accelerate`, `unsloth`, `bitsandbytes`, `triton`, `sentencepiece`, `huggingface-hub`

## Intended use

The package is intended for OCR training workflows where:

- source documents are annotated in Label Studio,
- variable-length sequences of document lines are needed (single lines, paragraphs, full pages),
- on-the-fly augmentation is desired during training,
- geometric transforms (rotation, scaling, distortion) should be applied to line crops,
- and the dataset integrates with `torch.utils.data.Dataset` for PyTorch training.

## Installation

Install the core package:

```bash
pip install .
```

Install training-related extras:

```bash
pip install ".[train]"
```
