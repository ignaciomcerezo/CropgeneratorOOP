# cropgen

`cropgen` is a Python package for generating training datasets from annotated document images. It converts full-page images and Label Studio annotations into cropped image/text pairs suitable for vision-language model fine-tuning, and also includes supporting utilities for training workflows.

## Overview

This repository provides the first stage of a dataset generation pipeline for handwritten document transcription. It is designed to:

- ingest Label Studio annotation exports,
- extract and normalize image crops,
- align each crop with its corresponding transcription,
- generate augmented samples from neighboring lines,
- prepare datasets in a `datasets.Dataset`-friendly format,
- and support downstream training with custom helpers such as collators and callbacks.

## Main components

- **`cropgen.shared`**  
  Shared data structures and utilities used across the pipeline.

- **`cropgen.external_interfaces`**  
  Interfaces to external systems such as Label Studio and remote storage.

- **`cropgen.processing`**  
  Core processing logic for turning annotations into image/text samples.

- **`cropgen.splitter`**  
  Dataset splitting logic, including train/test separation.

- **`cropgen.training_helpers`**  
  Helpers for training, such as collators, callbacks, and evaluation utilities.

- **`cropgen.tests`**  
  Test suite for validating the pipeline.

## Package metadata

- **Package name:** `cropgen`
- **Python:** `>=3.10`
- **Primary dependencies:** `numpy`, `scipy`, `datasets`, `fuzzywuzzy`, `pandas`, `pydantic`, `shapely`, `pillow`, `requests`, `tqdm`, `label-studio-sdk`
- **Training extras:** `torch`, `transformers`, `trl`, `accelerate`, `unsloth`, `bitsandbytes`, `triton`, `sentencepiece`, `huggingface-hub`

## Intended use

The package is meant for workflows where:

- source documents are already annotated in Label Studio,
- line-level crops and transcriptions must be produced automatically,
- synthetic multi-line samples are useful,
- and the resulting dataset will be used to fine-tune a multimodal model.

## Installation

Install the core package:

```bash
pip install .
```

Install training-related extras:

```bash
pip install ".[train]"
```
