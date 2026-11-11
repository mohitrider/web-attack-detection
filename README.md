# Web Attack Detection Using Network Flow Features

**ES 678 — MTech AI Project**  
**Indian Institute of Technology Gandhinagar**

## Project Overview

This project evaluates whether machine-learning models trained on network traffic from one day can detect web attacks in traffic collected on another day.

The study uses the CSE-CIC-IDS2018 dataset and focuses on three attack categories:
- Brute Force-Web
- Brute Force-XSS
- SQL Injection

The task is binary classification: Benign (0) versus Web Attack (1). Each observation represents a network flow described by CICFlowMeter-derived features.

## Research Question

To what extent can traditional machine-learning models detect application-layer web attacks using network-flow features, and how well do they generalize across different days?

Two transfer directions are evaluated:
- Thursday, 22 February 2018 → Friday, 23 February 2018
- Friday, 23 February 2018 → Thursday, 22 February 2018

## Dataset

| Date | Total flows | Attack flows |
|---|---:|---:|
| Thursday, 22 February 2018 | 1,048,575 | 362 |
| Friday, 23 February 2018 | 1,048,575 | 566 |
| **Combined** | **2,097,150** | **928** |

The raw dataset is not included in this repository. See [data/README.md](data/README.md) for download and placement instructions.

## Models

Three classifiers are compared:
1. Logistic Regression
2. Linear Support Vector Machine
3. XGBoost

The decision threshold is selected using validation data from the source day and kept fixed when evaluating on the opposite day.

## Main Results

| Train → Test | F1-score | Average Precision |
|---|---:|---:|
| Thursday → Friday | 0.8195 | 0.7111 |
| Friday → Thursday | 0.9071 | 0.9421 |

The transfer directions show different performance. These findings are limited to the two dates and attack categories evaluated.

## Graduate-Level Analysis

The project includes:
- Feature-group ablation.
- Training-composition sensitivity analysis.
- False-positive and false-negative analysis.
- Seen/unseen feature-signature analysis.

These experiments examine model behaviour beyond the main performance scores.

## Repository Structure

- `src/` — numbered audit, preparation, training and analysis scripts.
- `results/` — compact experiment outputs.
- `reports/` — local reports are excluded from the public repository.
- `paper/` — LaTeX paper source and formatting style.
- `docs/` — static results dashboard.
- `data/` — dataset access and placement instructions.

## Environment Setup

The experiments were developed using Python 3.14.7.

Create and activate a virtual environment:

    python -m venv .venv
    source .venv/bin/activate

Install dependencies:

    python -m pip install -r requirements.txt

## Reproducing the Experiments

Obtain the two dataset CSV files described in `data/README.md` and place them in `data/raw/`.

Run the numbered scripts from the repository root in order, beginning with data auditing and preparation, followed by model training and the additional analyses.

The scripts use random seed 2026 where applicable. Generated intermediate data and large prediction files are not included in the repository.

## Limitations

- Only two dates are evaluated.
- The study covers three web-attack categories.
- Attack examples are very rare compared with benign flows.
- Repeated feature signatures across dates limit how broadly cross-day independence can be interpreted.
- The dashboard is an offline results explorer, not a live intrusion-detection system.

## AI-Use Disclosure

ChatGPT was used for conceptual explanations, code development and debugging assistance, and editing suggestions. The experiments and reported results were checked by the student, who takes responsibility for the submitted work.

## Dataset Reference

A Realistic Cyber Defense Dataset (CSE-CIC-IDS2018):  
https://registry.opendata.aws/cse-cic-ids2018/

The dataset is not redistributed in this repository. Users must obtain it from the dataset source and follow its applicable access and usage terms.
