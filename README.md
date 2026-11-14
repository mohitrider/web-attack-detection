
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

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Reproducing the Experiments

Obtain the two dataset CSV files described in `data/README.md` and place them in `data/raw/`.

From the repository root:

1. Run scripts `01–05` in order to reproduce the data audits.
2. Run scripts `06–11` in order to reproduce the data split, main model comparison, and reported depth analyses.

Scripts `09` and `10` use predictions produced by script `07`.

Outputs are written to `results/`. All experiments use random seed `2026` where applicable. Generated intermediate data and large prediction files are not included in the repository.

## Project Dashboard

The project results dashboard is available at:

**https://mohitrider.github.io/web-attack-detection/**

The dashboard is an offline results explorer presenting selected experimental results. It is not a live intrusion-detection system.

## Limitations

- Only two dates are evaluated.
- The study covers three web-attack categories.
- Attack examples are very rare compared with benign flows.
- Repeated feature signatures across dates limit how broadly cross-day independence can be interpreted.
- The findings do not establish generalisation across other networks, dates, or attack categories.

## AI-Use Disclosure

ChatGPT and Claude were used for conceptual explanations, code development and debugging assistance, and editing suggestions. The experiments were run by the student, who checked the reported metrics against the saved result files and takes responsibility for the submitted work.

## References

1. Sharafaldin, I., Lashkari, A. H., and Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, 108–116.

2. A Realistic Cyber Defense Dataset (CSE-CIC-IDS2018). Registry of Open Data on AWS.  
   https://registry.opendata.aws/cse-cic-ids2018/

3. Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.  
   https://doi.org/10.1145/2939672.2939785

The dataset is not redistributed in this repository. Users must obtain it from the dataset source and follow its applicable access and usage terms.
