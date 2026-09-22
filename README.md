# RAMT-CGPA Reproducibility Package

Reproducible research implementation for the manuscript:

**Adaptive Risk-aware Multi-Tier Cryptography Framework for Secure Heart Disease Prediction through Distributed Blockchain Technology**

The repository implements the experimental workflow of the proposed **Risk-Adaptive Multi-Tiered Cryptographic Governance and Predictive Analytics (RAMT-CGPA)** framework in a flat, reviewer-friendly layout. The code is organized around the following reproducibility targets:

- information-theoretic sensitivity and utility-aware clinical attribute tiering;
- synthetic PII generation for controlled privacy-governance experiments;
- privacy/integrity processing through local differential privacy and BLAKE3-compatible hashing;
- threshold-based secret reconstruction and post-quantum KEM wrappers;
- Spherical Fuzzy Rough Set inspired feature reduction;
- Multi-Head Self-Attention Kolmogorov-Arnold representation learning;
- Adaptive Jellyfish Search optimization of XGBoost;
- classical, deep-learning, and Transformer-style baselines;
- DAG-style transaction benchmarking and emergency break-glass timing;
- cross-validation, confidence intervals, statistical significance tests, and scalability analysis.

## Important reproducibility note

This repository is designed to regenerate experimental measurements from executable code. Manuscript values are **not hard-coded as model outputs**. Exact numerical agreement depends on the same dataset version, preprocessing decisions, hardware, library versions, cryptographic backend, and benchmark load used in the study.

The package does not redistribute restricted clinical datasets. MIMIC-IV must be obtained through authorized PhysioNet access. PTB-XL and UCI datasets must be downloaded from their official sources or supplied locally.

## Flat repository

```
README.md
requirements.txt
config.yaml
reproduce_all.py
data_prepare.py
synthetic_pii.py
cait_qim.py
crypto_governance.py
dag_ledger.py
breakglass_stark.py
sfrs.py
mhkan.py
ajs_xgboost.py
baselines.py
train_evaluate.py
security_benchmark.py
statistical_analysis.py
```

## Environment

Recommended:

- Python 3.10+
- PyTorch 2.2+
- XGBoost 2.0+
- scikit-learn 1.4+
- NumPy 1.26+

Create an isolated environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset placement

Dataset paths are configured in `config.yaml`.

Supported input modes:

1. **UCI-compatible CSV**
2. **MIMIC-IV derived cardiovascular table**
3. **PTB-XL derived patient-level table**
4. **Generic tabular clinical CSV**
5. **Synthetic demo data**, enabled when no real dataset is available

Expected target options:

- `target`
- `num`
- `label`
- `class`
- `severity`

Target values are normalized into a five-class severity space `S0`–`S4` when possible.

## One-command reproduction

```bash
python reproduce_all.py --config config.yaml
```

A fast smoke test can be run with:

```bash
python reproduce_all.py --config config.yaml --demo
```

The script writes experiment outputs as CSV/JSON files into the current directory. No additional folders are required.

## Experimental flow

```
Clinical records
    |
    +--> Synthetic PII overlay
    |
    +--> CAIT-QIM sensitivity/utility tiering
             |
             +--> Tier 1: threshold / KEM benchmark
             +--> Tier 2: LDP + integrity hashing
             +--> Tier 3: DAG audit receipt
    |
    +--> SFRS feature reduction
    |
    +--> MH-KAN representation
    |
    +--> AJS-XGBoost classifier
    |
    +--> Cross-validation and external-cohort testing
    |
    +--> Statistical significance analysis
```

## Reviewer-oriented outputs

The package produces or supports generation of:

- fold-wise Accuracy, Macro-Precision, Macro-Recall, Macro-F1 and PR-AUC;
- 95% bootstrap confidence intervals;
- Wilcoxon signed-rank comparisons;
- Friedman multi-model comparison;
- Holm-adjusted pairwise post-hoc analysis;
- model inference latency;
- KEM and threshold-sharing timing;
- local differential privacy processing latency;
- BLAKE3/Merkle-style integrity timing;
- DAG throughput and latency;
- ledger-growth estimates;
- break-glass transition latency;
- scalability measurements across node and transaction counts;
- modern neural baselines including MLP, LSTM, Bi-LSTM, Transformer Encoder and attention-based tabular Transformer.

## Security implementation note

Where a native post-quantum library such as `liboqs-python` is available, the wrapper attempts to use it. When unavailable, a benchmark-safe fallback is used so the complete experimental pipeline remains executable. The fallback is **not presented as a cryptographic substitute for ML-KEM/Kyber** and is marked accordingly in output metadata.

The DAG module is an experimental consortium-ledger simulator intended to measure the architecture's throughput, transaction verification, storage growth and transition timing. It is not a production blockchain client.

The break-glass module implements the manuscript's control-flow and timing interface. A genuine zk-STARK prover may be integrated through the same API; the default implementation performs deterministic transcript commitment and verification to permit reproducible end-to-end benchmarking without claiming formal zero-knowledge security.

## Main configuration groups

`config.yaml` controls:

- random seed;
- dataset paths and target column;
- train/validation/test proportions;
- cross-validation folds;
- privacy epsilon/delta;
- CAIT-QIM thresholds;
- SFRS reduction limits;
- MH-KAN architecture;
- AJS population and iteration counts;
- XGBoost search bounds;
- ledger node counts;
- benchmark transaction volumes;
- AVSS normal/emergency thresholds;
- benchmark repetitions;
- statistical confidence level.

## Reproducibility principles

- preprocessing is fitted on training data only;
- all random components are explicitly seeded;
- cross-validation is stratified when labels permit it;
- fold-level metrics are retained;
- baseline models are evaluated using the same split;
- no test sample is used for hyperparameter selection;
- external cohorts can be evaluated separately;
- security measurements are generated from timed operations rather than copied from the manuscript.

## Typical commands

Prepare a dataset:

```bash
python data_prepare.py --config config.yaml
```

Train and compare predictive models:

```bash
python train_evaluate.py --config config.yaml
```

Run security and ledger benchmarks:

```bash
python security_benchmark.py --config config.yaml
```

Run statistical analysis:

```bash
python statistical_analysis.py --input predictive_results.csv
```

## Expected generated files

Typical outputs include:

```
prepared_clinical.csv
tier_assignments.csv
selected_features.json
predictive_results.csv
baseline_results.csv
security_benchmark.csv
dag_scalability.csv
statistical_results.json
reproducibility_summary.json
```

## Clinical-use disclaimer

This repository is a research reproducibility package. It is not a certified diagnostic system, medical device, clinical access-control product, or audited cryptographic deployment.
