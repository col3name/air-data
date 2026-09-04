# Taxi-Out Prediction

CatBoost baseline for the PRC Data Challenge taxi-out prediction task.

## Expected data layout

```
data/
├── train/
│   ├── 2025-01.parquet
│   ├── 2025-02.parquet
│   └── ...
└── ranking.parquet
└── submitting.parquet
```

## Install

```
pip install pandas pyarrow numpy scikit-learn catboost
```

## Train

```
python train.py
```

The pipeline includes:
- temporal features
- airport/airline/aircraft/destination interactions
- airport/runway/stand interactions
- 15/30/60 minute traffic counts
- runway load
- previous taxi-out statistics
- traffic/queue/runway regimes
- temporal validation for September-December
- CatBoost RMSE model
- saved models and feature list

## Important

The rolling target features use previous observations only. Do not replace
`closed="left"` or `shift(1)` with a centered/current-inclusive window.

This is a strong first baseline. For the final competition model, the next
step should be airport-normalized congestion regimes, airport-specific
historical baselines, and an inference pipeline for ranking.parquet.
