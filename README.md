# Erupt

This is the artifact for the paper __Learned UDF Placement for Efficient Query Optimization__.

## Setup

```bash
conda create -n erupt python=3.9.23
conda activate erupt
python -m pip install -r requirements.txt
```

## Run

Extract UDF features:

```bash
python udf_feature_extractor.py --db_name imdb
```

Training (Latency-based):

```bash
python run_train.py --db_name imdb --algo dqn
python run_train.py --db_name imdb --algo ppo
```

Training (Cost-based):

```bash
python run_train.py --db_name imdb --algo dqn --use_cost_model True
python run_train.py --db_name imdb --algo ppo --use_cost_model True
```

Test:

```bash
python run_test.py --db_name imdb --algo ppo --model_path xxxx_xxxxxx_ppo_best_policy_val.pt
```

## Project Structure

- `datasets/`: datasets, workloads, UDFs.
- `models/`: trained models.
- `cache/`: cached execution plans.
- `logs/`: training logs.

## Datasets

- IMDB: Follow instructions from [IMDB](https://github.com/gregrahn/join-order-benchmark); UDFs from [DataManagementLab/Graceful](https://github.com/DataManagementLab/Graceful).
- TPC-H: Download from the official [TPC-H](https://www.tpc.org/tpch/); UDFs from [DataManagementLab/Graceful](https://github.com/DataManagementLab/Graceful).
- UDFBench: Download from the official [UDFBench](https://github.com/athenarc/UDFBench). After loading the data, run [`datasets/udfbench/rename.sql`](datasets/udfbench/rename_column.sql) to rename the columns.

## Baselines

- Opt-rank: Chaudhuri and Shim, [*Optimization of Queries with User-Defined Predicates*](https://doi.org/10.1145/320248.320249).
- RTOS: [TsinghuaDatabaseGroup/RTOS](https://github.com/TsinghuaDatabaseGroup/AI4DBCode/tree/master/RTOS).
- Graceful: [DataManagementLab/Graceful](https://github.com/DataManagementLab/Graceful).
