# XGBoost Library Performance Experiment
This repository contains a brief experiment evaluating the performance of v1 and v2 versions of the XGBoost.jl library. It was inspired by the discussion in the XGBoost.jl library's (issue #160)[https://github.com/dmlc/XGBoost.jl/issues/160] and aims to address concerns about instability in version 2 that hinder its use for learning to rank tasks. The objective is to clear up any confusion on this matter and improve XGBoost for the benefit of the community.

## Introduction 

While updating the XGBoost library, I noticed a decrease in performance compared to previous versions. After conducting an investigation, I discovered one possible cause of this issue. My suspicion is that the scores generated for documents during prediction time are inconsistent after training. The experiment results show that this inconsistency occurs when the order between train and test data is changed and this does not happen with version 1 of the library. Additionally, a detailed chart of scores indicates a clear degradation in comparison to previous versions. The exact internal cause is unclear, but version 2 appears to be unstable for ranking tasks.

## Dependencies
- None, except XGBoost that will be installed for the test 

## Run Experiment
1. Clone Repo `git clone git@github.com:dmoliveira/xgboost-benchmark.git`
2. Run your experiment `VERSION=<XGBoost-VERSION> ITE=<NUM-ROUNDS-XGBOOST> ./run_xgb_experiment.jl`

**Examples:**
```
# Run XGBoost v1
VERSION=1.5.2 ITE=100 ./run_xgb_experiment.jl

# Run XGBoost v2
VERSION=2.2.3 ITE=100 ./run_xgb_experiment.jl
```

## Data Files 
In addition to the script for evaluating XGBoost, we have included two data files for training (train.svmlight) and testing (test.svmlight). These data files include group IDs (qids) to enable the ranking objective function in XGBoost.

- **Train Data X:** (10008, 30571) Y:10008 QIDs:1686
- **Test Data X:** (1000, 30564) Y:1000 QIDs:134

## Experiment Results

**Results Run XGBoost v1.5.2**
```
[ Info: (1) TRAIN - Precision@N: p@5:0.87714 p@10:0.85536 p@20:0.83332
[ Info: (2) TEST - Precision@N: p@5:0.88097 p@10:0.85939 p@20:0.83138
```

Invert Test and Train for Evaluation
```
[ Info: (1) TRAIN - Precision@N: p@5:0.88097 p@10:0.85939 p@20:0.83138
[ Info: (2) TEST - Precision@N: p@5:0.87714 p@10:0.85536 p@20:0.83332
```

**Results Run XGBoost v2.2.3**
```
[ Info: (1) TRAIN - Precision@N: p@5:0.853 p@10:0.83667 p@20:0.82057
[ Info: (2) TEST - Precision@N: p@5:0.87873 p@10:0.85398 p@20:0.83702
```

Invert Test and Train for Evaluation
```
[ Info: (1) TRAIN - Precision@N: p@5:0.54813 p@10:0.53545 p@20:0.51238
[ Info: (2) TEST - Precision@N: p@5:0.88808 p@10:0.87128 p@20:0.85495
```

## Conclusion
Please stay updated on the investigation by following issue 160 at https://github.com/dmlc/XGBoost.jl/issues/160.
