| model | family | n_seeds | n_train | tpr_at_fpr | roc_auc | pr_auc | brier | fpr_ratio | tpr_ratio | tpr_seed_std | tpr_ci_low | tpr_ci_high | fit_seconds | predict_seconds | best_model | tpr_diff_to_best | p_value_vs_best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tabm | deep_learning | 1 | 794989 | 0.5459 | 0.8908 | 0.1904 | 0.0124 | 0.3530 | 0.7041 |  | 0.5271 | 0.5632 | 405.1652 | 2.0703 | tabm | 0.0000 |  |
| tabpfn | foundation | 1 | 50000 | 0.5323 | 0.8893 | 0.1904 | 0.0124 | 0.3024 | 0.6630 |  | 0.5122 | 0.5507 | 1.3157 | 772.1546 | tabm | -0.0136 | 0.0580 |
| lightgbm | classical | 1 | 794989 | 0.5278 | 0.8898 | 0.1704 | 0.0128 | 0.3420 | 0.6888 |  | 0.5090 | 0.5462 | 2.1886 | 0.2160 | tabm | -0.0181 | 0.0060 |
