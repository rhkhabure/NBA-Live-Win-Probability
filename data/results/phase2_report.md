## Phase 2 — Training (2026-09-29 12:00 UTC)

### Architecture
- Input: 16 features
- Hidden: 128 → 64 → 32
- Output: 1 (sigmoid)
- Dropout: 0.25  BatchNorm: True
- Parameters: 12,769

### Walk-forward validation
 fold    test    brier  roc_auc      bss        T
    1 2022-23 0.166299 0.832403 0.316866 1.015199
    2 2023-24 0.149079 0.866112 0.398246 1.012249
    3 2024-25 0.155402 0.855298 0.373205 0.953428

Brier drift: -0.0109

### Final model performance (2024-25 validation)
- ROC-AUC      : 0.8565
- PR-AUC       : 0.8808
- Brier Score  : 0.1551
- Brier Skill  : 0.3746
- Log Loss     : 0.4619
- Accuracy     : 0.7622
- Temperature T: 1.003438
- Pathwise rate: 0.095 (target 0.08-0.15)

### Validation gates: 8/8 passed

### ESPN benchmark
ESPN head-to-head: PENDING — requires nba_api→ESPN game ID mapping.
The ESPN summary endpoint (site.api.espn.com/nba/summary?event=<id>) is
confirmed working. ID mapping will be added in a post-Phase-2 patch.
For now, compare against published benchmarks: ESPN Brier ≈ 0.166 (Beuoy 2018).

### Saved artefacts
- model/win_prob_net_v3.pth
- model/scaler_v3.pkl
- model/calibration_t.pkl
- model/walk_forward_results.csv
- model/training_history.json
