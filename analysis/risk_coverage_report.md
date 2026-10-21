# V7.2-B1：风险–覆盖率（选择性预测）分析结果

## 数据
- MOSI：4 seeds 逐样本 preds+confidence（由 `saved_models/v3_CGF_{seed}.pt` 重新评估导出，逐 seed 准确率与 V7.1 完全一致：65.45/67.06/66.03/66.76%），合并 2,744 条。
- MOSEI：`progress_mosei_controlled.json` 的 CGF-full 4 seeds（修复归一化后重训：61.52/62.23/62.14/62.91%），合并 18,648 条。
- 脚本：`analysis/risk_coverage.py`；逐样本记录：`analysis/mosi_test_records.json`；机器可读结果：`analysis/risk_coverage_results.json`。
- 口径说明：同一测试样本被 4 个种子重复预测（与 4.6 节既有口径一致），分析为描述性。

## 核心结果

| 数据集 | 全集准确率 | AURC（置信度排序） | 随机拒绝基线 AURC | coverage=50% 时准确率 | Pearson r |
|---|---|---|---|---|---|
| CMU-MOSI | 66.33% | **21.13%** | 33.67% | **78.94%** | 0.302 |
| CMU-MOSEI | 62.20% | **29.20%** | 37.80% | **70.17%** | 0.196 |

- 两个数据集上，按置信度降序保留预测都比随机拒绝基线**一致且明显地降低风险**。
- MOSI 在 coverage=50%（拒判一半）时准确率从 66.33% 提到 78.94%（+12.6 pp）；MOSEI 从 62.20% 提到 70.17%（+8.0 pp）。
- coverage→risk 曲线全程位于随机基线下方（见 risk_coverage_results.json）。

## 对论文的意义
- 直接回应审稿意见"calibration and utility for selective prediction remain untested"：
  现在有实证证据表明置信度分数具有**选择性预测效用**（relative diagnostic signal → decision utility）。
- 措辞边界：仍不是校准（calibration）证据；且记录跨种子合并、样本非独立，结论为描述性。
- 建议写入 4.6 节末：AURC 两数据集数值 + coverage=50% 时的准确率提升，1 个短段即可。
