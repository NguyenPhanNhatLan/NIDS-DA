# Tổng hợp class-aware development — 17/10/2026

Chiều UNSW → CICIDS, target validation; 5 seeds 42–46. Số liệu mean ± sample std (ddof=1). Workspace có 4 variants ablation đã đủ 5 seeds, nên báo cáo bao gồm cả 4 để tránh bỏ sót.

| Variant | AP | ROC-AUC | Macro-F1 | Recall | FPR |
| --- | --- | --- | --- | --- | --- |
| Source-only | 0.162569 ± 0.054400 | 0.385784 ± 0.104666 | 0.468429 ± 0.042274 | 0.111992 ± 0.110380 | 0.157195 ± 0.062177 |
| Reference | 0.180219 ± 0.059091 | 0.465906 ± 0.131902 | 0.483777 ± 0.039071 | 0.192798 ± 0.246543 | 0.177964 ± 0.142129 |
| sep_only | 0.163329 ± 0.042618 | 0.404427 ± 0.035255 | 0.470393 ± 0.036893 | 0.129074 ± 0.107729 | 0.170376 ± 0.096338 |
| sep003 | 0.150025 ± 0.031216 | 0.379554 ± 0.104353 | 0.447220 ± 0.079314 | 0.126977 ± 0.098402 | 0.240988 ± 0.126141 |
| sep010 | 0.203123 ± 0.012655 | 0.527468 ± 0.023879 | 0.503114 ± 0.032640 | 0.227260 ± 0.092913 | 0.210692 ± 0.101107 |
| tau90_95 | 0.163680 ± 0.049805 | 0.427971 ± 0.139777 | 0.465445 ± 0.044711 | 0.118566 ± 0.114100 | 0.172123 ± 0.062121 |

Reference: λMMD=0.001, λSep=0.001, τNormal=τAttack=0.90, margin=0.5.
sep_only: λMMD=0, λSep=0.001. sep003/sep010: λMMD=0.001, λSep=0.003/0.010. tau90_95: trọng số như Reference, τNormal=0.90, τAttack=0.95.

| Variant | AP tăng/giảm/giữ nguyên so source-only | Paired ΔAP | CI95 paired ΔAP |
| --- | --- | --- | --- |
| Reference | 3/1/1 | +0.017649 | [-0.021957; 0.057256] |
| sep_only | 2/2/1 | +0.000760 | [-0.101376; 0.102895] |
| sep003 | 1/2/2 | -0.012544 | [-0.073712; 0.048624] |
| sep010 | 4/1/0 | +0.040554 | [-0.039317; 0.120424] |
| tau90_95 | 1/2/2 | +0.001111 | [-0.077607; 0.079829] |

sep010 có AP trung bình cao nhất trong các cấu hình đã thử (0.203123), đồng thời std thấp hơn reference; FPR cao hơn source-only. Tắt MMD hoặc nâng riêng ngưỡng Attack lên 0.95 chưa cho thấy lợi ích AP trung bình so với reference. Không suy ra MMD luôn cần thiết hoặc separation càng mạnh càng tốt. Tất cả CI paired ΔAP so source-only đều chứa 0; đây là kết quả development sau nhiều thử nghiệm, không phải xác nhận độc lập trên target test.

Chi tiết theo seed và summaries: `results/development/class_aware_ablation_comparison_20261017.json`.
