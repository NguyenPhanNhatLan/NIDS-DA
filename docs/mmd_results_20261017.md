# Tổng hợp kết quả MMD — 17/10/2026

Kết quả target-validation development, mean ± sample std (ddof=1) qua seeds 42–46. Không phải final test. Canonical và development là các cấu hình khác nhau, không gộp trung bình giữa cấu hình. Separation-only là ablation đối chứng không có MMD. Manifest/config hashes và kết quả được kiểm tra bằng summarizer hiện tại; không train hay chạy inference lại.

## unsw_to_cicids

| Cấu hình | AP | ROC-AUC | Macro-F1 | Recall | FPR | ΔAP so source-only | Seeds tăng AP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Source-only | 0.1626 ± 0.0544 | 0.3858 ± 0.1047 | 0.4684 ± 0.0423 | 0.1120 ± 0.1104 | 0.1572 ± 0.0622 | — | — |
| Class-aware + Sep .003 | 0.1500 ± 0.0312 | 0.3796 ± 0.1044 | 0.4472 ± 0.0793 | 0.1270 ± 0.0984 | 0.2410 ± 0.1261 | -0.0125 | 1/5 |
| Class-aware + Sep .010 | 0.2031 ± 0.0127 | 0.5275 ± 0.0239 | 0.5031 ± 0.0326 | 0.2273 ± 0.0929 | 0.2107 ± 0.1011 | +0.0406 | 4/5 |
| Separation only (λMMD=0) | 0.1633 ± 0.0426 | 0.4044 ± 0.0353 | 0.4704 ± 0.0369 | 0.1291 ± 0.1077 | 0.1704 ± 0.0963 | +0.0008 | 2/5 |
| Class-aware + Sep .001, τ=.90/.95 | 0.1637 ± 0.0498 | 0.4280 ± 0.1398 | 0.4654 ± 0.0447 | 0.1186 ± 0.1141 | 0.1721 ± 0.0621 | +0.0011 | 1/5 |
| Class-aware, τ=.90, λMMD=.001 | 0.1467 ± 0.0216 | 0.3860 ± 0.0503 | 0.4662 ± 0.0282 | 0.0881 ± 0.0473 | 0.1418 ± 0.0536 | -0.0159 | 2/5 |
| Class-aware + Sep .001 | 0.1802 ± 0.0591 | 0.4659 ± 0.1319 | 0.4838 ± 0.0391 | 0.1928 ± 0.2465 | 0.1780 ± 0.1421 | +0.0176 | 3/5 |
| Class-aware MK-MMD + Sep .010 | 0.1726 ± 0.0423 | 0.4337 ± 0.0662 | 0.4766 ± 0.0461 | 0.1275 ± 0.0875 | 0.1652 ± 0.0601 | +0.0100 | 3/5 |
| Class-aware MK-MMD + Sep .010, LR=.0003 | 0.1558 ± 0.0415 | 0.3530 ± 0.0624 | 0.4538 ± 0.0597 | 0.1016 ± 0.0811 | 0.1961 ± 0.0699 | -0.0067 | 2/5 |
| Class-aware MMD (canonical) | 0.1358 ± 0.0123 | 0.3612 ± 0.0568 | 0.4496 ± 0.0197 | 0.0509 ± 0.0335 | 0.1272 ± 0.0489 | -0.0267 | 1/5 |
| Marginal MK-MMD | 0.1924 ± 0.0618 | 0.4773 ± 0.1181 | 0.4981 ± 0.0518 | 0.2067 ± 0.1776 | 0.1836 ± 0.0810 | +0.0298 | 3/5 |
| Marginal MMD | 0.1397 ± 0.0284 | 0.3596 ± 0.0891 | 0.4466 ± 0.0408 | 0.0850 ± 0.1065 | 0.1756 ± 0.0725 | -0.0228 | 0/5 |

| Cấu hình | CI95 của paired ΔAP |
| --- | --- |
| Class-aware + Sep .003 | [-0.0737; +0.0486] |
| Class-aware + Sep .010 | [-0.0393; +0.1204] |
| Separation only (λMMD=0) | [-0.1014; +0.1029] |
| Class-aware + Sep .001, τ=.90/.95 | [-0.0776; +0.0798] |
| Class-aware, τ=.90, λMMD=.001 | [-0.0715; +0.0398] |
| Class-aware + Sep .001 | [-0.0220; +0.0573] |
| Class-aware MK-MMD + Sep .010 | [-0.0868; +0.1068] |
| Class-aware MK-MMD + Sep .010, LR=.0003 | [-0.0625; +0.0490] |
| Class-aware MMD (canonical) | [-0.0959; +0.0425] |
| Marginal MK-MMD | [-0.0123; +0.0719] |
| Marginal MMD | [-0.0862; +0.0406] |

## cicids_to_unsw

| Cấu hình | AP | ROC-AUC | Macro-F1 | Recall | FPR | ΔAP so source-only | Seeds tăng AP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Source-only | 0.2189 ± 0.0952 | 0.8030 ± 0.0737 | 0.4920 ± 0.0005 | 0.0012 ± 0.0005 | 0.0001 ± 0.0001 | — | — |
| Class-aware MMD (canonical) | 0.1815 ± 0.0881 | 0.7637 ± 0.0715 | 0.4980 ± 0.0140 | 0.0076 ± 0.0148 | 0.0002 ± 0.0003 | -0.0374 | 2/5 |
| Marginal MK-MMD | 0.0935 ± 0.0357 | 0.6318 ± 0.0431 | 0.4953 ± 0.0075 | 0.0046 ± 0.0077 | 0.0002 ± 0.0002 | -0.1254 | 0/5 |
| Marginal MMD | 0.1021 ± 0.0231 | 0.6116 ± 0.0679 | 0.4919 ± 0.0009 | 0.0011 ± 0.0009 | 0.0001 ± 0.0001 | -0.1168 | 0/5 |

| Cấu hình | CI95 của paired ΔAP |
| --- | --- |
| Class-aware MMD (canonical) | [-0.1719; +0.0971] |
| Marginal MK-MMD | [-0.2717; +0.0209] |
| Marginal MMD | [-0.2274; -0.0061] |

## Runs không gộp vào bảng đủ 5 seeds

- `proposal_class_aware_selective_single`, unsw_to_cicids: seeds [42]; retired config or incomplete five-seed matrix. Số liệu từng run: `[{"seed": 42, "pr_auc": 0.17184991659301152, "delta_ap": -0.01775475416324515}]`.

## Diễn giải

Xếp hạng AP chỉ là so sánh development sau nhiều thử nghiệm. CI Student-t qua năm paired seeds mô tả run variation trên bộ dữ liệu cố định, không phải dataset uncertainty hay xác nhận độc lập sau lựa chọn hyperparameters. Báo đồng thời FPR/recall; AP tăng không chứng minh pseudo-label đúng hoặc class mixing giảm.

Chi tiết JSON và CSV theo seed: `results/development/mmd_summary_20261017/`.
