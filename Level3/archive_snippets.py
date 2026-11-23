# Archived snippet: Threshold check in detect_and_aggregate to handle non-attacking rounds
# -----------------------------------------------------------------------------------------
# if hard_outliers:
#     hard_outliers.sort(key=lambda idx: combined_scores[idx], reverse=True)
#     flagged_client = hard_outliers[0]
# else:
#     max_idx = int(torch.tensor(combined_scores).argmax().item())
#     if combined_scores[max_idx] > 0.35:
#         flagged_client = max_idx
#     else:
#         flagged_client = -1  # No anomalous client detected this round
# 
# if flagged_client != -1:
#     print(f"  [Defense] Flagged malicious client: Client {flagged_client} (norm={norms[flagged_client]:.2f}, cos={cos_scores[flagged_client]:.4f})")
# else:
#     print(f"  [Defense] No abnormal updates detected. All clients appear benign.")
