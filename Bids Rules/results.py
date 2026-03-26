import pandas as pd

# Load CSV
df = pd.read_csv("seizure_rule_based_EVENT_LEVEL_TEST.csv")

# Define seizure labels (TUH)
SEIZURE_LABELS = {
    "cpsz", "tcsz", "fnsz", "gnsz",
    "absz", "mysz", "atsz"
}

def is_correct(row):
    true_label = row["true_label"].lower()
    pred_label = row["predicted_label"].lower()

    # Case 1: Correct seizure detection
    if pred_label == "seizure" and true_label in SEIZURE_LABELS:
        return True

    # Case 2: Correct non-seizure detection
    if pred_label == "bckg" and true_label == "bckg":
        return True

    return False


# Apply correctness logic
df["correct"] = df.apply(is_correct, axis=1)

# Count
correct = df["correct"].sum()
wrong = len(df) - correct
total = len(df)

# Percentages
correct_pct = correct / total * 100
wrong_pct = wrong / total * 100

# Print results
print(f"Total segments: {total}")
print(f"Correct predictions: {correct} ({correct_pct:.2f}%)")
print(f"Wrong predictions: {wrong} ({wrong_pct:.2f}%)")
print(f"Ratio of correct to wrong: {correct}:{wrong}")



# import pandas as pd

# # =============================
# # LOAD RESULTS
# # =============================
# CSV_PATH = "seizure_rule_based_EVENT_LEVEL1.csv"  # change if needed
# df = pd.read_csv(CSV_PATH)

# total = len(df)

# # =============================
# # BASIC COUNTS
# # =============================
# correct = (df['predicted_label'] == df['true_label']).sum()
# wrong = total - correct

# accuracy = correct / total

# # =============================
# # SEIZURE-ONLY METRICS
# # =============================
# seizure_df = df[df['true_label'] != 'bckg']
# bckg_df = df[df['true_label'] == 'bckg']

# # Sensitivity (Recall)
# tp = (seizure_df['predicted_label'] != 'bckg').sum()
# fn = (seizure_df['predicted_label'] == 'bckg').sum()
# sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

# # False Positive Rate (background)
# fp = (bckg_df['predicted_label'] != 'bckg').sum()
# fpr = fp / len(bckg_df) if len(bckg_df) > 0 else 0

# # =============================
# # CONFIDENCE ANALYSIS
# # =============================
# avg_conf_correct = df[df['predicted_label'] == df['true_label']]['confidence'].mean()
# avg_conf_wrong = df[df['predicted_label'] != df['true_label']]['confidence'].mean()

# confidence_weighted_score = (
#     df.apply(
#         lambda r: r['confidence'] if r['predicted_label'] == r['true_label'] else 0,
#         axis=1
#     ).sum() / total
# )

# # =============================
# # PRINT RESULTS
# # =============================
# print("\n===== RULE-BASED EEG EVALUATION =====")
# print(f"Total segments            : {total}")
# print(f"Correct predictions       : {correct}")
# print(f"Wrong predictions         : {wrong}")
# print(f"Segment accuracy          : {accuracy*100:.2f}%")

# print("\n--- Seizure performance ---")
# print(f"Seizure sensitivity (TPR) : {sensitivity*100:.2f}%")
# print(f"Background FPR            : {fpr*100:.2f}%")

# print("\n--- Confidence analysis ---")
# print(f"Avg confidence (correct)  : {avg_conf_correct:.3f}")
# print(f"Avg confidence (wrong)    : {avg_conf_wrong:.3f}")
# print(f"Confidence-weighted score : {confidence_weighted_score:.3f}")
