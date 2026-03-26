import pandas as pd

OUTPUT_CSV = "seizure_rule_based_EVENT_LEVEL_TEST.csv"

# =============================
# EVALUATION
# =============================
def evaluate_results(df):
    """
    Computes segment-level accuracy, confidence, and weighted TestScore.
    """
    correct = df["true_label"] == df["predicted_label"]
    df["correct"] = correct.astype(int)

    # Average confidence on correct predictions
    avg_conf_correct = df.loc[df["correct"]==1, "confidence"].mean()
    
    # Segment-level accuracy
    accuracy = df["correct"].mean()
    
    # Weighted TestScore = sum(confidence of correct) / total segments
    test_score = (df["confidence"] * df["correct"]).sum() / len(df)
    
    print(f"Segment-level Accuracy: {accuracy:.3f}")
    print(f"Avg Confidence (correct predictions): {avg_conf_correct:.3f}")
    print(f"TestScore (confidence-weighted): {test_score:.3f}")
    
    return accuracy, avg_conf_correct, test_score

# Load your saved CSV
df_results = pd.read_csv(OUTPUT_CSV)
evaluate_results(df_results)


# =============================
# EVENT-LEVEL EVALUATION
# =============================
# def evaluate_events(df, min_overlap=1.0):
#     """
#     Evaluates seizure detection at the event level.
#     min_overlap: minimum seconds of overlap to count as TP
#     """
#     # Group predicted windows into events
#     predicted_events = []
#     current = []
#     for _, row in df.iterrows():
#         if row['predicted_label'] == 'SEIZURE':
#             if not current:
#                 current.append(row)
#             else:
#                 # Check gap
#                 gap = row['segment_start'] - current[-1]['segment_end']
#                 if gap <= 1.0:  # 1 sec gap allowed
#                     current.append(row)
#                 else:
#                     predicted_events.append(current)
#                     current = [row]
#         else:
#             if current:
#                 predicted_events.append(current)
#                 current = []
#     if current:
#         predicted_events.append(current)

#     # Convert predicted events to (start, end)
#     pred_intervals = [(ev[0]['segment_start'], ev[-1]['segment_end']) for ev in predicted_events]

#     # Extract ground-truth seizures
#     true_events = []
#     for edf_file in df['edf_file'].unique():
#         df_file = df[df['edf_file'] == edf_file]
#         gt_seizures = df_file[df_file['true_label'] != 'bckg']
#         if gt_seizures.empty:
#             continue
#         # Group consecutive seizure windows
#         current = []
#         for _, row in gt_seizures.iterrows():
#             if not current:
#                 current.append(row)
#             else:
#                 gap = row['segment_start'] - current[-1]['segment_end']
#                 if gap <= 1.0:
#                     current.append(row)
#                 else:
#                     true_events.append((current[0]['segment_start'], current[-1]['segment_end']))
#                     current = [row]
#         if current:
#             true_events.append((current[0]['segment_start'], current[-1]['segment_end']))

#     # Count TP, FP, FN
#     TP = 0
#     matched_pred = set()
#     matched_true = set()
#     for i, t in enumerate(true_events):
#         for j, p in enumerate(pred_intervals):
#             overlap = max(0, min(t[1], p[1]) - max(t[0], p[0]))
#             if overlap >= min_overlap:
#                 TP += 1
#                 matched_pred.add(j)
#                 matched_true.add(i)
#                 break

#     FP = len(pred_intervals) - len(matched_pred)
#     FN = len(true_events) - len(matched_true)

#     precision = TP / (TP + FP) if TP + FP > 0 else 0
#     recall = TP / (TP + FN) if TP + FN > 0 else 0
#     f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

#     print("Event-level Evaluation:")
#     print(f"True Positives (TP): {TP}")
#     print(f"False Positives (FP): {FP}")
#     print(f"False Negatives (FN): {FN}")
#     print(f"Precision: {precision:.3f}")
#     print(f"Recall: {recall:.3f}")
#     print(f"F1-score: {f1:.3f}")

#     return TP, FP, FN, precision, recall, f1


# # =============================
# # RUN EVENT-LEVEL EVAL
# # =============================
# df_results = pd.read_csv(OUTPUT_CSV)
# evaluate_events(df_results)
