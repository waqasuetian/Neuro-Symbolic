import pandas as pd

# ================= LOAD CSV =================
df = pd.read_csv("train1_SVM_RULE_COMBINED_continuous_test_bids.csv")

# ================= TUH SEIZURE LABELS =================
SEIZURE_LABELS = {"cpsz", "tcsz", "fnsz", "gnsz", "absz", "mysz", "atsz"}

def is_seizure(label):
    return str(label).lower() in SEIZURE_LABELS

def is_bckg(label):
    return str(label).lower() == "bckg"

# ================= TRUE vs SVM =================
def correct_svm(row):
    svm_pred = str(row["svm_pred"]).lower()
    true_label = str(row["true_label"]).lower()
    if row["svm_pred"].lower() == "seizure" and is_seizure(row["true_label"]):
        return 1
    if row["svm_pred"].lower() == "non-seizure" and is_bckg(row["true_label"]):
        return 1
    return 0

df["svm_correct"] = df.apply(correct_svm, axis=1)

# ================= TRUE vs RULE =================
def correct_rule(row):
    if row["rule_pred"].lower() == "seizure" and is_seizure(row["true_label"]):
        return 1
    if row["rule_pred"].lower() == "bckg" and is_bckg(row["true_label"]):
        return 1
    return 0

df["rule_correct"] = df.apply(correct_rule, axis=1)

# ================= SVM vs RULE AGREEMENT =================
def svm_rule_agree(row):
    svm = row["svm_pred"].lower()
    rule = row["rule_pred"].lower()
    if svm == "seizure" and rule == "seizure":
        return 1
    if svm == "non-seizure" and rule == "bckg":
        return 1
    return 0

df["svm_rule_agree"] = df.apply(svm_rule_agree, axis=1)

# ================= HYBRID (TRUST) =================
ALPHA = 0.3  # Weight for SVM, 1-ALPHA for Rule

def hybrid_correct(row, alpha=ALPHA):
    trust_score = alpha * row["svm_prob"] + (1-alpha) * row["rule_conf"]
    hybrid_pred = "Seizure" if trust_score >= 0.5 else "Non-Seizure"
    true_label = str(row["true_label"]).lower()

    correct = 0
    if hybrid_pred.lower() == "seizure" and is_seizure(true_label):
        correct = 1
    elif hybrid_pred.lower() == "non-seizure" and is_bckg(true_label):
        correct = 1

    return pd.Series([correct, trust_score])

df[["hybrid_correct", "trust_score"]] = df.apply(hybrid_correct, axis=1)

# ================= TEST SCORE =================
total = len(df)

rule_testscore = (df["rule_conf"] * df["rule_correct"]).sum() / total
hybrid_testscore = (df["trust_score"] * df["hybrid_correct"]).sum() / total

# ================= SUMMARY =================
def print_stats(name, correct_col):
    correct = df[correct_col].sum()
    wrong = total - correct
    print(f"\n========= {name} =========")
    print(f"Total segments: {total}")
    print(f"Correct segments: {correct} ({correct/total*100:.2f}%)")
    print(f"Wrong segments  : {wrong} ({wrong/total*100:.2f}%)")
    print(f"Ratio correct:wrong = {correct}:{wrong}")

print_stats("SVM", "svm_correct")
print_stats("Rule-Based", "rule_correct")
print_stats("Hybrid (SVM+Rule)", "hybrid_correct")

print(f"\nSVM vs Rule Agreement: {df['svm_rule_agree'].sum()} / {total} ({df['svm_rule_agree'].sum()/total*100:.2f}%)")
print(f"\nRule-Based TestScore : {rule_testscore:.4f}")
print(f"Hybrid TestScore     : {hybrid_testscore:.4f}")

# ================= SAVE =================
df.to_csv("evaluation_hybrid_results.csv", index=False)
print("\n Evaluation results saved to evaluation_hybrid_results.csv")
import pandas as pd

# ================= LOAD CSV =================
df = pd.read_csv("train1_SVM_RULE_COMBINED_continuous_test_bids.csv")

# ================= PARAMETERS =================
ALPHA = 0.3  # weight for neural model

# ================= TUH SEIZURE LABELS =================
SEIZURE_LABELS = {"cpsz", "tcsz", "fnsz", "gnsz", "absz", "mysz", "atsz"}

def is_seizure(label):
    return str(label).lower() in SEIZURE_LABELS

def is_bckg(label):
    return str(label).lower() == "bckg"

# ================= STEP 1: HYBRID CONFIDENCE =================
# Confidence_i = alpha * svm_prob + (1-alpha) * rule_conf
df["confidence_i"] = ALPHA * df["svm_prob"] + (1 - ALPHA) * df["rule_conf"]

# ================= STEP 2: FINAL DECISION =================
# if Confidence_i >= 0.5 → seizure
def final_pred(conf):
    return "seizure" if conf >= 0.5 else "bckg"

df["hybrid_pred"] = df["confidence_i"].apply(final_pred)

# ================= STEP 3: CORRECTNESS INDICATOR =================
def hybrid_correct(row):
    pred = row["hybrid_pred"]
    true = str(row["true_label"]).lower()

    if pred == "seizure" and is_seizure(true):
        return 1
    if pred == "bckg" and is_bckg(true):
        return 1
    return 0

df["hybrid_correct"] = df.apply(hybrid_correct, axis=1)

# ================= STEP 4: TEST SCORE =================
# TestScore = (1/N) * sum(Confidence_i * correct_i)

N = len(df)
test_score = (df["confidence_i"] * df["hybrid_correct"]).sum() / N

# ================= BASIC ACCURACY =================
accuracy = df["hybrid_correct"].mean()

# ================= RESULTS =================
print("\n===== HYBRID EVALUATION =====")
print(f"Total segments: {N}")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Confidence-weighted TestScore: {test_score:.4f}")

# ================= SAVE =================
df.to_csv("evaluation_hybrid_results.csv", index=False)
print("\nSaved: evaluation_hybrid_results.csv")
