"""
Day 6: The Shape of the Problem

This file contains:
1. Baseline model exercise - predicting "no breach" every time
2. Build exercise - 150-word analysis on why accuracy is misleading
3. Probe exercise - false positive vs false negative cost analysis
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ============================================================================
# EXERCISE 1: BUILD - BASELINE MODEL
# ============================================================================
# Build a model that predicts "no breach" every time and analyze its accuracy

print("=" * 70)
print("EXERCISE 1: BASELINE MODEL")
print("=" * 70)

# Load the incident dataset
df = pd.read_csv("incidents.csv")

# Baseline: Always predict "no breach"
df["prediction"] = False

# Calculate accuracy
accuracy = (df["prediction"] == df["sla_breached"]).mean()

print(f"\nDataset size: {len(df)} incidents")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy %: {accuracy * 100:.2f}%")

# Detailed metrics
actual = df["sla_breached"]
predicted = df["prediction"]

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(actual, predicted)
print(f"True Negatives (TN):  {cm[0,0]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"False Negatives (FN): {cm[1,0]}")
print(f"True Positives (TP):  {cm[1,1]}")

print("\n--- Classification Metrics ---")
print(f"Precision: {precision_score(actual, predicted):.4f}")
print(f"Recall: {recall_score(actual, predicted):.4f}")
print(f"F1 Score: {f1_score(actual, predicted):.4f}")

# ============================================================================
# EXERCISE 2: BUILD - 150-WORD ANALYSIS
# ============================================================================
# Why a 96%-accurate model can be entirely worthless
# Which metric should you present to a delivery manager?

print("\n" + "=" * 70)
print("EXERCISE 2: WHY ACCURACY IS MISLEADING")
print("=" * 70)

analysis = """
ANALYSIS: Why a 96%-Accurate Model Can Be Entirely Worthless

A baseline model that predicts "no breach" 100% of the time achieves 56.52%
accuracy on our dataset because approximately 56% of tickets don't breach SLA.
While this seems reasonable, accuracy alone is a dangerous metric for imbalanced
classification problems. The confusion matrix reveals the truth: this model
correctly identifies 28,260 non-breaches but MISSES ALL 21,740 actual SLA
breaches. Its recall is 0%, meaning it catches zero real breaches.

In production, this model is useless. A delivery manager doesn't care that the
model is "96% accurate" if it never alerts them to actual SLA breaches. Missed
breaches directly impact customer satisfaction, credit liability, and reputation
damage. The critical metric here is RECALL, which measures what percentage of
actual breaches we catch. Alternatively, we could use PRECISION to balance
false alarms against missed breaches, or PR-AUC to assess overall performance
across different thresholds.

To a delivery manager, I would present RECALL and PRECISION as primary metrics.
"This model catches 95% of real breaches but generates 10% false alarms" is far
more actionable than "96% accuracy."
"""

print(analysis)
print(f"Word count: {len(analysis.split())} words")

# ============================================================================
# EXERCISE 3: PROBE - FALSE POSITIVE vs FALSE NEGATIVE COST
# ============================================================================
# Which error is more expensive in operational terms?

print("\n" + "=" * 70)
print("EXERCISE 3: OPERATIONAL COST ANALYSIS")
print("=" * 70)

probe_answer = """
PROBE ANSWER: Which Error Is More Expensive?

For SLA breach prediction, a FALSE NEGATIVE (missed breach) is significantly
more expensive than a FALSE POSITIVE (false alarm).

COST OF FALSE NEGATIVE (Missed Breach):
- Actual breach occurs but model predicts no action needed
- Customer's SLA is violated, no proactive intervention happens
- Business consequences: Customer dissatisfaction, service credit/refund liability,
  potential contract termination, reputation damage, lost future revenue
- Operational cost: High (potential $1000s in credits + customer churn)

COST OF FALSE POSITIVE (False Alarm):
- Model alerts team to a potential breach that doesn't actually occur
- Operations team wastes time investigating a non-issue
- Operational cost: Lower (wasted investigation time, ~$50-200 in personnel cost)

FINANCIAL DEFENSE:
If a missed breach costs the company $1000-5000 in credits and reputation damage,
and a false alarm costs $100 in investigation time, the asymmetry is 10:1 to 50:1.
This means the business can tolerate many false positives to prevent even one false
negative. Therefore, an SLA breach prediction model should be tuned to MAXIMIZE RECALL,
even at the cost of lower precision. Better to alert the team about 10 potential
breaches and investigate 1 false alarm than to miss 1 actual breach.
"""

print(probe_answer)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key Learnings from Day 6:

1. Accuracy is misleading for imbalanced datasets
   - Our baseline has 56.52% accuracy but is completely useless
   - It catches 0% of actual breaches (recall = 0%)

2. Choose metrics based on business impact
   - For SLA breaches: Focus on RECALL and PRECISION
   - Never rely solely on accuracy for classification problems

3. False negatives are more expensive than false positives
   - Missed breach: $1000s in credits + reputation damage
   - False alarm: $100s in investigation time
   - Recommendation: Maximize recall, accept more false positives

4. Imbalanced classification requires careful metric selection
   - Use confusion matrix to understand model behavior
   - Use PR-AUC or ROC-AUC to evaluate performance fairly
   - Consider business costs when choosing thresholds
""")
