#Package Imports

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from fairlearn.metrics import MetricFrame, make_derived_metric
from fairlearn.metrics import selection_rate, count, false_positive_rate, false_negative_rate
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# We start by defining the model evaluation metrics

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plots the confusion matrix for a binary classification model.

    Arguments:
    y_true (array-like): True labels
    y_pred (array-like): Predicted labels
    labels (list): Class labels for the confusion matrix, 0 and 1 by default
    Returns:
    None: It prints and saves the confusion matrix of the predictions
    """
    cm = confusion_matrix(y_true, y_pred)

    if labels is None:
        labels = ["0", "1"]

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix of Model Predictions")
    plt.savefig("confusion_matrix.png")

def print_classification_report(y_true, y_pred):
    """
    Prints the classification report for a binary classification model.

    Parameters:
    y_true (array): True labels
    y_pred (array): Predicted labels
    Returns:
    None: it prints the classification report
    """
    report = classification_report(y_true, y_pred)
    print("Classification Report:\n", report)


def print_false_rates(y_true, y_pred):
    """
    Prints the false negative rate and false positive rate for a binary classification model.

    Parameters:
    y_true (array-like): True labels
    y_pred (array-like): Predicted labels
    Returns:
    None: Prints the misclassification rates
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate

    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")

# We continue by introducing the Group Fairness metrics

# compute metrics by group
def compute_selection_rate(y_true, y_pred, sensitive_feature):
    """
    Computes selection rate, demographic parity difference, and overall selection rate using fairlearn.

    Parameters:
    y_true (array): True labels
    y_pred (array): Predicted labels
    sensitive_feature (array): Sensitive group feature (e.g., race, gender)
    returns:
    None: prints the Group fairness metrics
    """
    mf = MetricFrame(metrics={'Selection Rate': selection_rate},
                     y_true=y_true,
                     y_pred=y_pred,
                     sensitive_features=sensitive_feature)

    print("Fairness Metrics by Group:")
    print(mf.by_group)
    print("Overall Selection Rate: %.2f" % mf.overall[0])
    print("demographic parity diff: %.2f" % mf.difference(method='between_groups')[0])


def compute_equalized_odds(y_true, y_pred, sensitive_feature):
    """
    Computes false positive and false negative rates by group, their differences, and equalized odds difference.

    Parameters:
    y_true (array): True labels
    y_pred (array): Predicted labels
    sensitive_feature (array): Sensitive group feature (e.g., race, gender)
    returns:
    None: prints the Group fairness metrics
    """
    mf = MetricFrame(metrics={
        'False Positive Rate': false_positive_rate,
        'False Negative Rate': false_negative_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature)
    print("False Rates by Group:")
    print(mf.by_group)

    # Summarize differences
    for metric, diff in mf.difference('between_groups').items():
        print("%s diff: %.2f" % (metric, diff))

    # Compute equalized odds difference
    eq_odds_diff = equalized_odds_difference(y_true=y_true,
                                             y_pred=y_pred,
                                             sensitive_features=sensitive_feature,
                                             method='between_groups')
    print("Equalized Odds Difference: %.2f" % eq_odds_diff)


# Example usage
y_true = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
y_pred = [0, 1, 0, 0, 0, 0, 1, 1, 0, 0]
# plot_confusion_matrix(y_true, y_pred)
# print_classification_report(y_true, y_pred)
# print_false_rates(y_true, y_pred)
