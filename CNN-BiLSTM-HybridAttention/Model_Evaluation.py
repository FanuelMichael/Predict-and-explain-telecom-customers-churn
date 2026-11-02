import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve,
                             roc_auc_score, precision_score, recall_score, f1_score)

#=======================================================================================
def plot_class_distribution(before_counts, after_counts):
    # Calculate percentages
    before_total = sum(before_counts.values())
    after_total = sum(after_counts.values())
    before_percent = {k: v / before_total * 100 for k, v in before_counts.items()}
    after_percent = {k: v / after_total * 100 for k, v in after_counts.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    bar_colors = ['blue', 'red']
    # Before SMOTE resampling
    axes[0].bar(before_counts.keys(), before_counts.values(), color=bar_colors)
    axes[0].set_title("Class Distribution Before Resampling")
    axes[0].set_ylabel("Sample Count")
    for i, (label, count) in enumerate(before_counts.items()):
        pct = before_percent[label]
        axes[0].text(i, count + 50, f"{count} ({pct:.1f}%)", ha='center', fontsize=10)
    # After SMOTE resampling
    axes[1].bar(after_counts.keys(), after_counts.values(), color=bar_colors)
    axes[1].set_title("Class Distribution After SMOTE")
    for i, (label, count) in enumerate(after_counts.items()):
        pct = after_percent[label]
        axes[1].text(i, count + 50, f"{count} ({pct:.1f}%)", ha='center', fontsize=10)
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_evaluation(y_test, y_proba, y_pred):
    fig = plt.figure(figsize=(18, 14))
    ax_conf = plt.subplot(2, 2, 1)
    ax_roc = plt.subplot(2, 2, 2)
    # 1. Confusion matrix subplot
    conf_matrix = confusion_matrix(y_test, y_pred)
    if ax_conf is None:
        fig, ax_conf = plt.subplots(figsize=(4, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_conf)
    ax_conf.set_title('Confusion Matrix')
    ax_conf.set_xlabel('Predicted')
    ax_conf.set_ylabel('True')

    # 2. ROC-AUC graph subplot
    if ax_roc is None:
        fig2, ax_roc = plt.subplots(figsize=(4, 4))
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # CALCULATE OPTIMAL THRESHOLD
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

    # Mark the optimal point on the plot
    ax_roc.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Best Threshold: {optimal_threshold:.2f}')
    ax_roc.set_title('ROC Curve')
    ax_roc.set_xlabel('FPR')
    ax_roc.set_ylabel('TPR')
    ax_roc.legend(loc='lower right')
    metrics = detailed_metrics(y_test, y_pred, y_proba)

    # 3. Detailed Metrics table
    ax_table = plt.subplot(2, 1, 2)
    ax_table.axis('off')
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    report_df.index.name = "Class"
    report_df.reset_index(inplace=True)

    # Rename row headers: accuracy -> Accuracy, macro avg -> Macro avg, weighted avg -> Weighted avg
    class_col = report_df.columns[0]  # 'Class'
    report_df[class_col] = report_df[class_col].replace({
        'accuracy': 'Accuracy','macro avg': 'Macro avg','weighted avg': 'Weighted avg'})

    numeric_cols = ['precision', 'recall', 'f1-score', 'support']
    report_df[numeric_cols] = report_df[numeric_cols].round(4)

    table_data = report_df.values.tolist()
    # Bold column headers
    column_labels = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    table = ax_table.table(cellText=table_data, colLabels=column_labels, loc='center', cellLoc='center')
    table.scale(0.5, 1.6)
    # Apply bold formatting to column headers
    for col_idx, _ in enumerate(column_labels):
        cell = table[(0, col_idx)]
        cell.set_text_props(weight='bold')
    # Apply bold formatting to specific row headers (first column)
    target_rows = ['Accuracy', 'Macro avg', 'Weighted avg']
    # row 0 is header, data rows start at 1
    for row_idx in range(1, len(table_data) + 1):
        first_cell = table[(row_idx, 0)]
        if first_cell.get_text().get_text() in target_rows:
            first_cell.set_text_props(weight='bold')

    ax_table.set_title("Detailed Metrics", fontsize=14, y=0.68)
    plt.subplots_adjust(top=0.90, bottom=0.02, hspace=0.1)
    plt.close(fig)
    return fig

def detailed_metrics(y_test, y_pred, y_proba):
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = (y_pred == y_test).mean()
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    return {'classification_report': report,'accuracy': round(float(acc),4),'precision': round(float(prec),4),
        'recall': round(float(rec),4),'f1': round(float(f1),4),'roc_auc': round(float(roc_auc),4)}