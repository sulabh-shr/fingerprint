import numpy as np
from typing import List
from sklearn import metrics
import matplotlib.pyplot as plt


def scores_to_metrics(y_true: List, y_score: List):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    fnr = 1 - tpr
    delta = np.absolute(fpr - fnr)

    eer_index = np.nanargmin(delta)
    eer_threshold = thresholds[eer_index]
    eer_fpr = fpr[eer_index]
    eer_fnr = fnr[eer_index]
    eer_delta = delta[eer_index]
    roc_auc = metrics.auc(fpr, tpr)

    ap = metrics.average_precision_score(y_true, y_score)

    y_pred_label = np.zeros_like(y_true)
    y_pred_label[y_score > eer_threshold] = 1
    accuracy = metrics.accuracy_score(y_true, y_pred_label)
    precision = metrics.precision_score(y_true, y_pred_label)
    recall = metrics.recall_score(y_true, y_pred_label)
    f1_score = metrics.f1_score(y_true, y_pred_label)

    P = np.sum(y_true)
    N = len(y_true) - P
    PP = np.sum(y_pred_label)
    PN = len(y_true) - PP

    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    # display = metrics.RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_score)
    display_result = display.plot(color='green', marker='x', markersize=6, alpha=0.5)
    fig = display_result.figure_
    # z = np.polyfit(fpr, tpr, deg=2)
    # f = np.poly1d(z)
    x = np.linspace(0.0, 1.0, 10)
    y = 1 - x
    # y = f(x)

    plt.plot(x, y, '--', color='orange', alpha=0.5)
    plt.xlim((0., 1.))
    plt.ylim((0., 1.))
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('Genuine Accept Rate (1 - FNMR)')

    plt_score = min(eer_fnr, eer_fpr)
    plt_score_name = 'EER'
    plt.scatter([plt_score], [1 - plt_score],
                s=200, marker='X', label=f'{plt_score_name} = {round(plt_score * 100, 2)}',
                c='red', alpha=1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))

    # plt.savefig('auc-curve.png')
    # plt.show()

    result = {
        'AUC': round(roc_auc * 100, 3),
        'EER': round(min(eer_fnr, eer_fpr) * 100, 3),
        'EER-delta': round(eer_delta * 100, 3),
        'EER threshold': round(eer_threshold, 3),
        'f1': round(f1_score * 100, 3),
        'Precision': round(precision * 100, 3),
        'Recall': round(recall * 100, 3),
        'Avg Precision': round(ap * 100, 3),
        'Accuracy': round(accuracy * 100, 3),
        'P': P,
        'N': N,
        'PP': PP,
        'PN': PN,
    }
    return result, fig
