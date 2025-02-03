#include "evaluation.h"

ConfusionMatrix compute_confusion_matrix(const int *y_true, const int *y_pred, int n_samples) {
    ConfusionMatrix cm = {0};
    for (int i = 0; i < n_samples; i++) {
        if (y_true[i] == 1 && y_pred[i] == 1) cm.tp++;
        else if (y_true[i] == 0 && y_pred[i] == 1) cm.fp++;
        else if (y_true[i] == 0 && y_pred[i] == 0) cm.tn++;
        else if (y_true[i] == 1 && y_pred[i] == 0) cm.fn++;
    }
    return cm;
}

double accuracy(const ConfusionMatrix *cm) {
    int total = cm->tp + cm->tn + cm->fp + cm->fn;
    return (double)(cm->tp + cm->tn) / total;
}

double precision(const ConfusionMatrix *cm) {
    if (cm->tp + cm->fp == 0) return 0.0; // Avoid division by zero
    return (double)cm->tp / (cm->tp + cm->fp);
}

double recall(const ConfusionMatrix *cm) {
    if (cm->tp + cm->fn == 0) return 0.0;
    return (double)cm->tp / (cm->tp + cm->fn);
}

double f1_score(const ConfusionMatrix *cm) {
    double p = precision(cm);
    double r = recall(cm);
    if (p + r == 0.0) return 0.0;
    return 2 * (p * r) / (p + r);
}

double prediction_error(const ConfusionMatrix *cm) {
    return 1.0 - accuracy(cm);
}