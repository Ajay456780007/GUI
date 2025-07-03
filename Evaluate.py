from sklearn.metrics import mean_squared_error, multilabel_confusion_matrix, accuracy_score, precision_score, \
    recall_score, f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np

import math
import warnings

from tensorflow.python.ops.metrics_impl import recall

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def tt(value):
    power = math.ceil(math.log10(value) - 1)
    A1 = 100 ** (math.log10(value) - power)
    return A1


def main_est_parameters(y_true, pred):
    """
    :param y_true: true labels
    :param pred: predicted labels
    :return: performance metrics in list dtype
    """
    cm = multilabel_confusion_matrix(y_true, pred)
    cm = sum(cm)
    TP = cm[0, 0]  # True Positive
    FP = cm[0, 1]  # False Positive
    FN = cm[1, 0]  # False Negative
    TN = cm[1, 1]  # True Negative
    Acc = (TP + TN) / (TP + TN + FP + FN)
    Sen = TP / (TP + FN)
    Spe = TN / (TN + FP)
    Pre = TP / (TP + FP)
    Rec = TP / (TP + FN)
    F1score = 2 * (Pre * Rec) / (Pre + Rec)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return [Acc, Sen, Spe, F1score,Rec,Pre,TPR,FPR]


def main_est_parameters_mul(y_true, pred):
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, average='macro', zero_division=0)
    rec = recall_score(y_true, pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    with np.errstate(divide='ignore', invalid='ignore'):
        specificity = TN / (TN + FP)
        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)

    spe = np.nanmean(specificity)
    fpr_macro = np.nanmean(fpr)
    tpr_macro = np.nanmean(tpr)

    return [acc, tpr_macro, spe, f1, rec, prec, tpr_macro, fpr_macro]



def Evaluation_Metrics1(y, y_pred):
    mse = tt(mean_squared_error(y, y_pred))
    rmse = np.sqrt(mse)
    mae = tt(mean_absolute_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    cor = np.sqrt(r2)
    return [mse, rmse, mae, r2, cor]

import numpy as np
from sklearn.metrics import confusion_matrix

def compute_metrics(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)

    num_classes = cm.shape[0]
    sensitivity_list = []
    specificity_list = []
    F1_score_list = []
    Pre_list = []
    Rec_list = []
    FPR_list=[]

    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        Pre = TP / (TP + FP) if (TP + FP) > 0 else 0
        Rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1_score = 2 * (Pre * Rec) / (Pre + Rec) if (Pre + Rec) > 0 else 0
        FPR=FP/FP+TN if (FP+TN)>0 else 0

        F1_score_list.append(F1_score)
        Pre_list.append(Pre)
        Rec_list.append(Rec)
        sensitivity_list.append(sensitivity)
        FPR_list.append(FPR)
        specificity_list.append(specificity)

    # Accuracy: total correct predictions / total predictions
    accuracy = np.trace(cm) / np.sum(cm)


    avg_precision = np.mean(Pre_list)
    avg_recall = np.mean(Rec_list)
    avg_f1_score = np.mean(F1_score_list)
    avg_sensitivity = np.mean(sensitivity_list)
    avg_specificity = np.mean(specificity_list)
    TPR = avg_recall
    avg_FPR= np.mean(FPR_list)
    return [accuracy, avg_sensitivity, avg_specificity, avg_f1_score, avg_recall, avg_precision,TPR,avg_FPR]

def compute_metrics2(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)

    num_classes = cm.shape[0]
    sensitivity_list = []
    specificity_list = []
    F1_score_list = []
    Pre_list = []
    Rec_list = []
    FPR_list=[]

    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        Pre = TP / (TP + FP) if (TP + FP) > 0 else 0
        Rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1_score = 2 * (Pre * Rec) / (Pre + Rec) if (Pre + Rec) > 0 else 0
        FPR=FP/(FP+TN) if (FP+TN)>0 else 0

        F1_score_list.append(F1_score)
        Pre_list.append(Pre)
        Rec_list.append(Rec)
        sensitivity_list.append(sensitivity)
        FPR_list.append(FPR)
        specificity_list.append(specificity)

    # Accuracy: total correct predictions / total predictions
    accuracy = np.trace(cm) / np.sum(cm)
    accuracy =accuracy

    avg_precision = np.mean(Pre_list)
    avg_precision = avg_precision
    avg_recall = np.mean(Rec_list)
    avg_recall = avg_recall
    avg_f1_score = np.mean(F1_score_list)
    avg_f1_score = avg_f1_score
    avg_sensitivity = np.mean(sensitivity_list)
    avg_sensitivity = avg_sensitivity
    avg_specificity = np.mean(specificity_list)
    avg_specificity = avg_specificity
    TPR = avg_recall
    avg_FPR= np.mean(FPR_list)


    return [accuracy, avg_sensitivity, avg_specificity, avg_f1_score, avg_recall, avg_precision,TPR,avg_FPR]