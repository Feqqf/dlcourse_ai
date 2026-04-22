import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    # Приводим к типу int для удобства подсчёта (True->1, False->0)
    pred = prediction
    truth = ground_truth
    
    # Вычисляем компоненты confusion matrix
    TP = np.sum((pred == True) & (truth == True))
    TN = np.sum((pred == False) & (truth == False))
    FP = np.sum((pred == True) & (truth == False))
    FN = np.sum((pred == False) & (truth == True))
    
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    total = TP + TN + FP + FN
    if total == 0:
        accuracy = 0.0
    else:
        accuracy = (TP + TN) / total
    
    # Precision = TP / (TP + FP)
    if (TP + FP) == 0:
        precision = 0.0
    else:
        precision = TP / (TP + FP)
    
    # Recall = TP / (TP + FN)
    if (TP + FN) == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)
    
    # F1 = 2 * precision * recall / (precision + recall)
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    # Приводим к float для единообразия (на всякий случай)
    return float(precision), float(recall), float(f1), float(accuracy)


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 0
