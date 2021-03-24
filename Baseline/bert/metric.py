import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def flat_accuracy(logits, labels):
    logits = logits.detach().cpu().numpy()
    labels = labels.cpu().numpy()
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


def flat_f1(logits, labels):
    logits = logits.detach().cpu().numpy()
    labels = labels.cpu().numpy()
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = labels.flatten()

    return f1_score(labels_flat, pred_flat)