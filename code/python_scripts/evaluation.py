from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def get_acc(y_test, y_pred):
    return round(accuracy_score(y_test, y_pred), 3)


def get_pre(y_test, y_pred):
    return round(precision_score(y_test, y_pred), 3)


def get_rec(y_test, y_pred):
    return round(recall_score(y_test, y_pred), 3)


def get_f1(y_test, y_pred):
    return round(f1_score(y_test, y_pred), 3)


def print_res(y_test, y_pred):
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", get_acc(y_test, y_pred))
    print("Precision Score:", get_pre(y_test, y_pred))
    print("Recall Score:", get_rec(y_test, y_pred))
    print("F1 Score:", get_f1(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
