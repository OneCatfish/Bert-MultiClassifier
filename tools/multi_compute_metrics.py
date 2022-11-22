from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import classification_report


def simple_accuracy(preds, labels):
    return (preds == labels)


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def acc_and_f1_classify(preds, labels):
    precision_macro = precision_score(labels, preds, average='macro')
    precision_micro = precision_score(labels, preds, average='micro')
    precision_weighted = precision_score(labels, preds, average='weighted')

    recall_macro = recall_score(labels, preds, average='macro')
    recall_micro = recall_score(labels, preds, average='micro')
    recall_weighted = recall_score(labels, preds, average='weighted')

    f1_macro = f1_score(y_true=labels, y_pred=preds, average='macro')
    f1_micro = f1_score(y_true=labels, y_pred=preds, average='micro')
    f1_weighted = f1_score(y_true=labels, y_pred=preds, average='weighted')

    report = classification_report(y_true=labels, y_pred=preds)
    report_dict = classification_report(y_true=labels, y_pred=preds, output_dict=True)

    return {
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'report': report,
        'report_dict': report_dict
    }


def compute_metrics(preds, labels):
    return {"acc": simple_accuracy(preds, labels),
            "acc_classify\n": acc_and_f1_classify(preds, labels)["report"],
            "acc_dict": acc_and_f1_classify(preds, labels)["report_dict"]}
