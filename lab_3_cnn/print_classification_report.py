from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt


def print_classification_report(classifier_name, true_label, predictions):
    print(
        "Classification report for classifier %s: \n %s\n"
        % (classifier_name, metrics.classification_report(true_label, predictions))
    )

    confusion_matrix = metrics.confusion_matrix(true_label, predictions)
    sn.set(font_scale=1.4)
    sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 16})
    plt.show()

