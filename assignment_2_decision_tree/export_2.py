# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Assignment 2: Decision Tree
#
#

# %%


# %% [markdown]
# We have implemented our own ID3 classifier. It's time to use it on the digit dataset

# %%
from sklearn import datasets, metrics
import matplotlib.pyplot as plt


# %%
digits = datasets.load_digits()
digits.data
digits.target_names

# %% [markdown]
# Prepare the data for training

# %%
import ID3
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=20
)
X_train.data.shape
X_train

# %% [markdown]
# Let's fit the data to our ID3 classifier to create a model and then plot it.

# %%
# import ToyData as td
# attributes, classes, data, target, data2, target2 = td.ToyData().get_data()

# id3_classifier_toy = ID3.ID3DecisionTreeClassifier()

# myTree = id3_classifier_toy.fit(data, target, attributes, classes, remaining_attributes = attributes)
# predictions_toy = id3_classifier_toy.predict(data2, myTree)


# %%
id3_classifier = ID3.ID3DecisionTreeClassifier()

attributes = dict()
for i in range(len(X_train[0])):
    attributes[i] = [float(j) for j in range(17)]

id3_tree = id3_classifier.fit(
    X_train, y_train, attributes, digits.target_names, remaining_attributes=attributes
)
predictions = id3_classifier.predict(X_test, id3_tree)


# %%
# plot = id3_classifier.make_dot_data()
# plot.render('digit_tree')


# %%
from sklearn import metrics
import seaborn as sn


def print_classification_report(classifier, classifier_name, true_label, predictions):
    print(
        "Classification report for classifier %s: \n %s\n"
        % (classifier, metrics.classification_report(true_label, predictions))
    )

    confusion_matrix = metrics.confusion_matrix(true_label, predictions)
    sn.set(font_scale=1.4)
    sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 16})
    plt.show()


print_classification_report(id3_classifier, "ID3 Toy", y_test, predictions)


# %%

