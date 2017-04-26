# Provides train/test split preserving percentage of samples for each class
from sklearn.model_selection import StratifiedKFold
# DecissionTreeClassifier algorythm
from sklearn import tree
from sklearn.metrics import confusion_matrix
from tools import ConfusionMatrixMetrics
import pydotplus
import numpy as np
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmax=sum(cm[0][:]))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')





def KFoldCrossValidation(df, report_folder, clf):

    # List with the different labels
    class_list = list(df["class"].drop_duplicates())
    # List with all the labels
    labels = list(df["class"].values)
    # List with the features
    features = []
    for j in range(df.shape[0]):
        item = df.ix[j]
        features.append([item[i] for i in range(len(item) - 1)])

    # Feature names list
    features_names_full = list(df.columns.values[:-1])

    # Create object to split the dataset (in 5 at random but preserving percentage of each class)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    # Split the dataset. The skf saves splits index
    skf.get_n_splits(features, labels)

    # Transform lists to np.arrays
    features = np.array(features)
    labels = np.array(labels)

    # Total predicted label kfold (Used for final confusion matrix)
    labels_kfold_predicted = []
    # Total labels kfold    (Used for final confusion matrix)
    labels_kfold = []
    # Accuracies for each kfold (Used for final accuracy and std)
    accuracies_kfold = []

    # Counter for the full report
    kcounter = 0

    # Report file with useful information
    report = open(report_folder + "report.txt", "w")

    # Iterate over the KFolds and do stuff
    for train_index, test_index in skf.split(features, labels):

        # Splits
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        # Train the classifier
        clf.fit(features_train, labels_train)
        accuracies_kfold.append(clf.score(features_test, labels_test))

        # Labels predicted for test split
        labels_pred_test = clf.predict(features_test)

        labels_kfold.extend(labels_test)
        labels_kfold_predicted.extend(labels_pred_test)

        kcounter += 1

    print(accuracies_kfold)
    print("\nMean accuracy: " + str(np.mean(accuracies_kfold)) + " +- " + str(np.std(accuracies_kfold)) + "\n")
    report.write("Accuracies: " + str(accuracies_kfold) + "\nMean accuracy: " + str(np.mean(accuracies_kfold)) + " +- " + str(
        np.std(accuracies_kfold)) + "\n")

    # Confusion matrix with all the predicted classes
    cm_kfold_total = confusion_matrix(labels_kfold, labels_kfold_predicted)



    # Get current size and making it bigger
    fig_size = plt.rcParams["figure.figsize"]

    # Set figure width to 12 and height to 9
    fig_size[0] = 14
    fig_size[1] = 14
    plt.rcParams["figure.figsize"] = fig_size


    plt.figure()
    plot_confusion_matrix(cm_kfold_total, class_list, False, "Full test Confusion")
    plt.savefig(report_folder + "cmkfolds.pdf")

    cmm = ConfusionMatrixMetrics(cm_kfold_total, class_list)
    report.write(cmm.report() + "\n\n")

    clf.fit(features, labels)

    return clf

def TreeKFoldReport(df, report_folder, clf):

    # List with the different labels
    class_list = list(df["class"].drop_duplicates())
    # List with all the labels
    labels = list(df["class"].values)
    # List with the features
    features = []
    for j in range(df.shape[0]):
        item = df.ix[j]
        features.append([item[i] for i in range(len(item) - 1)])

    # Feature names list
    features_names_full = list(df.columns.values[:-1])

    # Create object to split the dataset (in 5 at random but preserving percentage of each class)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    # Split the dataset. The skf saves splits index
    skf.get_n_splits(features, labels)

    # Transform lists to np.arrays
    features = np.array(features)
    labels = np.array(labels)

    # Total predicted label kfold (Used for final confusion matrix)
    labels_kfold_predicted = []
    # Total labels kfold    (Used for final confusion matrix)
    labels_kfold = []
    # Accuracies for each kfold (Used for final accuracy and std)
    accuracies_kfold = []

    # Counter for the full report
    kcounter = 0

    # Report file with useful information
    report = open(report_folder + "report.txt", "w")

    # Iterate over the KFolds and do stuff
    for train_index, test_index in skf.split(features, labels):
        report.write("KFold numero " + str(kcounter) + "\n")

        print("Train:", train_index, "Test:", test_index)
        report.write("\tTrain: " + str(train_index) + " Test:" + str(test_index) + "\n\n")
        # Splits
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        # Train the classifier
        clf.fit(features_train, labels_train)
        accuracies_kfold.append(clf.score(features_test, labels_test))
        print(accuracies_kfold[kcounter])
        report.write("\tAccuracy: " + str(accuracies_kfold[kcounter]) + "\n")

        # Confusion matrix for train and test
        labels_pred_test = clf.predict(features_test)
        labels_pred_train = clf.predict(features_train)

        cm_test = confusion_matrix(labels_test, labels_pred_test)
        cm_train = confusion_matrix(labels_train, labels_pred_train)

        cmm = ConfusionMatrixMetrics(cm_test, class_list)
        report.write("\t" + cmm.report() + "\n\n")

        """
        Ploting the test confusion for the test set
        """
        # Get current size
        fig_size = plt.rcParams["figure.figsize"]

        # Prints: [8.0, 6.0]
        print("Current size:", fig_size)
        # Set figure width to 12 and height to 9
        fig_size[0] = 12
        fig_size[1] = 12
        plt.rcParams["figure.figsize"] = fig_size

        plt.figure()

        plot_confusion_matrix(cm_test, class_list, False, "Test Confusion")
        plt.savefig(report_folder + "cmtest" + str(kcounter) + ".pdf")

        """
        Ploting the train confusion for the train set"""
        plt.figure()
        plot_confusion_matrix(cm_train, class_list, False, "Train Confusion")
        plt.savefig(report_folder + "cmtrain" + str(kcounter) + ".pdf")

        labels_kfold.extend(labels_test)
        labels_kfold_predicted.extend(labels_pred_test)

        kcounter += 1

    print(accuracies_kfold)
    print("\nMean accuracy: " + str(np.mean(accuracies_kfold)) + "+-" + str(np.std(accuracies_kfold)) + "\n")
    report.write(
        "Accuracies: " + str(accuracies_kfold) + "\nMean accuracy: " + str(np.mean(accuracies_kfold)) + "+-" + str(
            np.std(accuracies_kfold)) + "\n")
    cm_kfold_total = confusion_matrix(labels_kfold, labels_kfold_predicted)
    plt.figure()
    plot_confusion_matrix(cm_kfold_total, class_list, False, "Full test Confusion")
    plt.savefig(report_folder + "cmkfolds.pdf")

    cmm = ConfusionMatrixMetrics(cm_kfold_total, class_list)
    report.write(cmm.report() + "\n\n")

    clf.fit(features, labels)

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=features_names_full,
                                    class_names=class_list,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(report_folder + "FinalTree.pdf")

    importances = clf.feature_importances_
    X = features
    std = np.std([clf.feature_importances_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    # Get current size
    fig_size = plt.rcParams["figure.figsize"]

    # Prints: [8.0, 6.0]
    print("Current size:", fig_size)
    # Set figure width to 12 and height to 9
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()


def getClf(individual):
    clf = tree.DecisionTreeClassifier(criterion="gini",
                                      splitter="best",
                                      max_features=None,
                                      max_depth=8,
                                      min_samples_split=4,
                                      min_samples_leaf=10,
                                      min_weight_fraction_leaf=0,
                                      max_leaf_nodes=None,
                                      random_state=None,
                                      min_impurity_split=individual[0],
                                      presort=False)
    return clf

def KFoldAccuracy(df,clf):

    # List with the different labels
    class_list = list(df["class"].drop_duplicates())
    # List with all the labels
    labels = list(df["class"].values)
    # List with the features
    features = []
    for j in range(df.shape[0]):
        item = df.ix[j]
        features.append([item[i] for i in range(len(item) - 1)])

    # Feature names list
    features_names_full = list(df.columns.values[:-1])

    # Create object to split the dataset (in 5 at random but preserving percentage of each class)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    # Split the dataset. The skf saves splits index
    skf.get_n_splits(features, labels)

    # Transform lists to np.arrays
    features = np.array(features)
    labels = np.array(labels)

    # Total predicted label kfold (Used for final confusion matrix)
    labels_kfold_predicted = []
    # Total labels kfold    (Used for final confusion matrix)
    labels_kfold = []
    # Accuracies for each kfold (Used for final accuracy and std)
    accuracies_kfold = []

    # Counter for the full report
    kcounter = 0

    # Iterate over the KFolds and do stuff
    for train_index, test_index in skf.split(features, labels):
        # Splits
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        # Train the classifier
        clf.fit(features_train, labels_train)
        accuracies_kfold.append(clf.score(features_test, labels_test))

        # Labels predicted for test split
        labels_pred_test = clf.predict(features_test)

        labels_kfold.extend(labels_test)
        labels_kfold_predicted.extend(labels_pred_test)

        kcounter += 1

    meanAccuracy = np.mean(accuracies_kfold)
    std = np.std(accuracies_kfold)

    return meanAccuracy, std
