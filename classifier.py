from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from tools import ConfusionMatrixUtils
import pydotplus
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import joblib
import librosa
import logging
from featuresExtraction import extractFeatures
try:
    from xgboost import XGBClassifier
except ImportError:
    print("xgboost not installed!")

# Returns the best classifiers for faster experiments
def bestClfs():
    '''
    This method return a list of the best classifiers used in the beatsdataset.csv

    :return list: List of classifiers
    '''

    DTC23 = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=15,
            min_samples_split=61, min_weight_fraction_leaf=0,
            presort=False, random_state=None, splitter='best')
    #   ----> Accuracy: 0.553043478261 +- 0.0141287624428
    RFC23 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=0.497907908371,
            max_leaf_nodes=None, min_impurity_split=1e-07,
            min_samples_leaf=2, min_samples_split=2,
            min_weight_fraction_leaf=0, n_estimators=150, n_jobs=4,
            oob_score=True, random_state=None, verbose=0, warm_start=False)

    DTC7 = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=9,
            min_samples_split=40, min_weight_fraction_leaf=0,
            presort=False, random_state=None, splitter='best')
    #   ----> Accuracy: 0.553043478261 +- 0.0141287624428
    RFC7 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=0.59,
            max_leaf_nodes=None, min_impurity_split=1e-07,
            min_samples_leaf=2, min_samples_split=15,
            min_weight_fraction_leaf=0, n_estimators=84, n_jobs=4,
            oob_score=True, random_state=None, verbose=0, warm_start=False)

    ET7 = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
           max_depth=None, max_features=0.790926623187,
           max_leaf_nodes=None, min_impurity_split=1e-07,
           min_samples_leaf=1, min_samples_split=16,
           min_weight_fraction_leaf=0, n_estimators=135, n_jobs=4,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

    GB7 = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.0150834277809, loss='deviance',
              max_depth=None, max_features=0.982060609531,
              max_leaf_nodes=None, min_impurity_split=1e-07,
              min_samples_leaf=22, min_samples_split=51,
              min_weight_fraction_leaf=0, n_estimators=135, presort='auto',
              random_state=None, subsample=0.769360696352, verbose=0,
              warm_start=False)

    #1 0.548 +-0.015 with beatsdataset.csv (windows and steps 1 1 0.05 0.05) SIN ESSENTIA BPM 0.47
    #2 0.492 +- 0.015 with beatsdataset1-1-01-005.csv
    #3 0.486 +- 0.015 with beatsdataset1-1-01-01.csv
    #4 0.424 +- 0.023 with beatsdataset1-1-04-04.csv
    #5 0.4383 +- 0.0103 with beatsdataset1-05-005-0025.csv
    #6 0.463 +- 0.032 with beatsdataset138-stStep25.csv
    #7 0.493 +- 0.011 with beatsdataset138-stStep50.csv  CON ESSENTIA BPM 0.56 +- 0.007
    #10 0.694 +- 0.044 with gtzan.csv


    #ETC = ExtraTreesClassifier(bootstrap=True, criterion="gini",max_features=1, min_samples_leaf=2,min_samples_split=10, n_estimators=100)

    # Accuracy 138 step 50 with BPM essentia (0.56260869565217386, 0.012251306785743798)
    #ETC = ExtraTreesClassifier(bootstrap=False, criterion="gini",max_features=0.5, min_samples_leaf=2,min_samples_split=10, n_estimators=100)

    # Best with GTZAN
    #1 0.534 +- 0.01 with beatsdataset.csv
    #2 0.46 +- 0.01 with beatsdataset1-1-01-005.csv
    #3 0.48 +- 0.014 with beatsdataset1-1-01-01.csv
    #4 0.422 +- 0.019 with beatsdataset1-1-04-04.csv
    #5 0.4387 +- 0.0182 with beatsdataset1-05-005-0025.csv
    #6 0.452 +- 0.0198 with beatsdataset138-stStep25.csv
    #7 0.486 +- 0.024 with beatsdataset138-stStep50.csv
    #10 0.731 +- 0.021 with gtzan.csv

    #GBC = GradientBoostingClassifier(learning_rate=0.1, max_depth=6,max_features=0.5, min_samples_leaf=13,min_samples_split=6, subsample=0.8)

    #1 0.556 +-0.016 with beatsdataset.csv SIN ESSENTIA BPM 0.48
    #2 0.477 +- 0.012 with beatsdataset1-1-01-005.csv
    #3 0.477 +- 0.007 with beatsdataset1-1-01-01.csv
    #4 0.451 +- 0.007 with beatsdataset1-1-04-04.csv
    #5 0.443 +- 0.019 with beatsdataset1-05-005-0025.csv
    #6 0.479 +- 0.011 with beatsdataset138-stStep25.csv
    #7 0.5 +- 0.02 with beatsdataset138-stStep50.csv CON ESSENTIA BPM 0.557, 0.017
    #10 0.722 +- 0.012 with gtzan.csv

    #XGB = XGBClassifier(learning_rate=0.1, max_depth=5,min_child_weight=6, nthread=4,subsample=0.55)

    clfs = [DTC23, RFC23, DTC7, RFC7, ET7, GB7]
    return clfs

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    '''
    This function plots a confusion matrix

    :param numpy.array cm: Confusion matrix
    :param list classes: List of classes
    :param boolean normalize: True to normalize
    :param str title: Title of the plot
    :param cmap: Colours
    '''
    classes = sorted(classes)
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmax=sum(cm[0][:]))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    if normalize:
        cm = np.round(100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).astype('int')
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=16)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def KFoldCrossValidation(df, report_folder, clf, random_state=None):
    '''
    Generates a report using KFold cross validation.
    It generate train/test confusion matrix for each kfold, a final kfold with all the test splits
    and a report.txt with metrics and other data.

    :param pandas.DataFrame df: DataFrame with the dataset
    :param str report_folder: folder where save pics and report
    :param clf: classifier with methods fit, score and predict sklearn styled
    :return: clf trained with all the data
    '''

    class_list, features, labels = unpackDF(df)

    # Create object to split the dataset (in 5 at random but preserving percentage of each class)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
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
    if (os.path.isdir(report_folder)):

        logging.warning("The directory %s already exist", report_folder)

    else:

        logging.info("Creating directory %s", report_folder)

        os.mkdir(report_folder, 0o0755)


    report = open(os.path.join(report_folder,"report.txt"), "w")

    codes = []

    # Iterate over the KFolds and do stuff
    for train_index, test_index in skf.split(features, labels):

        # Splits
        features_train, features_test = features[train_index], features[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        # Train the classifier with 80% of samples
        clf.fit(features_train, labels_train)
        # And predict with the other 20%
        accuracies_kfold.append(clf.score(features_test, labels_test))

        # Labels predicted for test split
        labels_pred_test = clf.predict(features_test)

        labels_kfold.extend(labels_test)
        labels_kfold_predicted.extend(labels_pred_test)
        codes.extend(features_test[:,71])

        kcounter += 1

    print(accuracies_kfold)
    print("\nMean accuracy: " + str(np.mean(accuracies_kfold)) + " +- " + str(np.std(accuracies_kfold)) + "\n")
    report.write("Accuracies: " + str(accuracies_kfold) + "\nMean accuracy: " + str(np.mean(accuracies_kfold)) + " +- " + str(
        np.std(accuracies_kfold)) + "\n")

    # Confusion matrix with all the predicted classes
    cm_kfold_total = confusion_matrix(labels_kfold, labels_kfold_predicted)

    # Get current size and making it bigger
    fig_size = plt.rcParams["figure.figsize"]

    # Set figure according with the number of classes
    size = len(class_list) - len(class_list)*30/100
    fig_size[0] = size
    fig_size[1] = size
    plt.rcParams["figure.figsize"] = fig_size


    plt.figure()
    plot_confusion_matrix(cm_kfold_total, class_list, False, "Full test Confusion")
    plt.savefig(os.path.join(report_folder,"cmkfolds.pdf"))

    cmm = ConfusionMatrixUtils(cm_kfold_total, class_list)
    report.write(cmm.report() + "\n\n")

    joblib.dump(cmm,os.path.join(report_folder,"cmm"))
    joblib.dump(cmm.cmmToGraph(),os.path.join(report_folder,"cmgraph"))

    clf.fit(features, labels)

    return clf, labels_kfold_predicted, codes

def TreeKFoldReport(df, report_folder, clf, n_splits=10, random_state=None):
    '''
    Uses KFold cross validation over the dataset generating info in the report folder.

    :param df: pandas.DataFrame with the dataset
    :param report_folder: folder to save pics and report
    :param clf: DecissionTreeClassifier
    :param int n_splits: Number of kfolds
    :param float random:state: Random state seed
    :return: clf full trained with the whole dataset
    '''

    class_list, features, labels = unpackDF(df)

    # Feature names list
    features_names_full = list(df.columns.values[:-1])

    # Create object to split the dataset (in 5 at random but preserving percentage of each class)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
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
    report = open(os.path.join(report_folder, "report.txt"), "w")

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

        cmm = ConfusionMatrixUtils(cm_test, class_list)
        report.write("\t" + cmm.report() + "\n\n")

        """
        Ploting the test confusion for the test set
        """
        # Get current size and making it bigger
        fig_size = plt.rcParams["figure.figsize"]

        # Set figure according with the number of classes
        size = len(class_list) - len(class_list) * 30 / 100
        fig_size[0] = size
        fig_size[1] = size
        plt.rcParams["figure.figsize"] = fig_size

        plt.figure()

        plot_confusion_matrix(cm_test, class_list, False, "Test Confusion")
        plt.savefig(os.path.join(report_folder,"cmtest" + str(kcounter) + ".pdf"))

        """
        Ploting the train confusion for the train set"""
        plt.figure()
        plot_confusion_matrix(cm_train, class_list, False, "Train Confusion")
        plt.savefig(os.path.join(report_folder,"cmtrain" + str(kcounter) + ".pdf"))

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
    plt.savefig(os.path.join(report_folder,"cmkfolds.pdf"))

    cmm = ConfusionMatrixUtils(cm_kfold_total, class_list)
    report.write(cmm.report() + "\n\n")

    clf.fit(features, labels)

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=features_names_full,
                                    class_names=class_list,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(os.path.join(report_folder,"FinalTree.pdf"))

    return clf

def plot_feature_importances(tree_classifier, feat_names, nfeat=10,  dimy=6, dimx=8,):
    '''
    Plots the nfeat more important features of the tree or random forest given.

    :param tree_classifier: classifier DecissionTree or RandomForest
    :param feat_names: The name of the features in the tree
    :param nfeat: The number of top features to show
    :param dimx: fig size x
    :param dimy: fig size y
    '''
    importances = tree_classifier.feature_importances_
    std = np.std([importances], axis=0) #Does nothing
    indices = importances.argsort()[-nfeat:][::-1]

    print("Feature ranking:")
    for f in range(nfeat):
        print("%d. feature %d (%f)" % (f+1, indices[f], importances[indices[f]]))

    plt.figure()
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = dimx
    fig_size[1] = dimy
    plt.rc('ytick', labelsize=16)
    plt.rcParams["figure.figsize"] = fig_size
    plt.title("Feature importances")
    plt.bar(range(nfeat), importances[indices],
            color="b", yerr=std[indices], align="center")
    plt.xticks(range(nfeat), feat_names[indices], rotation=75, fontsize=12)
    plt.xlim([-1, nfeat])
    plt.show()

def unpackDF(df):
    '''
    Extract classes, features, and labels from a pandas.DataFrame.
    One column of the DataFrame should be called "class" and
    the rest are features.

    :param DataFrame df: pandas.DataFrame with the dataset
    :return: Classes, features, labels
    '''
    # List with the different labels
    class_list = list(df["class"].drop_duplicates())
    # List with all the labels (X)
    labels = list(df["class"].values)
    # List with the features (y)
    df = df.drop(["class"],axis=1).reset_index(drop=True)
    features = []
    for j in range(df.shape[0]):
        item = df.ix[j]
        features.append([item[i] for i in range(len(item))])

    return class_list, features, labels

def KFoldAccuracy(df, clf, n_splits=10, random_state=None):
    '''
    Computes KFold cross validation accuracy using n_splits folds over the data in the pandas.DataFrame given.
    Uses an stratified KFold with the random_state specified.

    :param df: pandas.DataFrame where is the data for train/test splits
    :param clf: classifier with methods fit, predict and score
    :param n_splits: number of splits
    :param random_state: random state seed
    :return: mean accuracy, std
    '''

    _, features, labels = unpackDF(df)

    # Create object to split the dataset (in 5 at random but preserving percentage of each class)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
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

def predictGenre(song_file_name, clf_pkl=os.path.join(os.path.dirname(__file__),'Examples','beats23classifier.pkl')):
    '''
    Receives an audio file route and a binary classifier and returns the genre of the song in a string

    :param str song_file_name: audio file route
    :param str clf_pkl: binary classifier route
    :return: genre of the song using the classifier given or the default beatport classifier
    '''
    clf = joblib.load(clf_pkl)
    x, Fs = librosa.load(song_file_name)
    x = librosa.resample(x, Fs, 22050)
    x = librosa.to_mono(x)
    feats = extractFeatures(22050, x[:22050 * 120], 1, 1, 0.05, 0.05)
    return clf.predict([feats])[0]