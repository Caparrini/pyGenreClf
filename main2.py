from scipy.io import wavfile
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import essentia
import essentia.standard
from essentia.standard import *


def essentiaExtract(dirs):

    # Dictionaries to transform the tonality key in a integer in relation to the cycle of fifths
    # Mayor keys and relative minor have the same number
    # The number is cyclic module 12 and it is calculated (index)*7%12
    # This index began in 0 at (C for mayor, A for minor)
    mayorKeys = {
            "C" : 0,
            "C#": 7,
            "D" : 2,
            "D#": 9,
            "E" : 4,
            "F" : 11,
            "F#": 6,
            "G" : 1,
            "G#": 8,
            "A" : 3,
            "A#": 10,
            "B" : 5
    }
    minorKeys = {
            "A" : 0,
            "A#": 7,
            "B" : 2,
            "C" : 9,
            "C#": 4,
            "D" : 11,
            "D#": 6,
            "E" : 1,
            "F" : 8,
            "F#": 3,
            "G" : 10,
            "G#": 5
    }
    features = []
    labels = []
    for i in range(len(genre_list)):
        songs = [x for x in os.walk(dirs[i])][0][2]
        for song_name in songs:
            if ".wav" in song_name:
                audio_file = dirs[i] + song_name
                # Essentia loader for audio
                loader = essentia.standard.MonoLoader(filename=audio_file)
                # Audio loaded
                audio = loader()

                # How to extract key, mode, and stregth of the prediction

                key_extract = KeyExtractor()
                key_song, scale_song, strength = key_extract(audio)

                # How to extract BPM

                rhythm_extractor = RhythmExtractor()
                bpm, _, _, _ = rhythm_extractor(audio)

                # Transform key to the cycle of fifths
                if(scale_song=="mayor"):
                    key_song_int = mayorKeys[key_song]
                else:
                    key_song_int = minorKeys[key_song]

                features.append([bpm,key_song_int])
                labels.append(genre_list[i])

    df = pd.DataFrame.from_records(features)




    features_names_full = ["BPM", "KEY"]


    df.columns = features_names_full
    df["BPM"] = np.round(df["BPM"])
    df["class"] = labels
    return df

def create_fft(fn):
    sample_rate, X = wavfile.read(fn)
    fft_features = abs(scipy.fft(X)[:1000])
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fft"
    np.save(data_fn, fft_features)

def dataset_fft():
    os.chdir("genres/")
    genre_folders = [x for x in os.walk("./")]
    genre_folders.pop(0)
    for g in genre_folders:
        for s in g[2][1:]:
            ruta = g[0]+"/"+s
            print(ruta)
            create_fft(ruta)


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








from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt



def read_fft():
    features = []
    labels = []
    os.chdir("genres/")
    genre_folders = [x for x in os.walk("./")]
    genre_folders.pop(0)
    for g in genre_folders:
        for s in g[2][1:]:
            if s[-3:]=="npy":
                ruta = g[0]+"/"+s 
                features.append(np.load(ruta))
                labels.append(g[0][2:])
    return features, labels

def song_feature_extraction():
    features = []
    labels = []
    os.chdir("genres/")
    genre_folders = [x for x in os.walk("./")]
    genre_folders.pop(0)
    for g in genre_folders:
        for s in g[2]:
            if s[-3:]=="wav":
                ruta = g[0]+"/"+s 
                [Fs, x] = audioBasicIO.readAudioFile(ruta)
                AX = audioFeatureExtraction.stFeatureExtraction(x, Fs, Fs*0.05, Fs*0.05)
                BPM, ratio = audioFeatureExtraction.beatExtraction(AX, 0.05)
                features.append(BPM)
                print(features)
                labels.append(g[0][2:])
    return features, labels


#features,labels = read_fft()
#features, labels = song_feature_extraction()

#DATASET FORMADO CON PYAUDIPANALYSIS
#Chapuzeando
import pandas as pd
data = "data7.csv"
#files_folder = "/Users/Capa/Datasets/beatsdataset"
files_folder = "genres/"
genre_list = [x for x in os.walk(files_folder)][0][1]
#del genre_list[1]

if(os.path.exists(data)):
    df = pd.DataFrame.from_csv(data)
    labels = list(df["class"].values)
    #features = list(df[df.columns.difference(["class"])].values)
    features = []
    for j in range(df.shape[0]):
        item = df.ix[j]
        features.append([item[i] for i in range(len(item)-1)])


else:
    dirs = [files_folder + f + "/" for f in genre_list]
    [featuresTotal, labelsTotal, _] = audioFeatureExtraction.dirsWavFeatureExtraction(dirs, 1, 1, 0.05, 0.05, True)

    features = []
    labels = []
    [features.append(featuresTotal[i][ii]) for i in range(len(genre_list)) for ii in range(100)]
    [labels.append(labelsTotal[i/100]) for i in range(len(genre_list)*100)]
    df = pd.DataFrame.from_records(features)

    features_names = ["ZCR", "Energy", "EnergyEntropy", "SpectralCentroid",
                      "SpectralSpread", "SpectralEntropy", "SpectralFlux",
                      "SpectralRolloff", "MFCCs", "ChromaVector", "ChromaDeviation"]

    features_names_full = []
    features_metrics = ["m", "std"]

    for j in range(2):
        offset = j * 34;
        for i in range(1, 9):
            features_names_full.append(str(i + offset) + "-" + features_names[i - 1] + features_metrics[j])
        for i in range(9, 22):
            features_names_full.append(str(i + offset) + "-" + features_names[8] + str(i - 8) + features_metrics[j])
        for i in range(22, 34):
            features_names_full.append(str(i + offset) + "-" + features_names[9] + str(i - 21) + features_metrics[j])
        features_names_full.append(str(34 + offset) + "-" + features_names[10] + features_metrics[j])
    features_names_full.append("69-BPM")
    features_names_full.append("70-BPMpaa")

    df.columns = features_names_full

    dfs = essentiaExtract(dirs)
    df["71-KEY"] = dfs["KEY"]
    df["72-BPMessentia"] = dfs["BPM"]

    df["class"] = labels
    pd.DataFrame.to_csv(df,data)




#Partiendo el dataset de forma random

# Provides train/test split preserving percentage of samples for each class
from sklearn.model_selection import StratifiedKFold
# DecissionTreeClassifier algorythm
from sklearn import tree
from sklearn.metrics import confusion_matrix
from tools import ConfusionMatrixUtils
import pydotplus

# features_names_full = list(df.columns.values[[34,68]])
features_names_full = list(df.columns.values[:-1])

# Create object to split the dataset (in 5 at random but preserving percentage of each class)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# Split the dataset. The skf saves splits index
skf.get_n_splits(features, labels)


clf = tree.DecisionTreeClassifier(criterion = "gini",
                                  splitter = "best",
                                  max_features = None,
                                  max_depth = None,
                                  min_samples_split = 40,
                                  min_samples_leaf = 10,
                                  min_weight_fraction_leaf = 0,
                                  max_leaf_nodes = None,
                                  random_state = None,
                                  min_impurity_split = 0.5,
                                  presort = False)
"""
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=37,
                              criterion="gini",
                              max_depth=None,
                              min_samples_split=8,
                              min_samples_leaf=1,
                              min_weight_fraction_leaf=0.0,
                              max_features=None,
                              max_leaf_nodes=None,
                              min_impurity_split=1e-07,
                              bootstrap=True,
                              oob_score=False,
                              n_jobs=-1,
                              random_state=None,
                              verbose=1,
                              warm_start=False,
                              class_weight=None)
"""
features = np.array(features)
labels = np.array(labels)

# Total predicted label kfold
labels_kfold_predicted = []
# Total labels kfold
labels_kfold = []
# Accuracies for each kfold
accuracies_kfold = []

# Counter for the full report
kcounter = 0
#
rfile = "report/"
report = open(rfile+"report.txt", "w")

# Iterate over the KFolds and do stuff
for train_index, test_index in skf.split(features, labels):
    report.write("KFold numero " + str(kcounter) + "\n")

    print("Train:", train_index, "Test:", test_index)
    report.write("\tTrain: " + str(train_index) + " Test:" + str(test_index)+"\n\n")
    # Splits
    features_train, features_test = features[train_index], features[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]
    # Train the classifier
    clf.fit(features_train, labels_train)
    accuracies_kfold.append(clf.score(features_test, labels_test))
    print(accuracies_kfold[kcounter])
    report.write("\tAccuracy: " + str(accuracies_kfold[kcounter])+"\n")


    # Confusion matrix for train and test
    labels_pred_test = clf.predict(features_test)
    labels_pred_train = clf.predict(features_train)

    cm_test = confusion_matrix(labels_test, labels_pred_test)
    cm_train = confusion_matrix(labels_train, labels_pred_train)

    cmm = ConfusionMatrixUtils(cm_test, genre_list)
    report.write("\t"+cmm.report()+"\n\n")

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

    plot_confusion_matrix(cm_test,genre_list,False,"Test Confusion")
    plt.savefig(rfile+"cmtest"+str(kcounter)+".pdf")

    """
    Ploting the train confusion for the train set"""
    plt.figure()
    plot_confusion_matrix(cm_train,genre_list,False,"Train Confusion")
    plt.savefig(rfile+"cmtrain"+str(kcounter)+".pdf")

    labels_kfold.extend(labels_test)
    labels_kfold_predicted.extend(labels_pred_test)


    kcounter+=1

"""

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=features_names_full,
                                    class_names=genre_list,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(rfile+"kfoldtree"+str(kcounter)+".pdf")
"""




print(accuracies_kfold)
print("\nMean accuracy: " + str(np.mean(accuracies_kfold)) +"+-" + str(np.std(accuracies_kfold))+"\n")
report.write("Accuracies: " + str(accuracies_kfold) + "\nMean accuracy: " + str(np.mean(accuracies_kfold)) +"+-" + str(np.std(accuracies_kfold))+"\n")
cm_kfold_total = confusion_matrix(labels_kfold, labels_kfold_predicted)
plt.figure()
plot_confusion_matrix(cm_kfold_total, genre_list, False, "Full test Confusion")
plt.savefig(rfile+"cmkfolds.pdf")

cmm = ConfusionMatrixUtils(cm_kfold_total, genre_list)
report.write(cmm.report() + "\n\n")

clf = tree.DecisionTreeClassifier(criterion = "gini", min_samples_split=10)
clf.fit(features, labels)

#print(clf.feature_importances_)


"""
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(n_estimators=10,
                              criterion="gini",
                              max_depth=None,
                              min_samples_split=2,
                              min_samples_leaf=1,
                              min_weight_fraction_leaf=0.0,
                              max_features='auto',
                              max_leaf_nodes=None,
                              min_impurity_split=1e-07,
                              bootstrap=True,
                              oob_score=False,
                              n_jobs=1,
                              random_state=None,
                              verbose=1,
                              warm_start=False,
                              class_weight=None)
clf2.fit(features_train, labels_train)
print(clf2.score(features_test,labels_test))

from sklearn.ensemble import ExtraTreesClassifier
clf3 = ExtraTreesClassifier(n_estimators=10)
clf3.fit(features_train, labels_train)
print(clf3.score(features_test,labels_test))
"""



dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=features_names_full,
                                class_names =genre_list,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf(rfile+"FinalTree.pdf")


importances = clf.feature_importances_
X = features
std = np.std([clf.feature_importances_ ],
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



