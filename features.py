import pandas as pd
from scipy.io import wavfile
import scipy
import numpy as np
import os
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioBasicIO
import sys
import time


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

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def pyAudioFeatures70(dataset_csv="beatsdataset.csv", dataset_folder="/Users/Capa/Datasets/beatsdataset"):

    genre_list = [x for x in os.walk(dataset_folder)][0][1]

    if(not os.path.exists(dataset_folder)):
        print("The dataset folder : " + dataset_folder + " does not exist.\n")

    if(os.path.exists(dataset_csv)):

        print("The dataset_csv file already exists.\n")
        if(not query_yes_no("Do you want to overwrite it?")):
            return

    dirs = [dataset_folder + f + "/" for f in genre_list]
    [featuresTotal, labelsTotal, _] = audioFeatureExtraction.dirsWavFeatureExtraction(dirs, 1, 1, 0.05, 0.05, True)

    features = []
    labels = []

    for i in range(len(genre_list)):
        for ii in range(len(featuresTotal[i])):
            features.append(featuresTotal[i][ii])
            labels.append(labelsTotal[i])

    df = pd.DataFrame.from_records(features)

    features_names = ["ZCR", "Energy", "EnergyEntropy", "SpectralCentroid",
                          "SpectralSpread", "SpectralEntropy", "SpectralFlux",
                          "SpectralRolloff", "MFCCs", "ChromaVector", "ChromaDeviation"]

    features_names_full = []
    features_metrics = ["m", "std"]

    for j in range(2):
        offset = j * 34
        for i in range(1, 9):
            features_names_full.append(str(i + offset) + "-" + features_names[i - 1] + features_metrics[j])
        for i in range(9, 22):
            features_names_full.append(str(i + offset) + "-" + features_names[8] + str(i - 8) + features_metrics[j])
        for i in range(22, 34):
            features_names_full.append(str(i + offset) + "-" + features_names[9] + str(i - 21) + features_metrics[j])
        features_names_full.append(str(34 + offset) + "-" + features_names[10] + features_metrics[j])
    features_names_full.append("69-BPM")
    features_names_full.append("70-BPMconf")

    df.columns = features_names_full

    df["class"] = labels
    pd.DataFrame.to_csv(df,dataset_csv)

    return df

def pyAudioFeatures(dataset_csv="beatsdataset.csv", dataset_folder="/Users/Capa/Datasets/beatsdataset"):

    genre_list = [x for x in os.walk(dataset_folder)][0][1]

    if(not os.path.exists(dataset_folder)):
        print("The dataset folder : " + dataset_folder + " does not exist.\n")
        return

    if(os.path.exists(dataset_csv)):

        print("The dataset_csv file already exists.\n")
        if(not query_yes_no("Do you want to overwrite it?")):
            return

    dirs = [dataset_folder + f + "/" for f in genre_list]
    [featuresTotal, labelsTotal, _] = audioFeatureExtraction.dirsWavFeatureExtraction(dirs, 1, 1, 0.05, 0.025, True)

    features = []
    labels = []

    for i in range(len(genre_list)):
        for ii in range(len(featuresTotal[i])):
            features.append(featuresTotal[i][ii])
            labels.append(labelsTotal[i])

    df = pd.DataFrame.from_records(features)

    features_names = ["ZCR", "Energy", "EnergyEntropy", "SpectralCentroid",
                          "SpectralSpread", "SpectralEntropy", "SpectralFlux",
                          "SpectralRolloff", "MFCCs", "ChromaVector", "ChromaDeviation"]

    features_names_full = []
    features_metrics = ["m", "std", "skew", "kurt"]

    for j in range(len(features_metrics)):
        offset = j * 34
        for i in range(1, 9):
            features_names_full.append(str(i + offset) + "-" + features_names[i - 1] + features_metrics[j])
        for i in range(9, 22):
            features_names_full.append(str(i + offset) + "-" + features_names[8] + str(i - 8) + features_metrics[j])
        for i in range(22, 34):
            features_names_full.append(str(i + offset) + "-" + features_names[9] + str(i - 21) + features_metrics[j])
        features_names_full.append(str(34 + offset) + "-" + features_names[10] + features_metrics[j])
    features_names_full.append(str(len(features_metrics)*34+1)+"-BPM")
    features_names_full.append(str(len(features_metrics)*34+2)+"-BPMconf")

    df.columns = features_names_full

    df["class"] = labels
    pd.DataFrame.to_csv(df,dataset_csv)

    return df

# Extract features from one single audio with x = samples and Fs = Frequency rate
def extractFeatures(Fs, x, mtWin, mtStep, stWin, stStep):
    t1 = time.clock()

    [MidTermFeatures, stFeatures] = audioFeatureExtraction.mtFeatureExtraction(x, Fs, round(mtWin*Fs), round(mtStep*Fs), round(Fs*stWin), round(Fs * stStep))
    [beat, beatConf] = audioFeatureExtraction.beatExtraction(stFeatures, stStep)

    MidTermFeatures = np.transpose(MidTermFeatures)
    MidTermFeatures = MidTermFeatures.mean(axis=0) # long term averaging of mid-term statistics

    MidTermFeatures = np.append(MidTermFeatures, beat)
    MidTermFeatures = np.append(MidTermFeatures, beatConf)

    t2 = time.clock()
    print("Processing time : " + str(t2-t1))
    return MidTermFeatures
