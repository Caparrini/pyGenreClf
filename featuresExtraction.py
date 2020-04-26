import pandas as pd
from scipy.io import wavfile
import scipy
import numpy as np
import os
import glob
from pyAudioAnalysis import audioFeatureExtraction
import sys
import time
from librosa.beat import beat_track
import librosa
try:
    import essentia
    import essentia.standard as es
except ImportError:
    print("Essentia not installed!")


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

def pyAudioFeatures(dataset_csv="CSV/beatsdataset.csv", dataset_folder="/Users/Capa/Datasets/beatsdataset/",
                    mtWin=1, mtStep=1, stWin=0.05, stStep=0.05):
    """
    This method extracts the pyAudioAnalysis features from all the dataset given and it saves them in a .csv file using
    pandas.

    :param str dataset_csv: File to write the dataset extraction of features
    :param str dataset_folder: Folder containing the dataset, with one folder for each class
    :param float mtWin: Size of the mid term analysis window
    :param float mtStep: Size of the step of the analysis mid term window
    :param float stWin: Size of the short term analysis window
    :param float stStep: Size of the step of the analysis short term window
    :return pandas.DataFrame: DataFrame object with features and labels of the dataset
    """
    genre_list = [x for x in os.walk(dataset_folder)][0][1] # The class folder names inside the dataset folder

    if(not os.path.exists(dataset_folder)): # Error if the folder given does not exist
        print("The dataset folder : " + dataset_folder + " does not exist.\n")
        return

    if(os.path.exists(dataset_csv)):    # If the .csv file exist, ask if the user wants to overwrite it

        print("The dataset_csv file already exists.\n")
        if(not query_yes_no("Do you want to overwrite it?")):
            return

    dirs = [os.path.join(dataset_folder,f) for f in genre_list]   # The class folder full address
    [featuresTotal, labelsTotal, _] = audioFeatureExtraction.dirsWavFeatureExtraction(dirs, mtWin, mtStep, stWin, stStep, True) # Using pyAudioAnalysis library

    features = [] # Features list
    labels = [] # Labels list

    for i in range(len(genre_list)): # Filling features and labels with the result of audioFeatureExtraction.dirsWavFeatureExtraction()
        for ii in range(len(featuresTotal[i])):
            features.append(featuresTotal[i][ii])
            labels.append(labelsTotal[i])

    df = pd.DataFrame.from_records(features) # Generate DataFrame with the features

    features_names = ["ZCR", "Energy", "EnergyEntropy", "SpectralCentroid",
                          "SpectralSpread", "SpectralEntropy", "SpectralFlux",
                          "SpectralRolloff", "MFCCs", "ChromaVector", "ChromaDeviation"]

    features_names_full = []

    if(df.shape[1]==70):    # pyAudioAnalysis gets 70 features
        features_metrics = ["m", "std"]
    else:                   # if using pyAudioAnalysis modified which gives skews and kurts
        features_metrics = ["m", "std", "skew", "kurt"]

    for j in range(len(features_metrics)): # Generate the names for the features
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

    df.columns = features_names_full # Added the features names to the DataFrame object

    df["class"] = labels    # Finally added the class column
    pd.DataFrame.to_csv(df,dataset_csv) # Export the DataFrame to a .csv file for easy import later

    return df # And return the DataFrame

def dirsExtractBPM(dataset_folder):
    '''
    Folder with sub-folder to extract all the BPMs

    :param str dataset_folder: Folder with folder of audio files
    :return list: List ob BPMs of all the songs in the sub-folders
    '''
    BPMs = [] # features
    genre_list = [x for x in os.walk(dataset_folder)][0][1]
    dirs = [os.path.join(dataset_folder,f) for f in genre_list]
    for d in dirs:
        BPMs.extend(dirExtractBPM(d))

    return BPMs

def dirExtractBPM(folder):
    '''
    Generate a list of BPM of each song in the given folder

    :param str folder: Folder with audio file to extract BPMs
    :return list: List of BPMs of every song in the folder
    '''
    BPMs = []

    types = ('*.wav', '*.aif', '*.mp3', '*.au', '*.aiff', '*.flac')
    audioFiles = []
    for files in types:
        audioFiles.extend(glob.glob(os.path.join(folder, files)))
    for audioFile in audioFiles:
        BPMs.append(fileExtractBPM(audioFile))
    return BPMs

def fileExtractBPM(fileRoute):
    '''
    It gets the BPM from an audio file

    :param str fileRoute: Audio file to extract BPM
    :return float: BPM of the audio file
    '''
    x, Fs = librosa.load(fileRoute)
    x = librosa.resample(x, Fs, 22050)
    x = librosa.to_mono(x)
    return extractBPM(x)

def extractBPM(x):
    '''
    Extract the BPM from the list of samples of an audio file.
    It tries to use essentia, if it is not installed then it used librosa.

    :param list x: List of samples from an audio file
    :return float: BPM of the samples of the audio file
    '''
    try:
        rhythm_extractor = RhythmExtractor()
        bpm, _, _, _ = rhythm_extractor(x)
    except NameError:
        # Essentia is not in the system, using librosa instead
        bpm, _ = beat_track(x)
    return round(bpm)

def extractFeatures(Fs, x, mtWin, mtStep, stWin, stStep):
    '''
    Extract 71 feature of a singe audio file where x are the sample and Fs the frequency rate.

    :param Fs: Frequency rate of the audio file
    :param x: List of samples
    :param float mtWin: Mid-Term analysis window
    :param float mtStep: Mid-Term step
    :param float stWin: Short-Term analysis window
    :param float stStep: Short-Term step
    :return list: List of 71 features
    '''
    t1 = time.clock()

    [MidTermFeatures, stFeatures, _] = audioFeatureExtraction.mtFeatureExtraction(x, Fs, round(mtWin*Fs), round(mtStep*Fs), round(Fs*stWin), round(Fs * stStep))
    [beat, beatConf] = audioFeatureExtraction.beatExtraction(stFeatures, stStep)

    MidTermFeatures = np.transpose(MidTermFeatures)
    MidTermFeatures = MidTermFeatures.mean(axis=0) # long term averaging of mid-term statistics

    MidTermFeatures = np.append(MidTermFeatures, beat)
    MidTermFeatures = np.append(MidTermFeatures, beatConf)
    #MidTermFeatures = np.append(MidTermFeatures, extractBPM(x))

    t2 = time.clock()
    print("Processing time : " + str(t2-t1))
    return MidTermFeatures

def extractFeaturesFolder(dataset_csv="CSV/beatsdataset.csv", dataset_folder="/Users/Capa/Datasets/beatsdataset/",
                    mtWin=1, mtStep=1, stWin=0.05, stStep=0.05):
    '''
    MAIN METHOD.
    Extracts 71 audio features from every audio file in the dataset_folder.
    Write the result in the dataset_csv file which is a pandas.DataFrame.
    The dataset_folder must have audio files classified in sub-folders and
    the name of the class is the name of the sub-folder which contains them.

    :param str dataset_csv: File to write the pandas.DataFrame with features and labels
    :param str dataset_folder: Folder with the audio samples to extract features from them
    :param float mtWin: Size of mid term analysis windows
    :param float mtStep: Step of mid term analysis windows
    :param float stWin: Size of short term analysis windows
    :param float stStep: Step of short term analysis windows
    :return: DataFrame with the classes and 71 features for each audio file
    '''
    if(not os.path.exists(dataset_folder)): # Error if the folder given does not exist
        print("The dataset folder : " + dataset_folder + " does not exist.\n")
        return

    if(os.path.exists(dataset_csv)):    # If the .csv file exist, ask if the user wants to overwrite it

        print("The dataset_csv file already exists.\n")
        if(not query_yes_no("Do you want to overwrite it?")):
            return

    df = pyAudioFeatures("pyaa-"+dataset_csv, dataset_folder, mtWin, mtStep, stWin, stStep)
    bpms = dirsExtractBPM(dataset_folder)
    df["71-BPM"] = bpms
    #swap
    columns = list(df.columns)
    n = len(columns)-1
    aux = columns[n]
    columns[n]=columns[n-1]
    columns[n-1] = aux
    df = df[columns]
    pd.DataFrame.to_csv(df,dataset_csv) # Export the DataFrame to a .csv file for easy import later
    return df

def file_extract_essentia(audio_file):
    rhythm_feats = ["rhythm.bpm",
                    "rhythm.bpm_histogram_first_peak_bpm",
                    "rhythm.bpm_histogram_first_peak_weight",
                    "rhythm.bpm_histogram_second_peak_bpm",
                    "rhythm.bpm_histogram_second_peak_spread",
                    "rhythm.bpm_histogram_second_peak_weight",
                    "rhythm.danceability",
                    "rhythm.beats_loudness.mean",
                    "rhythm.beats_loudness.stdev",
                    "rhythm.onset_rate",
                    "rhythm.beats_loudness_band_ratio.mean",
                    "rhythm.beats_loudness_band_ratio.stdev"
                    ]

    features_total, features_frames = es.MusicExtractor(endTime=120)(audio_file)

    features = []
    for i in range(0,len(rhythm_feats)-2):
        features.append(features_total[rhythm_feats[i]])

    for i in range(len(rhythm_feats)-2,len(rhythm_feats)):
        bands = features_total[rhythm_feats[i]]
        for j in range(0,len(bands)):
            features.append(bands[j])

    x, Fs = librosa.load(audio_file)
    x = librosa.resample(x, Fs, 22050)
    x = librosa.to_mono(x)
    max_len = 22050*120
    if(len(x)>max_len):
        x = x[:max_len]
    py_feats = extractFeatures(22050, x, 1, 1, 0.05, 0.05)

    py_feats = list(np.append(py_feats, np.array(features)))


    return py_feats

def essentia_from_folder(dataset_csv="beatsdataset_essentia.csv", dataset_folder="/Users/Capa/Datasets/beatsdataset/"):
    if(not os.path.exists(dataset_folder)): # Error if the folder given does not exist
        print("The dataset folder : " + dataset_folder + " does not exist.\n")
        return

    if(os.path.exists(dataset_csv)):    # If the .csv file exist, ask if the user wants to overwrite it

        print("The dataset_csv file already exists.\n")
        if(not query_yes_no("Do you want to overwrite it?")):
            return
    features = dirs_extract_essentia(dataset_folder)

    df = pd.DataFrame.from_records(features)  # Generate DataFrame with the features

    features_names_essentia = ["bpm",
                    "bpm_histogram_first_peak_bpm",
                    "bpm_histogram_first_peak_weight",
                    "bpm_histogram_second_peak_bpm",
                    "bpm_histogram_second_peak_spread",
                    "bpm_histogram_second_peak_weight",
                    "danceability",
                    "beats_loudness.mean",
                    "beats_loudness.stdev",
                    "onset_rate",
                    "beats_loudness_band_ratio.mean1","beats_loudness_band_ratio.mean2",
                    "beats_loudness_band_ratio.mean3","beats_loudness_band_ratio.mean4",
                    "beats_loudness_band_ratio.mean5","beats_loudness_band_ratio.mean6",
                    "beats_loudness_band_ratio.stdev1","beats_loudness_band_ratio.stdev2",
                    "beats_loudness_band_ratio.stdev3","beats_loudness_band_ratio.stdev4",
                    "beats_loudness_band_ratio.stdev5","beats_loudness_band_ratio.stdev6",
                    "class","id"
                    ]

    features_names = ["ZCR", "Energy", "EnergyEntropy", "SpectralCentroid",
                      "SpectralSpread", "SpectralEntropy", "SpectralFlux",
                      "SpectralRolloff", "MFCCs", "ChromaVector", "ChromaDeviation"]

    features_names_full = []

    features_metrics = ["m", "std"]

    for j in range(len(features_metrics)):  # Generate the names for the features
        offset = j * 34
        for i in range(1, 9):
            features_names_full.append(str(i + offset) + "-" + features_names[i - 1] + features_metrics[j])
        for i in range(9, 22):
            features_names_full.append(str(i + offset) + "-" + features_names[8] + str(i - 8) + features_metrics[j])
        for i in range(22, 34):
            features_names_full.append(str(i + offset) + "-" + features_names[9] + str(i - 21) + features_metrics[j])
        features_names_full.append(str(34 + offset) + "-" + features_names[10] + features_metrics[j])
    features_names_full.append(str(len(features_metrics) * 34 + 1) + "-BPM")
    features_names_full.append(str(len(features_metrics) * 34 + 2) + "-BPMconf")

    features_names_full.extend(features_names_essentia)


    df.columns = features_names_full  # Added the features names to the DataFrame object

    pd.DataFrame.to_csv(df, dataset_csv)  # Export the DataFrame to a .csv file for easy import later

    return df  # And return the DataFrame


def dirs_extract_essentia(dataset_folder):
    '''
    Folder with sub-folder to extract all the essentia features

    :param str dataset_folder: Folder with folder of audio files
    :return list: List ob features of all the songs in the sub-folders
    '''
    features = [] # features
    genre_list = [x for x in os.walk(dataset_folder)][0][1]
    dirs = [os.path.join(dataset_folder,f) for f in genre_list]
    for d in dirs:
        features.extend(dir_extract_essentia(d))

    return features

def dir_extract_essentia(folder):
    '''
    Generate a list of essentia features of each song in the given folder

    :param str folder: Folder with audio files to extract features
    :return list: List of features of every song in the folder
    '''
    features = []

    types = ('*.wav', '*.aif', '*.mp3', '*.au', '*.aiff', '*.flac')
    audioFiles = []
    genre = os.path.basename(folder)
    for files in types:
        audioFiles.extend(glob.glob(os.path.join(folder, files)))
    for audioFile in audioFiles:
        feats = file_extract_essentia(audioFile)
        feats.append(genre)
        feats.append(os.path.basename(audioFile))
        features.append(feats)
    return features
