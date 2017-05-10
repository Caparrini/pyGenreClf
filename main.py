import pandas as pd
import featuresExtraction
import classifier
from sklearn import tree
import optimize
import librosa
import numpy as np

def main():
# display some lines
    df = pd.DataFrame.from_csv("CSV/beatsdataset.csv")
    clf = classifier.bestClfs()[1]
    classes, features, labels = classifier.unpackDF(df)
    clf.fit(features, labels)

    while(True):
        audio_file = raw_input("Write the path to an audio file (at least 2 min duration)")

        x, Fs = librosa.load(audio_file)
        x = librosa.resample(x, Fs, 22050)
        x = librosa.to_mono(x)

        feats = classifier.extractFeatures(22050, x[:22050*120], 1, 1, 0.05, 0.05)
        feats = np.append(feats, featuresExtraction.extractEssentiaBPM(x))

        print(clf.predict(feats))



if __name__ == "__main__": main()
