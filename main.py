import pandas as pd
import classifier
import librosa

def main():
# display some lines
    df = pd.DataFrame.from_csv("CSV/beatsdataset.csv")
    clf = classifier.bestClfs()[1]
    classes, features, labels = classifier.unpackDF(df)
    clf.fit(features, labels)

    while(True):
        #TODO To decide exact input, maybe web link and use youtube-dl?
        #TODO Use pickle to save/load model EASY
        audio_file = raw_input("Write the path to an audio file (at least 2 min duration)")

        x, Fs = librosa.load(audio_file)
        x = librosa.resample(x, Fs, 22050)
        x = librosa.to_mono(x)
        #TODO To decide which 2 minutes of the song we use
        feats = classifier.extractFeatures(22050, x[:22050*120], 1, 1, 0.05, 0.05)
        print(clf.predict([feats]))



if __name__ == "__main__": main()
