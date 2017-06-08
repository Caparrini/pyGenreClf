import pandas as pd
import classifier
import librosa
import argparse
from featuresExtraction import extractFeaturesFolder

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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Examples of use")
    tasks = parser.add_subparsers(title="subcommands", description="available tasks", dest="task", metavar="")

    featExt = tasks.add_parser("featureExtractionDataset", help="Extract audio features from file")
    featExt.add_argument("-f", "--dataset_folder", required=True, help="Input audio file")
    featExt.add_argument("-o", "--output_DataFrame", required=True, help="Output file")
    featExt.add_argument("-mw", "--mtwin", type=float, default=1, help="Mid-term window size")
    featExt.add_argument("-ms", "--mtstep", type=float, default=1, help="Mid-term window step")
    featExt.add_argument("-sw", "--stwin", type=float, default=0.050, help="Short-term window size")
    featExt.add_argument("-ss", "--ststep", type=float, default=0.050, help="Short-term window step")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.task == "featureExtractionDataset":
        extractFeaturesFolder(args.output_DataFrame, args.dataset_folder, args.mtwin, args.mtstep, args.stwin, args.ststep)
