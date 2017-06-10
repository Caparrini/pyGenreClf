import pandas as pd
import classifier
import librosa
import argparse
import joblib
from featuresExtraction import extractFeaturesFolder, extractFeatures
from optimize import ForestOptimizer, TreeOptimizer
from classifier import KFoldCrossValidation, TreeKFoldReport

def main():
    args = parse_arguments()

    if args.task == "featureExtractionDataset":
        extractFeaturesFolder(args.output_DataFrame, args.dataset_folder, args.mtwin, args.mtstep, args.stwin, args.ststep)
    elif args.task == "bestForestClassifier":
        df = pd.DataFrame.from_csv(args.DataFrame)
        opt = ForestOptimizer(df)
        clf = opt.optimizeClf(args.population, args.generations)
        clf = KFoldCrossValidation(df, args.report_folder, clf)
        joblib.dump(clf, args.clf_file)
    elif args.task == "bestTreeClassifier":
        df = pd.DataFrame.from_csv(args.DataFrame)
        opt = TreeOptimizer(df)
        clf = opt.optimizeClf(args.population, args.generations)
        clf = TreeKFoldReport(df, args.report_folder, clf)
        joblib.dump(clf, args.clf_file)
    elif args.task == "predictClass":
        clf = joblib.load(args.classifier)
        x, Fs = librosa.load(args.input)
        x = librosa.resample(x, Fs, 22050)
        x = librosa.to_mono(x)
        feats = extractFeatures(22050, x[:22050 * 120], 1, 1, 0.05, 0.05)
        print(clf.predict([feats]))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Examples of use")
    tasks = parser.add_subparsers(title="subcommands", description="available tasks", dest="task", metavar="")

    featExt = tasks.add_parser("featureExtractionDataset", help="Extract audio features from dataset folder")
    featExt.add_argument("-f", "--dataset_folder", required=True, help="Input structured folder with subfolders of audio files. Each subfolder is")
    featExt.add_argument("-o", "--output_DataFrame", required=True, help="Output file")
    featExt.add_argument("-mw", "--mtwin", type=float, default=1, help="Mid-term window size")
    featExt.add_argument("-ms", "--mtstep", type=float, default=1, help="Mid-term window step")
    featExt.add_argument("-sw", "--stwin", type=float, default=0.050, help="Short-term window size")
    featExt.add_argument("-ss", "--ststep", type=float, default=0.050, help="Short-term window step")

    bestForestClf = tasks.add_parser("bestForestClassifier", help="Generate the best random forest classifier and generates a report")
    bestForestClf.add_argument("-df", "--DataFrame", required=True, help="Input pandas.DataFrame dataset")
    bestForestClf.add_argument("-o", "--clf_file", required=True, help="Generated binary classifier file")
    bestForestClf.add_argument("-f", "--report_folder", required=True, help="Folder to save all the report data")
    bestForestClf.add_argument("-p", "--population", type=int, default=30, help="Initial population for genetic algorithm")
    bestForestClf.add_argument("-g", "--generations", type=int, default=50, help="Number of generations")

    bestTreeClf = tasks.add_parser("bestTreeClassifier", help="Generate the best decission tree classifier and generates a report")
    bestTreeClf.add_argument("-df", "--DataFrame", required=True, help="Input pandas.DataFrame dataset")
    bestTreeClf.add_argument("-o", "--clf_file", required=True, help="Generated binary classifier file")
    bestTreeClf.add_argument("-f", "--report_folder", required=True, help="Folder to save all the report data")
    bestTreeClf.add_argument("-p", "--population", type=int, default=30, help="Initial population for genetic algorithm")
    bestTreeClf.add_argument("-g", "--generations", type=int, default=50, help="Number of generations")

    predictClass = tasks.add_parser("predictClass", help="Predict the class of the given audio file or address")
    predictClass.add_argument("-i", "--input", required=True, help="Audio filename")
    predictClass.add_argument("-clf", "--classifier", required=True, help="Classifier filename")

    return parser.parse_args()

if __name__ == "__main__": main()
