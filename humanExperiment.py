import os
import numpy as np
import pandas as pd
from shutil import copy2
from sklearn.metrics import confusion_matrix, accuracy_score


class ExperimentUtils(object):
    """
    This class is used to create experiments splitting the dataset for human classification.
    Also evaluate the performance of that classification.
    """

    def __init__(self, genres_folder, exp_folder):
        """
        Initialize the class

        :param genres_folder: Folder where is the subfolder with musical genres
        :param exp_folder: Folder to create the experiment
        """
        self.genres_folder = genres_folder
        self.exp_folder = exp_folder
        self.genres_dict = {"BigRoom": "A", "ElectroHouse": "B", "DrumAndBass": "C",
                            "Dubstep": "D", "HipHop": "E", "Dance": "F", "FutureHouse": "G"}
        self.inv_genres_dict = dict(map(reversed, self.genres_dict.items()))

    def _generate_full_alias_dataframes(self, n_songs):
        """
        
        :param n_songs: Number of songs to select at random for every genre
        :return: two pd.DataFrame
        dftest -> contains the names of the test songs of all the genres
        dftrain -> contains the names of the train songs of all the genres
        """
        genres_folders = [self.genres_folder + "/" + g for g in self.genres_dict.keys()]
        dftest = pd.DataFrame()
        dftrain = pd.DataFrame()
        for g in genres_folders:
            df1, df2 = self._generate_alias_dataframes(g, n_songs)
            dftest = pd.concat([dftest, df1])
            dftrain = pd.concat([dftrain, df2])
        anonymized_names = ["song" + str((x % 100) / 10) + str(x % 10) for x in range(1, dftest.shape[0] + 1)]
        dftest = dftest.sample(frac=1)  # Relocation at random, avoiding songs grouped by genre
        dftest["test_name"] = anonymized_names  # Use the anonymized_names with the random DataFrame
        return dftest, dftrain

    def _generate_alias_dataframes(self, genre_folder, n_songs):
        """
        
        :param genre_folder: Genre folder to extract alias DataFrames
        :param n_songs: Number of songs to select at random of the genre
        :return: two pd.DataFrame
        dftest -> contains the names of the test songs of genre_folder
        dftrain -> contains the names of the train songs of genre_folder
        """
        songs = [y for y in [x for x in os.walk(genre_folder)][0][2] if ".mp3" in y]
        if (len(songs) != 100):
            print("WARNING: The genre folder " + genre_folder + " only have " + str(len(songs)) + " songs.")
        genre = os.path.basename(genre_folder)
        label_names = [genre + str(x / 100) + str((x % 100) / 10) + str(x % 10) for x in range(1, len(songs) + 1)]
        labels = [genre] * len(songs)
        df = pd.DataFrame()
        df["class"] = labels  # genre label
        df["real_name"] = songs  # filename
        df["label_name"] = label_names  # dataset name
        dftest, dftrain = np.split(df.sample(n_songs), 2, axis=0)
        train_names = [self.genres_dict[genre] +  # Names according to the label in genres_dict
                       str((x % 100) / 10) + str(x % 10) for x in range(1, dftrain.shape[0] + 1)]

        dftrain["train_name"] = train_names  # Anonymizing train
        return dftest, dftrain

    def build_experiment(self, number_of_songs):
        """
        Creates the experiment folder at exp_folder path

        :param number_of_songs: Number of songs of each genre (half to train, half to test)
        """
        dftest, dftrain = self._generate_full_alias_dataframes(n_songs=number_of_songs)  # Get names DataFrames
        os.mkdir(self.exp_folder)  # Create main exp folder
        train_folder = os.path.join(self.exp_folder, "train")
        test_folder = os.path.join(self.exp_folder, "test")
        eval_folder = os.path.join(self.exp_folder, "evaluation")
        os.mkdir(train_folder)
        os.mkdir(test_folder)
        # Create one folder for each genre in train and test folders
        for key in self.genres_dict.keys():
            gen = dftrain["class"] == key  # Rule to select the actual genre later in dftrain
            os.mkdir(os.path.join(train_folder, self.genres_dict[key]))
            os.mkdir(os.path.join(test_folder, self.genres_dict[key]))  # Empty folder
            for index, row in dftrain[gen].iterrows():
                source = os.path.join(self.genres_folder, key, row["real_name"])  # Song in the dataset folder
                target = os.path.join(train_folder,
                                      self.genres_dict[key], row["train_name"] + ".mp3")  # Song anonymized
                copy2(source, target)  # Copying into train/label folder
                print(source, target)
        # Add the test songs to the test folder
        for index, row in dftest.iterrows():
            source = os.path.join(self.genres_folder, row["class"], row["real_name"])  # Song in the dataset folder
            target = os.path.join(test_folder, row["test_name"] + ".mp3")  # Song anonymized
            copy2(source, target)  # Copying into test folder
            print(source, target)

        os.mkdir(eval_folder)  # Create the evaluation folder to save results
        # Save the alias files to evaluate when the experiments will be done
        pd.DataFrame.to_csv(dftrain, os.path.join(eval_folder, "train_songs.csv"))
        pd.DataFrame.to_csv(dftest, os.path.join(eval_folder, "test_songs.csv"))

    def evaluate_results(self):
        """
        Evaluates the performance in the experiment given the saved .CSV test file with the alias and the folder test
        modified by the subjects. This evaluation is shown as a confusion_matrix
        """
        dftest = pd.DataFrame.from_csv(os.path.join(self.exp_folder, "evaluation", "test_songs.csv"))
        dfeval = self.get_evaluation_df(os.path.join(self.exp_folder, "test"))
        # Joining the DataFrame we used and the one results from the user by the "test_name" column in both
        dfeval = pd.merge(dftest, dfeval)
        labels = dfeval["class"]  # Real class
        pred_labels = dfeval["pred_class"]  # Predicted class
        cm = confusion_matrix(labels, pred_labels, self.genres_dict.keys())
        print(cm)
        return cm

    def get_evaluation_df(self, test_folder):
        dfeval = pd.DataFrame()
        # Generate DataFrame with the name of the songs and the label predicted by the subject
        for key in self.inv_genres_dict:
            dfaux = pd.DataFrame()
            songs = [y.replace(".mp3", "") for y in
                     [x for x in os.walk(os.path.join(test_folder, key))][0][2] if ".mp3" in y]
            dfaux["test_name"] = songs
            pred_labels = [self.inv_genres_dict[key]] * len(songs)
            dfaux["pred_class"] = pred_labels
            dfeval = pd.concat([dfeval, dfaux])
        return dfeval

    def evaluate_all_exps(self, experiments_folder):

        accuracies = []
        dftest = pd.DataFrame.from_csv(os.path.join(self.exp_folder, "evaluation", "test_songs.csv"))
        experiment_folders = [x for x in os.walk(experiments_folder)][0][1]
        dffinal = pd.DataFrame()

        for f in experiment_folders:
            dfeval = self.get_evaluation_df(os.path.join(experiments_folder,f,"test"))
            dfeval = pd.merge(dftest, dfeval)
            labels = dfeval["class"]  # Real class
            pred_labels = dfeval["pred_class"]  # Predicted class
            accuracies.append(accuracy_score(labels, pred_labels))
            dffinal = pd.concat([dffinal, dfeval])

        labels = dffinal["class"]  # Real class
        pred_labels = dffinal["pred_class"]  # Predicted class
        cm = confusion_matrix(labels, pred_labels, sorted(self.genres_dict.keys()))
        print(cm)
        print(accuracies)
        print(np.mean(accuracies))
        print(np.std(accuracies))
        return cm

    def evaluate_all_forest_style(self, experiments_folder):
        accuracies = []
        dftest = pd.DataFrame.from_csv(os.path.join(self.exp_folder, "evaluation", "test_songs.csv"))
        experiment_folders = [x for x in os.walk(experiments_folder)][0][1]
        dffinal = pd.DataFrame()
        for f in experiment_folders:
            dfeval = self.get_evaluation_df(os.path.join(experiments_folder,f,"test"))
            dfeval = pd.merge(dftest, dfeval)
            labels = dfeval["class"]  # Real class
            pred_labels = dfeval["pred_class"]  # Predicted class
            accuracies.append(accuracy_score(labels, pred_labels))
            dffinal = pd.concat([dffinal, dfeval])
        return dffinal