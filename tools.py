import os
import numpy as np
import networkx as nx
import pandas as pd


# Given the route of the genres directory, transforms every song into 
# .wav format and deletes the old one.
# Receives nothing
# Returns nothing
def songs_to_wav():
    os.chdir("./")

    genre_folders = [x for x in os.walk("./")]
    genre_folders.pop(0)
    for g in genre_folders:
        for s in g[2]:
            if(".wav" not in s):
                ruta = g[0]+"/"+s
                os.system('ffmpeg -i "' + ruta +
                '" -acodec pcm_s16le -ar 22050 -ac 1 "' + ruta + '.wav"')
                os.remove(ruta)


class ConfusionMatrixUtils(object):
    """
    Class object to generate and save the metrics of the confusion matrix.
    Whe it is created it builds a list of dictionaries with the TP,TN,FP,FN of the cm.
    Summoning each function calculate the asked metric.

    cm : confusion matrix (equal rows and columns)
    class_names : list of names of the classes present in the confusion matrix ( len = cm.shape[0] = cm.shape[1] )
    """
    def __init__(self, cm, class_names):
        self.cm = cm
        self.class_names = class_names
        self.generateMetrics()

    def generateMetrics(self):
        '''
        It generates the basic metrics for each class (TP,FP,TN,FN)

        '''
        self.metrics_class = []
        for i in range(self.cm.shape[0]):
            metrics = {}
            metrics["TP"] = self.cm[i,i]
            metrics["FP"] = sum(self.cm[[j for j in range(self.cm.shape[1]) if j!=i],i])
            aux = np.delete(self.cm,(i),axis=0)
            aux = np.delete(aux,(i),axis=1)
            metrics["TN"] = sum(sum(aux))
            metrics["FN"] = sum(self.cm[i,[j for j in range(self.cm.shape[1]) if j!=i]])
            self.metrics_class.append(metrics)

    def recall(self,index):
        '''
        Sensitivity, recall, hit rate or true positive rate(TPR)
        TPR = TP / (TP + FN)

        :param index: index of the class
        :return: recall or -1 if it can not be calculated
        '''
        dict = self.metrics_class[index]
        try:
            recall = float(dict["TP"])/(dict["TP"]+dict["FN"])
        except ZeroDivisionError:
            print("ERROR: TP+FN=0 (division by zero), recall can't be calculated.")
            recall = -1
        return recall

    # Specifity or true negative rate TNR
    def specificity(self,index):
        # TNR = TN / (TN + FP)
        dict = self.metrics_class[index]
        return float(dict["TN"])/(dict["TN"]+dict["FP"])

    def precision(self,index):
        '''
        Precision or positive predictive value PPV
        PPV = TP / (TP + FP)

        :param index: index of the class
        :return: precision or -1 if it can not be calculated
        '''
        dict = self.metrics_class[index]
        try:
            precision = float(dict["TP"])/(dict["TP"]+dict["FP"])
        except ZeroDivisionError:
            print("ERROR: TP+FP=0 (division by zero), precision can't be calculated.")
            precision = -1
        return precision

    # Negative predictive value NPV
    def NPV(self,index):
        # NPV = TN / (TN+FN)
        dict = self.metrics_class[index]
        return float(dict["TN"])/(dict["TN"]+dict["FN"])

    # Miss rate or false negative rate FNR
    def missrate(self,index):
        # FNR = 1 - TPR
        return 1-self.recall(index)

    # Fall-out rate of false negative rate FNR
    def fallout(self,index):
        # FPR = 1 - TNR
        return 1- self.specificity(index)

    def accuracy(self,index):
        # ACC = (TP + TN) / ( (FN + TP) + (FP + TN) )
        dict = self.metrics_class[index]
        return float(dict["TP"]+dict["TN"]) / (dict["FN"]+dict["TP"]+dict["FP"]+dict["TN"])

    def F1score(self,index):
        '''
        Harmonic mean of precision and sensitivity
        2TP / (2*TP + FP + FN)

        :param index: index of the class
        :return: f1score or -1 if it can not be calculated
        '''
        dict = self.metrics_class[index]
        try:
            f1s = float(2*dict["TP"])/(2*dict["TP"]+dict["FP"]+dict["FN"])
        except ZeroDivisionError:
            print("ERROR: TP=FP=FN=0 (division by zero), F1score can't be calculated.")
            f1s = -1
        return f1s

    # Matthews correlation coefficient
    def MCC(self,index):
        # MCC = (TP*TN - FP*FN) / SQRT[(TP+FP) (TP+FN) (TN+FP) (TN+FN)]
        dict = self.metrics_class[index]
        return float((dict["TP"]*dict["TN"]) - (dict["FP"]*dict["FN"]))/\
               np.sqrt((dict["TP"]+dict["FP"])*(dict["TP"]+dict["FN"])*(dict["TN"]+dict["FP"])*(dict["TN"]+dict["FN"]))

    # Informedness or Bookmaker Informedness BM
    def informedness(self,index):
        # BM = TPR + TNR -1
        return self.recall(index)+self.specificity(index)-1

    def markedness(self,index):
        # MK = PPV + NPV -1
        return self.precision(index)+self.NPV(index)-1

    def report(self):
        '''
        Generates a string as a report given all the class names,
        recall, precision and f1 score.

        :return str: report
        '''
        r = "Confusion Matrix Metrics Report\n\n"
        for i in range(len(self.class_names)):
            r+= self.class_names[i]+":\n\t"
            r+= str(self.metrics_class[i])+"\n"
            r+= "\tPrecision: " + str(round(self.precision(i), 2)) + " Recall: " + str(round(self.recall(i), 2)) + \
                " F1score: " + str(round(self.F1score(i), 2)) +"\n"
        return r

    def cmmToGraph(self):
        '''
        Transforms the current cm into a graph

        :return networkx.DiGraph: confusion graph
        '''
        G = nx.DiGraph()

        # Add node for each class_name with weight equal to
        # his accuracy rate
        for i in range(len(self.class_names)):
            G.add_node(self.class_names[i], weight=self.cm[i][i])

        # Add directed edge between each class when it is a confusion
        # in the matrix
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if(i!=j and self.cm[i][j]>0):
                    G.add_edge(self.class_names[i], self.class_names[j],
                               weight=self.cm[i][j])

        return G

def beatsdataset():
    # Base folder with songs in mp3
    old = "../../Machine Learning/Datasets/genres-beatport/"
    new = "beatsdataset/"
    genresFolder = [x for x in os.walk(old)]

    genres = genresFolder[0][1]

    for g in genres:
        print("Generating wav files for " + g + "\n")

        os.mkdir(new+g)
        # File to save the song names list
        songsListFile = open(new + g + "/" + g + ".csv","w")
        songsListFile.write("Alias;Name\n")

        songs = [x for x in os.walk(old+g)][0][2]
        count = 0
        for s in songs:
            if ".mp3" in s:
                count = count+1
                oldfile = old + g + "/" + s
                newfile = new + g + "/" + g + str(count/100) + str((count%100)/10) + str(count%10)
                os.system('ffmpeg -i "' + oldfile +
                '" -acodec pcm_s16le -ar 22050 -ac 1 "' + newfile + '.wav"')
                songsListFile.write(newfile+";"+s+"\n")
        print("Transferred " + str(count) + " songs\n")
        print("Closing list file: " + g + "/")
        songsListFile.close()

def create_registry_dataset(dataset_foler, format=".wav"):
    folders = [x for x in os.walk(dataset_foler)][0][1]
    df = pd.DataFrame()
    for f in folders:
        audioFiles = [y.replace(format,"") for y in [x for x in os.walk(os.path.join(dataset_foler, f))][0][2] if format in y]
        if (len(audioFiles) != 100):
            print("WARNING: The genre folder " + f + " only have " + str(len(audioFiles)) + " songs.")
        label_names = [f + str(x / 100) + str((x % 100) / 10) + str(x % 10) for x in range(1, len(audioFiles) + 1)]
        labels = [f] * len(audioFiles)
        dfaux = pd.DataFrame()
        dfaux["class"] = labels  # genre label
        dfaux["real_name"] = audioFiles  # filename
        dfaux["label_name"] = label_names  # dataset name
        df = pd.concat([df, dfaux])
    return df
