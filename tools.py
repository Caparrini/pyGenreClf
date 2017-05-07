import os
import numpy as np
import networkx as nx


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


# Class object to generate and save the metrics of the confusion matrix
# In creation it builds a list of dictionaries with the TP,TN,FP,FN of the cm
# Summoning each function calculate the asked metric
#TODO care with zero at the denominator
class ConfusionMatrixUtils(object):
    """
    cm : confusion matrix (equal rows and columns)
    class_names : list of names of the classes present in the confusion matrix ( len = cm.shape[0] = cm.shape[1] )
    """
    def __init__(self, cm, class_names):
        self.cm = cm
        self.class_names = class_names
        self.generateMetrics()

    # It generates the basic metrics for each class (TP,FP,TN,FN)
    def generateMetrics(self):
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

    # Sensitivity, recall, hit rate or true positive rate(TPR)
    def recall(self,index):
        # TPR = TP / (TP + FN)
        dict = self.metrics_class[index]
        return float(dict["TP"])/(dict["TP"]+dict["FN"])

    # Specifity or true negative rate TNR
    def specificity(self,index):
        # TNR = TN / (TN + FP)
        dict = self.metrics_class[index]
        return float(dict["TN"])/(dict["TN"]+dict["FP"])

    # Precision or positive predictive value PPV
    def precision(self,index):
        # PPV = TP / (TP + FP)
        dict = self.metrics_class[index]
        return float(dict["TP"])/(dict["TP"]+dict["FP"])

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

    # Harmonic mean of precision and sensitivity
    def F1score(self,index):
        # 2TP / (2*TP + FP + FN)
        dict = self.metrics_class[index]
        return float(2*dict["TP"])/(2*dict["TP"]+dict["FP"]+dict["FN"])

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
        r = "Confusion Matrix Metrics Report\n\n"
        for i in range(len(self.class_names)):
            r+= self.class_names[i]+":\n\t"
            r+= str(self.metrics_class[i])+"\n"
            r+= "\tPrecision: " + str(self.precision(i)) + " Recall: " + str(self.recall(i)) + " Accuracy: " + str(self.accuracy(i)) +"\n"
        return r

    # Transforms the current cmm into a graph
    def cmmToGraph(self):
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
