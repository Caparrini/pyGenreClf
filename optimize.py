import numpy as np
import pandas as pd
from classifier import getClf, KFoldAccuracy
from tpot import TPOTClassifier


def autoTPOT(df, export='tpot_pipe.py'):
    #labels = list(df["class"].values)
    features = []
    for j in range(df.shape[0]):
        item = df.ix[j]
        features.append([item[i] for i in range(len(item) - 1)])

    features = np.array(features)

    labels, uniques = pd.factorize(df["class"])

    #from sklearn.model_selection import train_test_split

    #features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

    pipeline_optimizer = TPOTClassifier(generations=10, population_size=10, verbosity=3)

    pipeline_optimizer.fit(features, labels)
    #print(pipeline_optimizer.score(features_test, labels_test))
    pipeline_optimizer.export(export)



class Optimizer():
    def __init__(self, df):
        self.df = df

    def evaluateClf(self, individual):
        mean, std = KFoldAccuracy(self.df, getClf(individual))
        print("min_samples_split= "+str(individual[0])+"  ----> Accuracy: " + str(mean) + " +- " + str(std) + "\n")
        return mean, std

    #TODO Unfinished
    def optimizeClf(self):
        #Using deap, custom for decision tree
        from deap import base, creator, tools, algorithms
        from random import random, randint
        creator.create("FitnessMax", base.Fitness, weights=(1.0,-1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        IND_SIZE=2

        toolbox = base.Toolbox()
        toolbox.register("min_sample_split", randint, 2, 100)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.min_sample_split, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("evaluate", self.evaluateClf)

        pop = toolbox.population(n=10)
        fpop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=2)
        return fpop, logbook
