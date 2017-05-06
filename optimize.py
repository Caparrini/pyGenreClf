import numpy as np
import pandas as pd
from classifier import KFoldAccuracy
from tpot import TPOTClassifier
from random import random, randint
from sklearn.tree import DecisionTreeClassifier
from abc import ABCMeta, abstractmethod


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

class BaseOptimizer():
    __metaclass__ = ABCMeta

    def __init__(self, df):
        self.df = df

    @abstractmethod
    def getClf(self, individual): pass
    @abstractmethod
    def initIndividual(self, pcls, maxints): pass

    def evaluateClf(self, individual):
        mean, std = KFoldAccuracy(self.df, self.getClf(individual))
        print("min_samples_split= "+str(individual[0])+"\n"+"min_samples_leaf= "+str(individual[1])+
              "\n" + "max_features= " + str(individual[2])
              +"  ----> Accuracy: " + str(mean) + " +- " + str(std) + "\n")
        return mean,std

    #TODO Unfinished
    def optimizeClf(self, population=30, generations=3):
        #Using deap, custom for decision tree
        from deap import base, creator, tools, algorithms
        creator.create("FitnessMax", base.Fitness, weights=(1.0,-1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # min_samples_split
        # min_samples_leaf
        maxints=[100,100]

        # Creation of individual and population
        toolbox = base.Toolbox()
        toolbox.register("individual", self.initIndividual, creator.Individual, maxints=maxints)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Methods for genetic algorithm
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=0, low=(2,1,0), up=(maxints[0],maxints[1],1), indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=5)
        toolbox.register("evaluate", self.evaluateClf)

        # Tools
        pop = toolbox.population(n=population)
        hof = tools.HallOfFame(1)

        # History
        hist = tools.History()
        toolbox.decorate("mate", hist.decorator)
        toolbox.decorate("mutate", hist.decorator)
        hist.update(pop)

        fpop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, halloffame=hof)

        best_score = hof[0].fitness.values[:]
        print(best_score)


        return fpop, logbook


class TreeOptimizer(BaseOptimizer):

    # TODO
    def getClf(self, individual):

        min_samples_split = int(individual[0])
        if (min_samples_split < 2):
            min_samples_split = 2

        min_samples_leaf = int(individual[1])
        if (min_samples_leaf) < 1:
            min_samples_leaf = 1

        max_features = individual[2]
        if (max_features > 1):
            max_features = 1

        clf = DecisionTreeClassifier(criterion="gini",
                                     splitter="best",
                                     max_features=max_features,
                                     max_depth=None,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     min_weight_fraction_leaf=0,
                                     max_leaf_nodes=None,
                                     random_state=None,
                                     min_impurity_split=0,
                                     presort=False)
        return clf
    def initIndividual(self, pcls, maxints):
        min_samples_split = randint(2,maxints[0])
        min_samples_leaf = randint(1,maxints[1])
        max_features = random()
        ind = pcls([min_samples_split,min_samples_leaf, max_features])
        return ind


