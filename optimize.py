import numpy as np
import pandas as pd
from classifier import KFoldAccuracy
from tpot import TPOTClassifier
from random import random, randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from abc import ABCMeta, abstractmethod


def autoTPOT(df, export='tpot_pipe.py'):
    # labels = list(df["class"].values)
    features = []
    for j in range(df.shape[0]):
        item = df.ix[j]
        features.append([item[i] for i in range(len(item) - 1)])

    features = np.array(features)

    labels, uniques = pd.factorize(df["class"])

    # from sklearn.model_selection import train_test_split

    # features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

    pipeline_optimizer = TPOTClassifier(generations=10, population_size=10, verbosity=3)

    pipeline_optimizer.fit(features, labels)
    # print(pipeline_optimizer.score(features_test, labels_test))
    pipeline_optimizer.export(export)


class Param(object):
    """
    This class object saves the params to generalize the optimize class
    """
    def __init__(self, name, minValue, maxValue, typeParam):
        """
        :param name: (str) Name of the param
        :param minValue: (int) Minimum value of the param
        :param maxValue: (int) Maximum value of the param
        :param typeParam: (type) type of the param
        """
        self.name = name
        self.minValue = minValue
        self.maxValue = maxValue
        self.type = typeParam

    def correct(self, value):
        """
        :param value: value to verify if accomplishes type, min and max due to mutations
        :return: value fixed
        """
        if self.type==int:
            ret=int(value)
        else:
            ret=value

        if ret>self.maxValue:
            ret=self.maxValue
        elif ret<self.minValue:
            ret=self.minValue

        return ret


class BaseOptimizer(object):
    """
    Abstract class to create optimizer for different machine learning classifier algorithms
    """
    __metaclass__ = ABCMeta

    def __init__(self, df):
        """
        :param df: (DataFrame) DataFrame to train and test the classifier 
        (maybe in the future this must be change for features, labes list whis is more usual)
        """
        self.df = df
        self.params = self.getParams()

    def initIndividual(self, pcls):
        """
        Method to initialize an individual instance
        :param pcls: Method to create the individual as an extension of the class list
        :return: individual
        """
        ps = []
        for p in self.params:
            if p.type == int:
                ps.append(randint(p.minValue, p.maxValue))
            else:
                ps.append(random())
        ind = pcls(ps)
        return ind

    @abstractmethod
    def getParams(self):
        pass

    @abstractmethod
    def getClf(self, individual):
        pass

    def evaluateClf(self, individual):
        """
        Method to evaluate the individual, in this cases the classifier
        :param individual: individual for evaluation
        :return: mean accuracy, standard deviation accuracy
        """
        for i in range(len(self.params)):
            individual[i] = self.params[i].correct(individual[i])

        mean, std = KFoldAccuracy(self.df, self.getClf(individual))
        out = "Individual evaluation:\n"
        for i in range(len(self.params)):
            out += self.params[i].name + " = " + str(individual[i]) + "\n"
        out += "  ----> Accuracy: " + str(mean) + " +- " + str(std) + "\n"
        print(out)
        return mean, std

    def optimizeClf(self, population=10, generations=3):
        # Using deap, custom for decision tree
        from deap import base, creator, tools, algorithms
        creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Creation of individual and population
        toolbox = base.Toolbox()
        toolbox.register("individual", self.initIndividual, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Methods for genetic algorithm
        toolbox.register("mate", tools.cxTwoPoint)
        # TODO the mut changes if the params change, refactor?
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.5, low=[x.minValue for x in self.params],
                         up=[x.maxValue for x in self.params], indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=2)
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
        # TODO Visualize results

        return fpop, logbook


class TreeOptimizer(BaseOptimizer):
    """
    Concrete optimizer for sklearn classifier -> sklearn.tree.DecisionTreeClassifier
    """
    def getClf(self, individual):
        """
        Build a classifier object from an individual one
        :param individual: individual to create classifier
        :return: classifier sklearn.tree.DecisionTreeClassifier
        """

        clf = DecisionTreeClassifier(criterion="gini",
                                     splitter="best",
                                     max_features=individual[2],
                                     max_depth=None,
                                     min_samples_split=individual[0],
                                     min_samples_leaf=individual[1],
                                     min_weight_fraction_leaf=0,
                                     max_leaf_nodes=None,
                                     random_state=None,
                                     min_impurity_split=1e-7,
                                     presort=False)
        return clf

    def getParams(self):
        """
        Params for the creation of individuals (relative to the algorithm)
        These params define the name of the param, min value, max value, and type
        :return: list of params
        """
        params = []
        # min_samples_split
        params.append(Param("min_samples_split", 2, 100, int))
        # min_samples_leaf
        params.append(Param("min_samples_leaf", 2, 100, int))
        # max_features
        params.append(Param("max_features", 0, 1, float))
        # Return all the params
        return params


class ForestOptimizer(BaseOptimizer):
    """
    Concrete optimizer for sklearn random forest -> sklearn.enemble.RandomForestClassifier
    """
    def getClf(self, individual):
        """
        Build a classifier object from an individual one
        :param individual: individual to create classifier
        :return: classifier sklearn.ensemble.RandomForestClassifier
        """
        clf = RandomForestClassifier(n_estimators=individual[3],
                                     criterion="gini",
                                     max_depth=None,
                                     min_samples_split=individual[0],
                                     min_samples_leaf=individual[1],
                                     min_weight_fraction_leaf=0,
                                     max_features=individual[2],
                                     max_leaf_nodes=None,
                                     min_impurity_split=1e-7,
                                     bootstrap=True,
                                     oob_score=True,
                                     n_jobs=4,
                                     random_state=None,
                                     verbose=0,
                                     warm_start=False,
                                     class_weight=None
                                     )
        return clf

    def getParams(self):
        """
        Params for the creation of individuals (relative to the algorithm)
        These params define the name of the param, min value, max value, and type
        :return: list of params
        """
        params = []
        # min_samples_split
        params.append(Param("min_samples_split", 2, 100, int))
        # min_samples_leaf
        params.append(Param("min_samples_leaf", 2, 100, int))
        # max_features
        params.append(Param("max_features", 0, 1, float))
        # n_estimator
        params.append(Param("n_estimators", 10, 100, int))
        # Return all the params
        return params
