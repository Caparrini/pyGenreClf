import numpy as np
from classifier import KFoldAccuracy
from random import random, randint, uniform
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import pathos.multiprocessing as multiprocessing


class Param(object):
    """
    This class object saves the params to generalize the optimize class
    """
    def __init__(self, name, minValue, maxValue, typeParam):
        """
        Init object

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

    def __init__(self, df, log_file):
        """

        :param df: (DataFrame) DataFrame to train and test the classifier 
        (maybe in the future this must be change for features, labels list which is more usual)
        """
        self.df = df
        self.params = self.getParams()
        self.eval_dict = {}
        self.file_out = open(log_file, "w")


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
                ps.append(round(uniform(p.minValue, p.maxValue), 3))
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
        Method to evaluate the individual, in this case the classifier

        :param individual: individual for evaluation
        :return: mean accuracy, standard deviation accuracy
        """
        for i in range(len(self.params)):
            individual[i] = self.params[i].correct(individual[i])

        if tuple(individual) in self.eval_dict:
            self.file_out.write("Individual has been evaluated before\n")
            meanstd = self.eval_dict[tuple(individual)]
            mean = meanstd[0]
            std = meanstd[1]
        else:
            self.file_out.write("Individual has NOT been evaluated before\n")
            mean, std = KFoldAccuracy(self.df, self.getClf(individual))
            self.eval_dict[tuple(individual)] = tuple((mean, std))

        out = "Individual evaluation:\n"
        for i in range(len(self.params)):
            out += self.params[i].name + " = " + str(individual[i]) + "\n"
        out += "  ----> Accuracy: " + str(mean) + " +- " + str(std) + "\n"
        self.file_out.write(out)
        return mean,

    def optimizeClf(self, population=10, generations=3):
        '''
        Searches through a genetic algorithm the best classifier

        :param int population: Number of members of the first generation
        :param int generations: Number of generations
        :return: Trained classifier
        '''
        #self.eval_dict = {}
        self.file_out.write("Optimizing accuracy:\n")
        # Using deap, custom for decision tree
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Creation of individual and population
        toolbox = base.Toolbox()

        # Paralel
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

        toolbox.register("individual", self.initIndividual, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Methods for genetic algorithm
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.5, low=[x.minValue for x in self.params],
                         up=[x.maxValue for x in self.params], indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("evaluate", self.evaluateClf)

        # Tools
        pop = toolbox.population(n=population)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda  ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # History
        hist = tools.History()
        toolbox.decorate("mate", hist.decorator)
        toolbox.decorate("mutate", hist.decorator)
        hist.update(pop)

        fpop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                            ngen=generations, stats=stats,
                                            halloffame=hof)

        best_score = hof[0].fitness.values[:]

        self.file_out.write("LOGBOOK: \n"+str(logbook)+"\n")
        self.file_out.write("Best accuracy: "+str(best_score[0])+"\n")
        self.file_out.write("Best classifier: "+str(self.getClf(hof[0])))

        self.plotLogbook(logbook=logbook)
        return self.getClf(hof[0])

    def plotLogbook(self, logbook):
        '''
        Plots the given loogboook

        :param logbook: logbook of the genetic algorithm
        '''

        gen = logbook.select("gen")
        fit_max = logbook.select("max")
        fit_avg = logbook.select("avg")

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, fit_max, "b-", label="Max fit")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness", color="b")

        line2 = ax1.plot(gen, fit_avg, "r-", label="Avg fit")

        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="lower right")

        plt.savefig("optfig")


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
                                     max_features=None,
                                     max_depth=None,
                                     min_samples_split=individual[0],
                                     min_samples_leaf=individual[1],
                                     min_weight_fraction_leaf=0,
                                     max_leaf_nodes=None,
                                     random_state=None,
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
        params.append(Param("min_samples_leaf", 1, 100, int))
        # Return all the params
        return params


class ForestOptimizer(TreeOptimizer):
    """
    Concrete optimizer for sklearn random forest -> sklearn.ensemble.RandomForestClassifier
    """
    def getClf(self, individual):
        """
        Builds a classifier object from an individual one

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
                                     bootstrap=True,
                                     oob_score=True,
                                     n_jobs=-1,
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
        params = super(ForestOptimizer, self).getParams()
        # max_features
        params.append(Param("max_features", 0, 1, float))
        # n_estimator
        params.append(Param("n_estimators", 100, 350, int))
        # Return all the params
        return params


class ExtraTreesOptimizer(ForestOptimizer):
    """
    Concrete optimizer for sklearn extra trees -> sklearn.ensemble.ExtraTreesClassifier
    Use the same getParams() as ForestOptimizer
    """
    def getClf(self, individual):
        """
        Builds a classifier object from an individual one

        :param individual: individual to create a classifier
        :return: classifier ExtraTreesClassifier
        """
        clf = ExtraTreesClassifier(n_estimators=individual[3],
                                   criterion="gini",
                                   max_depth=None,
                                   min_samples_split=individual[0],
                                   min_samples_leaf=individual[1],
                                   min_weight_fraction_leaf=0,
                                   max_features=individual[2],
                                   max_leaf_nodes=None,
                                   bootstrap=False,
                                   oob_score=False,
                                   n_jobs=-1,
                                   random_state=None,
                                   verbose=0,
                                   warm_start=False,
                                   class_weight=None)
        return clf


class GradientBoostingOptimizer(ForestOptimizer):
    '''
    Concrete optimizer for sklearn gradient boosting -> sklearn.ensemble.GradientBoostingClassifier
    Use the same getParams() as ForestOptimizer
    '''
    def getParams(self):
        """
        Params for the creation of individuals (relative to the algorithm)
        These params define the name of the param, min value, max value, and type

        :return: list of params
        """
        params = super(GradientBoostingOptimizer, self).getParams()
        # learning_rate
        params.append(Param("learning_rate", 0.00001, 0.1, float))
        # subsample
        params.append(Param("subsample", 0, 1, float))
        # Return all the params
        return params

    def getClf(self, individual):
        """
        Builds a classifier object from an individual one

        :param individual: individual to create a classifier
        :return: classifier ExtraTreesClassifier
        """
        clf = GradientBoostingClassifier(n_estimators=individual[3],
                                         criterion="friedman_mse",
                                         max_depth=None,
                                         min_samples_split=individual[0],
                                         min_samples_leaf=individual[1],
                                         min_weight_fraction_leaf=0,
                                         max_features=individual[2],
                                         max_leaf_nodes=None,
                                         random_state=None,
                                         verbose=0,
                                         warm_start=False,
                                         learning_rate=individual[4],
                                         subsample=individual[5])
        return clf