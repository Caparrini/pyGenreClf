import pandas as pd
import features
import classifier
from sklearn import tree

df = pd.DataFrame.from_csv("beatsdataset.csv")


clf = tree.DecisionTreeClassifier(criterion = "gini",
                                  splitter = "best",
                                  max_features = None,
                                  max_depth = 8,
                                  min_samples_split = 40,
                                  min_samples_leaf = 10,
                                  min_weight_fraction_leaf = 0,
                                  max_leaf_nodes = None,
                                  random_state = None,
                                  min_impurity_split = 0.5,
                                  presort = False)

classifier.TreeKFoldReport(df,"report/", clf)