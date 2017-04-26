import numpy as np
import pandas as pd

def autoTPOT(df):
    labels = list(df["class"].values)
    features = []
    for j in range(df.shape[0]):
        item = df.ix[j]
        features.append([item[i] for i in range(len(item) - 1)])

    features = np.array(features)
    labels = np.array(labels)

    labelsd, uniques = pd.factorize(df["class"])

    from sklearn.model_selection import train_test_split

    features_train, features_test, labels_train, labels_test = train_test_split(features, labelsd, test_size=0.2)

    from tpot import TPOTClassifier
    pipeline_optimizer = TPOTClassifier(generations=5, population_size=3, verbosity=3)

    pipeline_optimizer.fit(features_train, labels_train)
    print(pipeline_optimizer.score(features_test, labels_test))
    pipeline_optimizer.export('tpot_exported_pipeline3.py')