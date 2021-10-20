import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.classification import classification_report
from keras.models import load_model
from sklearn.metrics import plot_confusion_matrix
matplotlib.rcParams.update({'font.size': 16})


# ----------- data extraction and splitting ----------------------


def get_instances(data, split_train_dev=True, proportion_train=0.9, proportion_dev=0.1):
    train_X, train_y = data.train_instances()
    if split_train_dev:
        return split(train_X, train_y, proportion_train, proportion_dev)
    else:
        test_X, test_y = data.test_instances()
        return train_X, train_y, test_X, test_y


def split(train_X, train_y, proportion_train, proportion_dev, shuffle=True):
    """splits training data in training/dev data

    :param proportion_train: proportion of training data to extract as training data
    :param proportion_dev: proportion of training data to extract as dev data
    :param shuffle: shuffles data prior to splitting
    :return: X_train, y_train, X_dev, y_dev
    """
    if proportion_train + proportion_dev > 1:
        raise ValueError("proportions of training and dev data may not go beyond 1")
    nb_total = int(round(train_X.shape[0] * (proportion_train + proportion_dev)))
    nb_train = int(round(train_X.shape[0] * proportion_train))
    df = pd.DataFrame({'X': train_X, 'y': train_y})
    if shuffle:
        df = df.sample(n=nb_total, random_state=1)
    df_train = df[:nb_train]
    df_dev = df[nb_train:]
    return df_train['X'].values, df_train['y'].values, df_dev['X'].values, df_dev['y'].values


# ----------- grid search ----------------------

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def grid_search(pipeline, parameters, train_X, train_y, test_X):
    grid_search = GridSearchCV(pipeline, parameters, verbose=1)
    grid_search.fit(train_X, train_y)
    print("best params: {}".format(grid_search.best_params_))
    report(grid_search.cv_results_, n_top=10)
    return grid_search.predict(test_X)


# ----------- evaluation --------------------------

def eval(test_y, sys_y):
    return classification_report(test_y, sys_y)


# ----------- saving / loading --------------------

def write_data_to_disk(train_X, train_y, test_X, test_y, dir):
    with open(dir + 'train_X.pkl', 'wb') as f:
        pickle.dump(train_X, f)
    with open(dir + 'train_y.pkl', 'wb') as f:
        pickle.dump(train_y, f)
    with open(dir + 'test_X.pkl', 'wb') as f:
        pickle.dump(test_X, f)
    with open(dir + 'test_y.pkl', 'wb') as f:
        pickle.dump(test_y, f)


def load_data(dir):
    with open(dir + 'train_X.pkl', 'rb') as f:
        train_X = pickle.load(f)
    with open(dir + 'train_y.pkl', 'rb') as f:
        train_y = pickle.load(f)
    with open(dir + 'test_X.pkl', 'rb') as f:
        test_X = pickle.load(f)
    with open(dir + 'test_y.pkl', 'rb') as f:
        test_y = pickle.load(f)
    return train_X, train_y, test_X, test_y


def write_keras_model_to_disk(model, dir):
    model.save(dir + "model.h5")


def load_keras_model(dir):
    model = load_model(dir + 'model.h5')
    model.summary()
    return model

# --------------reporting predictions per document --------------------


def print_all_predictions(test_X, test_y, sys_y, logger):
    logger.info("\npred\tgold\ttweet\n----\t----\t----")
    for i in range(0, len(sys_y)):
        to_print = "{}\t{}\t{}".format(sys_y[i], test_y.values[i], test_X[i])
        logger.info(to_print)

def print_error_analysis(clf,test_X, test_y, sys_y, logger):
    plot_confusion_matrix(clf,test_X,test_y,normalize="true",cmap="Blues")
    plt.tight_layout()
    plt.show()

def important_features_per_class(vectorizer,classifier,n=80):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    for i in range(0, len(class_labels)):
        topn_class = sorted(zip(classifier.feature_count_[0], feature_names),reverse=True)[:n]
        print('\nImportant features in Class \'{}\''.format(class_labels[i]))
        for coef, feat in topn_class:
            print(class_labels[i], coef, feat)

def important_features_per_class_SVM(vectorizer,classifier,n=80):
    names = vectorizer.get_feature_names()
    [imp] = classifier.coef_
    topn_class = sorted(zip(imp,names),reverse=True)
    for coef, feat in topn_class:
        print(coef, feat)

    # imp,names = zip(*sorted(zip(imp,names)))
    # imp = imp[-15:]
    # names = names[-15:]
    # plt.barh(range(len(names)), imp, align='center')
    # plt.yticks(range(len(names)), names)
    # plt.show()


# -------------- polarity lexicons -------------------

def hate_lexicon():
    Dlex = {}
    lex_name = "resources/hatebase_dict_vua_format.csv"
    df1 = pd.read_csv(lex_name, sep=";")
    #print("\ndataset:{}\tnr of rows:{}\tnr of columns:{}".format(lex_name, df1.shape[0], df1.shape[1]))

    for i, row in df1.iterrows():
        entry = df1.loc[i]['Entry']
        pos = df1.loc[i]['Pos']
        label = df1.loc[i]['Label']
        Dlex[entry] = {}
        Dlex[entry]["label"] = label
        Dlex[entry]["pos"] = pos
    return Dlex

def lexobj2():
    Dlex={}
    lex_name="resources/subj_clues_vua_format.csv"
    df1=pd.read_csv(lex_name,sep=";")
    #print("\ndataset:{}\tnr of rows:{}\tnr of columns:{}".format(lex_name, df1.shape[0], df1.shape[1]))

    for i, row in df1.iterrows():
        entry = df1.loc[i]['Entry']
        pos = df1.loc[i]['Pos']
        label = df1.loc[i]['Label']
        Dlex[entry]={}
        Dlex[entry]["label"]=label
        Dlex[entry]["pos"] = pos
    return Dlex