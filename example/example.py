from mrsqm import MrSQMClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import util
#from sktime.datasets import load_arrow_head, load_basic_motions

import logging
import sys
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def basic_example():

    X_train,y_train = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TRAIN.arff")
    X_test,y_test = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TEST.arff")

    # train MrSQM-R with SAX only
    clf = MrSQMClassifier(strat = 'R')
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    print("MrSQM-R accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

    # train MrSQM-RS with SAX only
    clf = MrSQMClassifier(strat = 'RS') # default 
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    print("MrSQM-RS accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

    # train MrSQM-S with SAX only
    clf = MrSQMClassifier(strat = 'S')
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    print("MrSQM-S accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

    # train MrSQM-SR with SAX only
    clf = MrSQMClassifier(strat = 'SR')
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    print("MrSQM-SR accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

    

def mrsqm_with_sfa():
    X_train,y_train = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TRAIN.arff")
    X_test,y_test = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TEST.arff")

    # SFA transform
    # since the sfa python implementation is significantly slow, we opt for the java version https://github.com/patrickzib/SFA
    itrain = "data/Coffee/Coffee_TRAIN.txt" # txt and arff data are identical, however, the sfa java implementation can only read txt
    itest = "data/Coffee/Coffee_TEST.txt"
    otrain = 'sfa.train'
    otest = 'sfa.test'

    # run a jar file to do sfa transform
    util.sfa_transform(itrain, itest, otrain, otest)

    # train MrSQM-RS with SFA only
    clf = MrSQMClassifier(use_sax = False) # train MrSQM-RS
    clf.fit(X_train,y_train, ext_rep = otrain) # use ext_rep to add sfa transform
    predicted = clf.predict(X_test, ext_rep = otest) # use ext_rep to add sfa transform
    print("MrSQM-RS with SFA only accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

    # train MrSQM-RS with both SAX and SFA
    clf = MrSQMClassifier() # train MrSQM-RS
    clf.fit(X_train,y_train, ext_rep = otrain) # use ext_rep to add sfa transform
    predicted = clf.predict(X_test, ext_rep = otest) # use ext_rep to add sfa transform
    print("MrSQM-RS with both SAX and SFA accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

def multivariate_mrsqm():
    X, y = load_basic_motions(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # train MrSQM-R with SAX only
    clf = MrSQMClassifier(strat = 'R')
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    print("MrSQM-R accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

def multivariate_mrsqm_with_sfa():
    itrain = "data/BasicMotions/BasicMotions_TRAIN.arff"
    itest = "data/BasicMotions/BasicMotions_TEST.arff"

    X_train,y_train = util.load_from_arff_to_dataframe(itrain)
    X_test,y_test = util.load_from_arff_to_dataframe(itest)

    # SFA transform
    # since the sfa python implementation is significantly slow, we opt for the java version https://github.com/patrickzib/SFA
    
    otrain = 'sfa.train'
    otest = 'sfa.test'

    # run a jar file to do sfa transform
    util.mvts_sfa_transform(itrain, itest, otrain, otest)

    # train MrSQM-RS with SFA only
    clf = MrSQMClassifier(use_sax = False) # train MrSQM-RS
    clf.fit(X_train,y_train, ext_rep = otrain) # use ext_rep to add sfa transform
    predicted = clf.predict(X_test, ext_rep = otest) # use ext_rep to add sfa transform
    print("MrSQM-RS with SFA only accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

    # train MrSQM-RS with both SAX and SFA
    clf = MrSQMClassifier() # train MrSQM-RS
    clf.fit(X_train,y_train, ext_rep = otrain) # use ext_rep to add sfa transform
    predicted = clf.predict(X_test, ext_rep = otest) # use ext_rep to add sfa transform
    print("MrSQM-RS with both SAX and SFA accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

if __name__ == "__main__":
    multivariate_mrsqm_with_sfa()
    # mrsqm_with_sfa() # require running jar file