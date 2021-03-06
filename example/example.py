from mrsqm import MrSQMClassifier
from sklearn import metrics
import util

import logging
import sys
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def basic_example():

    X_train,y_train = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TRAIN.arff")
    X_test,y_test = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TEST.arff")

    # train MrSQM-R with SAX only
    clf = MrSQMClassifier(xrep=4)
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    print("MrSQM-R accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

 

    

def mrsqm_with_sfa():
    X_train,y_train = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TRAIN.arff")
    X_test,y_test = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TEST.arff")

   

    # train MrSQM-RS with SFA only
    clf = MrSQMClassifier(use_sax = False, use_sfa = True) # train MrSQM-RS
    clf.fit(X_train,y_train) # use ext_rep to add sfa transform
    predicted = clf.predict(X_test) # use ext_rep to add sfa transform    
    print("MrSQM-RS with SFA only accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

    # train MrSQM-RS with both SAX and SFA
    clf = MrSQMClassifier(use_sax = True, use_sfa = True) # train MrSQM-RS
    clf.fit(X_train,y_train) # use ext_rep to add sfa transform
    predicted = clf.predict(X_test) # use ext_rep to add sfa transform    
    print("MrSQM-RS with both SAX and SFA accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

def test_sfa():
    from sktime.utils.data_processing import from_nested_to_2d_array
    from mrsqm import PySFA

    X_train,y_train = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TRAIN.arff")
    X_test,y_test = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TEST.arff")
    y_train = [float(l) for l in y_train]
    y_test = [float(l) for l in y_test]
    X_train = from_nested_to_2d_array(X_train).values
    X_test = from_nested_to_2d_array(X_test).values

    
    
    sfa = PySFA(8,4,4,True)
    sfa.fit(X_train,y_train)
    print(sfa.transform2n(X_test,y_test))

if __name__ == "__main__":
    # basic_example()
    basic_example() # require running jar file