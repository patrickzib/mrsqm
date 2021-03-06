from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

import numpy as np
import pandas as pd
from numpy.random import randint
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sktime.utils.data_processing import from_nested_to_2d_array

import logging

def debug_logging(message):
    logging.info(message)




######################### SAX #########################

cdef extern from "sax_converter.h":
    cdef cppclass SAX:
        SAX(int, int, int, int)        
        vector[string] timeseries2SAX(vector[double])
        vector[double] map_weighted_patterns(vector[double], vector[string], vector[double])

cdef class PySAX:
    '''
    Wrapper of SAX C++ implementation.
    '''
    cdef SAX * thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, int N, int w, int a, int di = 1):
        self.thisptr = new SAX(N, w, a, di)

    def __dealloc__(self):
        del self.thisptr

    def timeseries2SAX(self, ts):
        return self.thisptr.timeseries2SAX(ts)
        

    def timeseries2SAXseq(self, ts):
        words = self.thisptr.timeseries2SAX(ts)
        seq = b''
        
        for w in words:
            seq = seq + b' ' + w
        if seq:  # remove extra space
            seq = seq[1:]
        return seq

    def map_weighted_patterns(self, ts, sequences, weights):
        return self.thisptr.map_weighted_patterns(ts, sequences, weights)

###########################################################################

cdef extern from "sfa/SFAWrapper.cpp":
    cdef cppclass SFAWrapper:
        SFAWrapper(int, int, int, bool)        
        void fit(vector[vector[double]], vector[double])
        vector[string] transform(vector[vector[double]], vector[double])
        vector[vector[vector[int]]] transform2n(vector[vector[double]], vector[double])
    cdef void printHello()

cdef class PySFA:
    '''
    Wrapper of SFA C++ implementation.
    '''
    cdef SFAWrapper * thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, int N, int w, int a, bool norm):
        self.thisptr = new SFAWrapper(N, w, a, norm)

    def __dealloc__(self):
        del self.thisptr

    def fit(self, X, y):
        self.thisptr.fit(X, y)
        return self

    def transform(self, X, y):
        return self.thisptr.transform(X,y)

    def transform2n(self, X, y):
        return self.thisptr.transform2n(X,y)



#########################SQM wrapper#########################


cdef extern from "strie.cpp":
    cdef cppclass SeqTrie:
        SeqTrie(vector[string])
        vector[int] search(string)


cdef extern from "sqminer.h":
    cdef cppclass SQMiner:
        SQMiner(double, double)
        vector[string] mine(vector[string] &, vector[int] &)

cdef class PyFeatureTrie:
    cdef SeqTrie *thisptr

    def __cinit__(self, vector[string] sequences):
        self.thisptr = new SeqTrie(sequences)
    def __dealloc__(self):
        del self.thisptr

    def search(self, string sequence):
        return self.thisptr.search(sequence)


cdef class PySQM:
    cdef SQMiner *thisptr

    def __cinit__(self, double selection, double threshold):
        self.thisptr = new SQMiner(selection,threshold)
    def __dealloc__(self):
        del self.thisptr

    def mine(self, vector[string] sequences, vector[int] labels):
        return self.thisptr.mine(sequences, labels)     


######################### MrSQM Classifier #########################

class MrSQMClassifier:    
    '''     
    Overview: MrSQM is an efficient time series classifier utilizing symbolic representations of time series. MrSQM implements four different feature selection strategies (R,S,RS,SR) that can quickly select subsequences from multiple symbolic representations of time series data.
    
    Parameters
    ----------
    
    strat               : str, feature selection strategy, either 'R','S','SR', or 'RS'. R and S are single-stage filters while RS and SR are two-stage filters.
    
    use_sax             : bool, whether to use the sax transformation. if False, ext_rep must be provided in the fitting and predicting stage.
    
    custom_config       : dict, customized parameters for the symbolic transformation.

    features_per_rep    : int, (maximum) number of features selected per representation.

    selection_per_rep   : int, (maximum) number of candidate features selected per representation. Only applied in two stages strategies (RS and SR)

    xrep                : int, control the number of representations produced by sax transformation.

    '''

    def __init__(self, strat = 'SR', features_per_rep = 100, selection_per_rep = 200, use_sax = False, use_sfa = True, custom_config=None, xrep = 1):

        self.use_sax = use_sax
        self.use_sfa = use_sfa

        if custom_config is None:
            self.config = [] # http://effbot.org/zone/default-values.htm
        else:
            self.config = custom_config

        self.strat = strat   

        # all the unique labels in the data
        # in case of binary data the first one is always the negative class
        self.classes_ = []
        self.clf = None # scikit-learn model       

        self.fpr = features_per_rep
        self.spr = selection_per_rep
        self.xrep = xrep  
        self.filters = [] # feature filters (one filter for a rep) for test data transformation

        debug_logging("Initialize MrSQM Classifier.")
        debug_logging("Feature Selection Strategy: " + strat)
        debug_logging("Mode: " + str(self.xrep))
        debug_logging("Number of features per rep: " + str(self.fpr))
        debug_logging("Number of candidates per rep (only for SR and RS):" + str(self.spr))

        self.max_alphabet = [b'!', b'"', b'#', b'$', b'%', b'&', b"'", b'(', b')', b'*', b'+', b',', b'-', b'.', b'/', b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b':', b';', b'<', b'=', b'>', b'?', b'@', b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O', b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W', b'X', b'Y', b'Z', b'[', b'\\', b']', b'^', b'_', b'`', b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o', b'p', b'q', b'r', b's', b't', b'u', b'v', b'w', b'x', b'y', b'z', b'{', b'|', b'}', b'~']
        self.sfa_start = 0
        self.sax_start = 64
        
     

    def create_pars(self, min_ws, max_ws, random_sampling=False):
        pars = []            
        if random_sampling:    
            debug_logging("Sampling window size, word length, and alphabet size.")       
            ws_choices = [int(2**(w/self.xrep)) for w in range(3*self.xrep,self.xrep*int(np.log2(max_ws))+ 1)]            
            wl_choices = [6,7,8,9,10]
            alphabet_choices = [4]
            for w in range(3*self.xrep,self.xrep*int(np.log2(max_ws))+ 1):
                pars.append([np.random.choice(ws_choices) , np.random.choice(wl_choices), np.random.choice(alphabet_choices)])
        else:
            debug_logging("Doubling the window while fixing word length and alphabet size.")                   
            pars = [[int(2**(w/self.xrep)),8,4] for w in range(3*self.xrep,self.xrep*int(np.log2(max_ws))+ 1)]     

        debug_logging("Symbolic Parameters: " + str(pars))      
            
        
        return pars            
            
            

    def transform_time_series(self, ts_x, y):
        debug_logging("Transform time series to symbolic representations.")
        
        multi_tssr = []   

        ts_x_array = from_nested_to_2d_array(ts_x).values
        
     
        if not self.config:
            self.config = []
            
            min_ws = 16
            min_len = max_len = len(ts_x.iloc[0, 0])
            for a in ts_x.iloc[:, 0]:
                min_len = min(min_len, len(a)) 
                max_len = max(max_len, len(a))
            max_ws = (min_len + max_len)//2            
            
            if self.use_sax:
                pars = self.create_pars(min_ws, max_ws, True)
                for p in pars:
                    features = self.random_sequences(p[1],p[2])
                    self.config.append(
                        {'method': 'sax', 'window': p[0], 'word': p[1], 'alphabet': p[2], 'features': features ,
                        # 'dilation': np.int32(2 ** np.random.uniform(0, np.log2((min_len - 1) / (p[0] - 1))))})
                        'dilation': 1})
                    
            if self.use_sfa:
                pars = self.create_pars(min_ws, max_ws, True)
                for p in pars:
                    features = self.random_sequences(p[1],p[2])
                    self.config.append(
                        {'method': 'sfa', 'window': p[0], 'word': p[1], 'alphabet': p[2], 'features': features
                        })        

        
        for cfg in self.config:
            for i in range(ts_x.shape[1]):
                tssr = []

                if cfg['method'] == 'sax':  # convert time series to SAX                    
                    ps = PySAX(cfg['window'], cfg['word'], cfg['alphabet'], cfg['dilation'])
                    for ts in ts_x.iloc[:,i]:
                        sr = ps.timeseries2SAXseq(ts)
                        tssr.append(sr)
                elif  cfg['method'] == 'sfa':
                    if 'signature' not in cfg:
                        cfg['signature'] = PySFA(cfg['window'], cfg['word'], cfg['alphabet'], True).fit(ts_x_array,y)
                    
                    tssr = np.array(cfg['signature'].transform2n(ts_x_array,y))
                multi_tssr.append(np.array(tssr))        

        return multi_tssr
  


    def random_sequences(self, seq_length, alphabet_size):

        output = np.zeros((self.fpr,seq_length))
        alphabet = [i for i in range(0,alphabet_size)]
        for i in range(self.fpr):
            output[i,:] = np.random.choice(alphabet,seq_length)
            
        return np.unique(output, axis = 0)


    
    def feature_transform(self, mr_seqs):
        fm = []
        for rep, cfg in zip(mr_seqs, self.config):
            threshold = cfg["word"]
            for ft in cfg["features"]:
                #print(rep)
                distances = np.sum(np.abs(rep - ft),axis=2)
                ft_output = np.sum(distances < threshold,axis=1)
                fm.append(ft_output)        
        return np.array(fm).T






   





    def fit(self, X, y, ext_rep = None):
        debug_logging("Fit training data.")
        self.classes_ = np.unique(y) #because sklearn also uses np.unique

        int_y = [np.where(self.classes_ == c)[0][0] for c in y]

        self.sequences = []

        debug_logging("Search for subsequences.")
        mr_seqs = []        
        mr_seqs = self.transform_time_series(X,y)     
    


    
        # first computing the feature vectors
        # then fit the new data to a logistic regression model
        
        debug_logging("Compute feature vectors.")
        train_x = self.feature_transform(mr_seqs)
        
        debug_logging("Fit logistic regression model.")
        #print(train_x)
        self.clf = LogisticRegression(solver='newton-cg',multi_class = 'multinomial', class_weight='balanced').fit(train_x, y)        
        self.classes_ = self.clf.classes_ # shouldn't matter       
    
    def transform_test_X(self, X, ext_rep = None):
        mr_seqs = []        
        y = np.random.choice([-1.0,1.0], X.shape[0])
        mr_seqs = self.transform_time_series(X,y)        

        return self.feature_transform(mr_seqs)

    def predict_proba(self, X):        
        test_x = self.transform_test_X(X)
        return self.clf.predict_proba(test_x) 

    def predict(self, X):
        test_x = self.transform_test_X(X)
        return self.clf.predict(test_x)





 






