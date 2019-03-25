# -*- coding: utf-8 -*-
import numpy as np
import argparse
from time import time
from SparseVector import SparseVector
from LogisticRegression import readBeta, writeBeta, gradLogisticLoss, logisticLoss, lineSearch
from operator import add
from pyspark import SparkContext

def readDataRDD(input_file, spark_context):
    """  Read data from an input file. Each line of the file contains tuples of the form

                    (x,y)  

         x is a dictionary of the form:                 

           { "feature1": value, "feature2":value, ...}

         and y is a binary value +1 or -1.

         The return value is an RDD containing tuples of the form
                 (SparseVector(x),y)             

    """ 
    return spark_context.textFile(input_file)\
                        .map(eval)\
                        .map(lambda (x,y):(SparseVector(x),y))

def getAllFeaturesRDD(dataRDD):                
    """ Get all the features present in grouped dataset dataRDD.
 
	  The input is:
        - dataRDD containing pairs of the form (SparseVector(x),y).  

    The return value is an RDD containing the union of all unique features present in sparse vectors inside dataRDD.
    """                
    return dataRDD.map(lambda x:x[0]).reduce(lambda x,y: x+y).keys() 

def totalLossRDD(dataRDD, beta, lam = 0.0):
    """  Get total logistic loss in grouped dataset dataRDD:
    
    The input is:
        - dataRDD containing pairs of the form (SparseVector(x),y).
        
    Function:
        - each instance (SparseVector(x),y) is mapped with logisticLoss() function and then reduced into single output value
     
    The output/return is a value of total logistic loss.   
    """
    return dataRDD.map(lambda (x,y): logisticLoss(beta, x, y)).reduce(lambda x,y: x+y) + lam * beta.dot(beta)

def gradTotalLossRDD(dataRDD, beta, lam = 0.0):
    """  Get total gradient logistic loss in grouped dataset dataRDD:
    
    The input is:
        - dataRDD containing pairs of the form (SparseVector(x),y).
        
    Function:
        - each instance (SparseVector(x),y) is mapped with gradLogisticLoss() function and then reduced into a SparseVector 
     
    The output/return is a SparseVector of total gradient logistic loss.   
    """
    return dataRDD.map(lambda (x,y): gradLogisticLoss(beta, x, y)).reduce(lambda x,y: x+y) + 2 * lam * beta 

def decisionMaker(x, y):
    """  Get TP, FP, FN and TN number for each binary instance: 
    
    The input is:
        - x and y should be binary, 1 or -1
        
    Function:
        - Encoded original instance with four-quadrant law 
          (referring to Wikipedia, https://en.wikipedia.org/wiki/Precision_and_recall)
          TP- [1, 0, 0, 0], FP- [0, 1, 0, 0], FN- [0, 0, 1, 0], TN- [0, 0, 0, 1]   
    
    The output/return is encoded value.
    """
    if x == y:
       if x == 1:
           return [1, 0, 0, 0]
       else:
           return [0, 0, 0, 1]
    else:
       if x == 1:
           return [0, 1, 0, 0]
       else:
           return [0, 0, 1, 0]
          
def test(dataRDD, beta):
    """   Output the quantities necessary to compute the accuracy, precision, and recall of the prediction of labels in a dataset under a given RDD
        
    The accuracy (ACC), precision (PRE), and recall (REC) are defined in terms of the following sets:

          P = datapoints (x,y) in data for which <beta,x> > 0
          N = datapoints (x,y) in data for which <beta,x> <= 0
                 
          TP = datapoints in (x,y) in P for which y=+1  
          FP = datapoints in (x,y) in P for which y=-1  
          TN = datapoints in (x,y) in N for which y=-1
          FN = datapoints in (x,y) in N for which y=+1

    For #XXX the number of elements in set XXX, the accuracy, precision, and recall of parameter vector beta over data are defined as:
         
          ACC(beta,data) = ( #TP+#TN ) / (#P + #N)
          PRE(beta,data) = #TP / (#TP + #FP)
          REC(beta,data) = #TP/ (#TP + #FN)

    Inputs are:
          - data: an RDD containing pairs of the form (x,y)
          - beta: vector beta

    Function:
          - decisionMaker() will return encoded result of TP, FP, TN and FN, like (1, 0, 0, 0) for TP case
          - result is a list of total numbers of TP, FP, TN and FN, individually.

    The return values are
          - ACC, PRE, REC
    
    
    """
    result = dataRDD.map(lambda (x,y): (1,y) if beta.dot(x)>0 else (-1,y))\
                    .map(lambda (x,y): decisionMaker(x,y))\
                    .reduce(lambda x,y: [x[i]+y[i] for i in range(len(x))])
    return 1.0*(result[0]+result[3])/sum(result), 1.0*result[0]/(result[0]+result[1]), 1.0*result[0]/(result[0]+result[2])

def train(dataRDD, beta_0, lam, max_iter, eps, test_data=None):
    k = 0
    gradNorm = 2*eps
    beta = beta_0
    start = time()
    accList = []
    preList = []
    recList = []
    gradNormList = []
    timeList = []
    while k < max_iter and gradNorm > eps:
        obj = totalLossRDD(dataRDD, beta, lam)   

        grad = gradTotalLossRDD(dataRDD, beta, lam)  
        gradNormSq = grad.dot(grad)
        gradNorm = np.sqrt(gradNormSq)

        fun = lambda x: totalLossRDD(dataRDD, x, lam)
        gamma = lineSearch(fun, beta, grad, obj, gradNormSq)
        
        beta -= gamma * grad
        if test_data == None:
            print 'k = ',k,'\tt = ',time()-start,'\tL(beta_k) = ',obj,'\t||L\'(beta_k)||_2 = ',gradNorm,'\tgamma = ',gamma
        else:
            acc, pre, rec = test(test_data,beta)
            accList.append(acc)
            preList.append(pre)
            recList.append(rec)
            print 'k = ',k,'\tt = ',time()-start,'\tL(beta_k) = ',obj,'\t||L\'(beta_k)||_2 = ',gradNorm,'\tgamma = ',gamma,'\tACC = ',acc,'\tPRE = ',pre,'\tREC = ',rec
        k = k + 1
        gradNormList.append(gradNorm)
        timeList.append(time()-start)

    return beta, gradNormList, k, accList, preList, recList, timeList

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Logistic Regression.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('traindata',default=None, help='Input file containing (x,y) pairs, used to train a logistic model')
    parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a logistic model')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter lambda')
    parser.add_argument('--max_iter', type=int,default=100, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.1, help='eps-tolerance. If the l2_norm gradient is smaller than eps, gradient descent terminates.')
    
    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    
    args = parser.parse_args()
    sc = SparkContext(appName='Parallel Logistic Regression')
    
    if not args.verbose :
        sc.setLogLevel("ERROR")

    print 'Reading training data from',args.traindata
    traindata = readDataRDD(args.traindata, sc).repartition(100).cache()
    print 'Read', traindata.keys().count(), 'data points with', len(getAllFeaturesRDD(traindata)), 'features in total'

    if args.testdata is not None:
        print 'Reading test data from',args.testdata
        testdata = readDataRDD(args.testdata, sc).repartition(100).cache()
        print 'Read', testdata.keys().count(), 'data points with', len(getAllFeaturesRDD(testdata)), 'features'
    else:
        testdata = None

    beta0 = SparseVector({})

    print 'Training on data from', args.traindata, 'with lam =', args.lam, ', eps =', args.eps, ', max iter = ', args.max_iter
    beta, gradNorm, k, _, _, _, _ = train(traindata, beta_0 = beta0, lam = args.lam, max_iter = args.max_iter, eps = args.eps, test_data = testdata) 
    print 'Algorithm ran for', k, 'iterations. Converged:', gradNorm[-1] < args.eps
    print 'Saving trained beta in',args.beta
    writeBeta(args.beta,beta)