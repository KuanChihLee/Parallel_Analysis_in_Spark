# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np
from operator import add
import ParallelRegression as PR
import numpy as np
import csv
from pyspark import SparkContext


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Ridge Regression with Multi-Lambda',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--traindata', default=None, help='Input file containing (x,y) pairs, used to train a linear model')
    parser.add_argument('--testdata', default=None, help='Input file containing (x,y) pairs, used to test a linear model')
    parser.add_argument('--beta', default='beta', help='File where best beta is stored after testing')
    parser.add_argument('--listLam', type=float, nargs='+', help='Regularization parameter lambda List', required=True)
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.01, help='eps-tolerance. If the l2_norm gradient is smaller than eps, gradient descent terminates.') 
    parser.add_argument('--N', type=int, default=2, help='Level of parallelism')
    parser.add_argument('--mseCSV', default='mseCSV', help='File where lambda, MSE in training, and MSE in testing is stored')

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    args = parser.parse_args()
  
    sc = SparkContext(appName='Parallel Ridge Regression with Multi-Lambda')
    
    if not args.verbose :
        sc.setLogLevel("ERROR")
    
    lambdas = args.listLam

    print 'Reading training data from',args.traindata
    traindata = PR.readData(args.traindata,sc)
    traindata = traindata.repartition(args.N).cache()
    print 'Reading test data from',args.testdata
    testdata = PR.readData(args.testdata,sc)
    testdata = testdata.repartition(args.N).cache()
            
    minMSE = 100000
    bestBeta = None
    bestLam = 0.0
    csvData = list()
    MSEListTrain = list()
    MSEListTest = list()
    for lam in lambdas:
        x,y = traindata.take(1)[0]
        if args.traindata is not None:
            # Train a linear model beta from data with regularization parameter lambda
            beta0 = np.zeros(len(x))
	    print 'Training on data from',args.traindata,'with λ =',lam,', ε =',args.eps,', max iter = ',args.max_iter
            beta, gradNorm, k = PR.train(traindata, beta_0=beta0, lam=lam, max_iter=args.max_iter, eps=args.eps)
            #beta, gradNorm, k = PR.trainEstimate(traindata, beta_0=beta0, lam=lam, max_iter=args.max_iter, eps=args.eps)
            MSE = PR.test(traindata, beta)
            MSEListTrain.append(MSE)
	    print 'Converged:',gradNorm<args.eps
        
        if args.testdata is not None:
	    #print 'Computing MSE on data',args.testdata
            MSE = PR.test(testdata, beta)
            MSEListTest.append(MSE)
            #print 'MSE is:', MSE 
            if MSE < minMSE:
                minMSE = MSE
                bestBeta = beta
                bestLam = lam
                print 'Current Min MSE: ', minMSE
    print 'Best Lambda: ', bestLam
    print 'Saving best trained β in', args.beta
    PR.writeBeta(args.beta, bestBeta)
    
    # Write Lambda, MSE in train, MSE in test to file
    for i in range(len(lambdas)):
        storeMSE = [lambdas[i], MSEListTrain[i], MSEListTest[i]]
        csvData.append(storeMSE)
    with open(args.mseCSV, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()
