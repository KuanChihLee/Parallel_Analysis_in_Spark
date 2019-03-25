import numpy as np
import argparse
from time import time
import csv
import LogisticRegression as LR
import ParallelLogisticRegression as PLR

from SparseVector import SparseVector
from pyspark import SparkContext, SparkConf


def CreateParser(description):
    parser = argparse.ArgumentParser(description = description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('traindata', default=None, help='Input file containing (x,y) pairs, used to train a logistic model')
    parser.add_argument('--testdata', default=None, help='Input file containing (x,y) pairs, used to test a logistic model')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float, default=0.0, help='Regularization parameter lambda')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.1, help='eps-tolerance. If the l2_norm gradient is smaller than eps, gradient descent terminates.')
    
    parser.add_argument('--LR', default=None, help='Use LogisticRegression pyscript or not')
    parser.add_argument('--plotCSV', default='plotCSV', help='File where gradient norm, accuracy, precision, recall and time are stored')
    parser.add_argument('--lamList', type=float, nargs='+', help='Regularization parameter lambda List')
    parser.add_argument('--accCSV', default='accCSV', help='File where lambda and accuracy in testing is stored')
    return parser.parse_args()
    
def SetLogger(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)
    
def CreateSparkContext(appName):
    sparkConf = SparkConf()\
                    .setAppName(appName)\
                    .set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)   
    SetLogger(sc)
    return (sc)
    
def writeBeta(output, beta):
    """    write best beta into a file
    Input- output directionay
           beta
    Output- a file with best beta
    """
    with open(output,'w') as fh:
        for (key, value) in beta:
            fh.write('(%s,%f)\n' % (key, value))

if __name__ == "__main__":
    args = CreateParser(description='Evaluator')
    sc = CreateSparkContext(appName='Evaluator Script')
    
    print '----------------Start Spark Training------------------------'
    print 'Reading training data from', args.traindata
    traindataRDD = PLR.readDataRDD(args.traindata, sc).repartition(100).cache()
    print 'Read', traindataRDD.keys().count(), 'data points with', len(PLR.getAllFeaturesRDD(traindataRDD)), 'features in total'
    
    if args.testdata is not None:
        print 'Reading test data from', args.testdata
        testdataRDD = PLR.readDataRDD(args.testdata, sc).repartition(100).cache()
        print 'Read', testdataRDD.keys().count(), 'data points with', len(PLR.getAllFeaturesRDD(testdataRDD)), 'features'
    else:
        testdata = None
        
    if not args.lamList:
        beta0 = SparseVector({})
        print 'Training on data from', args.traindata, 'with lam =', args.lam, ', eps =', args.eps, ', max iter = ', args.max_iter
        betaRDD, gradNormRDD, k, accRDD, preRDD, recRDD, timeRDD = PLR.train(traindataRDD, beta_0 = beta0, \
                                                                        lam = args.lam, max_iter = args.max_iter, eps = args.eps, test_data = testdataRDD) 
    else:
        bestAcc = -1
        bestLam = -1
        bestIter = -1
	csvData = list()
	bestBeta = list()
        for lam in args.lamList:
            beta0 = SparseVector({})
            print 'Training on data from', args.traindata, 'with lam =', lam, ', eps =', args.eps, ', max iter = ', args.max_iter
            betaRDD, gradNormRDD, k, accRDD, preRDD, recRDD, timeRDD = PLR.train(traindataRDD, beta_0 = beta0, \
                                                                        lam = lam, max_iter = args.max_iter, eps = args.eps, test_data = testdataRDD)
            ## Question b
            for i in range(args.max_iter):
                storePlot = [gradNormRDD[i], accRDD[i], preRDD[i], recRDD[i], timeRDD[i]]
                csvData.append(storePlot)    

            ## Question c   
            if bestAcc < max(accRDD):
                bestLam = lam
                bestAcc = max(accRDD)
                bestIter = np.argmax(accRDD) + 1
                bestBeta = sorted(betaRDD.items(), key=lambda x:x[1], reverse=True)
	    print 'Historical Best Lambda: ', bestLam, ' Iteration: ', bestIter
	print 'Best Accuracy: ', bestAcc, 'happens with Best Lambda: ', bestLam, 'in ', bestIter, ' iterations'
	
	## Question b
        print 'Saving Accuracy for different Lambda'
        with open(args.accCSV, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)
        csvFile.close()	
	## Question c 
	print 'Saving trained beta in', args.beta
        writeBeta(args.beta, bestBeta)
	
            
    if args.LR is not None:
        print '----------------Start Python Training------------------------'
        print 'Reading training data from', args.traindata
        traindata = LR.readData(args.traindata)
        print 'Read', len(traindata), 'data points with', len(LR.getAllFeatures(traindata)), 'features in total'
        
        if args.testdata is not None:
            print 'Reading test data from',args.testdata
            testdata = LR.readData(args.testdata)
            print 'Read', len(testdata), 'data points with', len(LR.getAllFeatures(testdata)), 'features'
        else:
            testdata = None
    
        beta0 = SparseVector({})
        print 'Training on data from', args.traindata, 'with lam =', args.lam, ', eps =', args.eps, ', max iter = ', args.max_iter
        beta, gradNorm, k, acc, pre, rec, timelist = LR.train(traindata, beta_0 = beta0, lam = args.lam, max_iter = args.max_iter, eps = args.eps, test_data = testdata)
    
    ## Question a
    print 'Saving the comparison between Spark code and Python code' 
    csvData = list()
    for i in range(args.max_iter):
        storePlot = [gradNormRDD[i], accRDD[i], preRDD[i], recRDD[i], timeRDD[i]]
        csvData.append(storePlot)
    if args.LR is not None:
        for i in range(args.max_iter):
            storePlot = [gradNorm[i], acc[i], pre[i], rec[i], timelist[i]]
            csvData.append(storePlot)
    with open(args.plotCSV, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()
        
