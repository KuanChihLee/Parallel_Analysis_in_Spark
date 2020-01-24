# -*- coding: utf-8 -*-
import numpy as np
from time import time
import csv
import sys,argparse
from pyspark import SparkContext
from operator import add
from pyspark.mllib.random import RandomRDDs


def readRatings(file,sparkContext):
    return sparkContext.textFile(file, use_unicode=False).map(eval).map(lambda (bid,stars,uid,city,name,state): (uid,bid,stars))

def swap((x,y)):
    return (y,x)

def tuple_adding(pair1,pair2):
    """ Helper function that adds each elememt individually 
        return a tuple 
    """
    val11, val12 = pair1
    val21, val22 = pair2
    return (val11+val21, val12+val22)
   
def globalMean(R):
    ''' return scalar: global mean- the average rating
    '''
    x, num = R.map(lambda (i,j,x): (x,1)).reduce(tuple_adding)
    return  1.0 * x / num

def generateItemProfiles(R,d,seed,sparkContext,N):
    ''' Create Item Profiles
        j: item id
        vj: dimemsion d, which is latent dimension
        bv: the average rating given a particular item
        return (j, (vj,bv))
    '''
    V = R.map(lambda (i,j,x): (j,(x,1))).reduceByKey(tuple_adding, numPartitions = N).map(lambda (j,(x,num)): (j, x/num))
    numItems = V.count()
    randRDD = RandomRDDs.normalVectorRDD(sparkContext, numItems, d, numPartitions=N, seed=seed)
    V = V.zipWithIndex().map(swap).repartition(N)
    randRDD = randRDD.zipWithIndex().map(swap).repartition(N)
    return V.join(randRDD).map(lambda (idx, ((j,bv),vj)): (j, (vj,bv)))

def generateUserProfiles(R,d,seed,sparkContext,N):
    ''' Create User Profiles
        i: user id
        ui: dimemsion d, which is latent dimension
        bu: the average rating given a particular user
        Nu: total amount of items that a particulur user has seen 
        return (i, (ui,bu,1/sqrt(Nu)))
    '''
    U = R.map(lambda (i,j,x): (i,(x,1))).reduceByKey(tuple_adding, numPartitions = N).map(lambda (i,(x,num)): (i, x/num, num))
    numUsers = U.count()
    randRDD = RandomRDDs.normalVectorRDD(sparkContext, numUsers, d,numPartitions=N, seed=seed)
    U = U.zipWithIndex().map(swap).repartition(N)
    randRDD = randRDD.zipWithIndex().map(swap).repartition(N)
    return U.join(randRDD).map(lambda (idx, ((i,bu,num),ui)): (i, (ui,bu,1.0/np.sqrt(num))))

def generateImplicitProfiles(R,d,seed,sparkContext,N):
    ''' Create Implicit Profiles
        j: item id
        yj: dimemsion d, which is latent dimension
        return (j, yj)
    '''
    Y = R.map(lambda (i,j,x): j).distinct()
    numItems = Y.count()
    randRDD = RandomRDDs.normalVectorRDD(sparkContext, numItems, d, numPartitions=N, seed=seed)
    Y = Y.zipWithIndex().map(swap).repartition(N)
    randRDD = randRDD.zipWithIndex().map(swap).repartition(N)
    return Y.join(randRDD).values()     

def generateSumY(R,Y,N):
    ''' Create sum of implicit profile given a particular user
        Note: SumY is defined as sum of implicit vectors of the items that user has seen
        j: item id
        yj: dimemsion d, which is latent dimension
        return (j, yj)
    '''
    items = R.map(lambda (i,j,rij): (j,i)).repartition(N)
    return Y.join(items).map(lambda (j, (yj,i)): (i, (j,yj))) \
                        .reduceByKey(lambda x,y: (x[0],x[1]+y[1])) \
                        .map(lambda (i, (j,Yj)): (i,Yj))
            
def pred_diff(rij,vj,bu,bv,Zu,Rmean,N):
    return Rmean + (bu-Rmean) + (bv-Rmean) + vj.T.dot(Zu) - rij
     

def joinAndPredictAll(R,U,V,Y,M,Rmean,N):
    ''' return (i,j,ui,vj,bu,bv,rij,Nu,yj,Zu,delta)
    '''
    return R.join(U) \
            .join(M) \
            .map(lambda (i, (((rij,j),(ui,bu,Nu)),sumY)): (j, (i,ui,bu,rij,Nu,sumY))) \
            .join(V, numPartitions = N) \
            .join(Y) \
            .map(lambda (j, (((i,ui,bu,rij,Nu,sumY),(vj,bv)),yj)): (i,j,ui,vj,bu,bv,rij,Nu,yj,ui+Nu*sumY,pred_diff(rij,vj,bu,bv,ui+Nu*sumY,Rmean,N))) 
 
def custom_add_gradients_U(pair1,pair2):
    """ Return added gradients for U and bu, and Nu 
        return (ui,bu,grad_u,grad_bu,Nu) for given user id- i
    """
    ui1,bu1,grad_u1,grad_bu1,Nu1 =pair1
    ui2,bu2,grad_u2,grad_bu2,Nu2 =pair2
    return (ui1,bu1,grad_u1+grad_u2,grad_bu1+grad_bu2,Nu1)
    
def custom_add_gradients_V(pair1,pair2):
    """ Return added gradients for V and bv
        return (vj,bv,grad_v,grad_bv) for given item id- j
    """
    vj1,bv1,grad_v1,grad_bv1 =pair1
    vj2,bv2,grad_v2,grad_bv2 =pair2
    return (vj1,bv1,grad_v1+grad_v2,grad_bv1+grad_bv2)
    
def custom_add_gradients_y(pair1,pair2):
    """ Return added gradients for Y
        return (yj,grad_y) for given item id- j
    """
    y1, grad_y1 =pair1
    y2, grad_y2 =pair2
    return (y1, grad_y1+grad_y2)

def gradient_u(delta,v):
    ''' return d-dimension vector: gradient of square error given user
    '''
    return 2*delta*v
    
def gradient_bu(delta):
    ''' return scalar: gradient of square error given bu
    '''
    return 2*delta

def gradient_v(delta,Zu):
    ''' return d-dimension vector: gradient of square error given item
    '''
    return 2*delta*Zu

def gradient_bv(delta):
    ''' return scalar: gradient of square error given bv
    '''
    return 2*delta
    
def gradient_y(delta,Nu,v):
    ''' return scalar: gradient of square error given yj
    '''
    return 2*Nu*delta*v

def adaptU(joinedRDD,gamma,lam,N):
    ''' return updated user profile U
    '''
    return joinedRDD.map(lambda (i,j,ui,vj,bu,bv,rij,Nu,yj,Zu,delta): (i, (ui,bu,gradient_u(delta,vj),gradient_bu(delta),Nu))) \
                    .reduceByKey(custom_add_gradients_U, numPartitions = N) \
                    .mapValues(lambda (ui,bu,grad_u,grad_bu,Nu): (ui-gamma*(grad_u+2*lam*ui),bu-gamma*(grad_bu+2*lam*bu),Nu))
                    
def adaptV(joinedRDD,gamma,mu,N):
    ''' return updated item profile V
    '''
    return joinedRDD.map(lambda (i,j,ui,vj,bu,bv,rij,Nu,yj,Zu,delta): (j, (vj,bv,gradient_v(delta,Zu),gradient_bv(delta)))) \
                    .reduceByKey(custom_add_gradients_V, numPartitions = N) \
                    .mapValues(lambda (vj,bv,grad_v,grad_bv): (vj-gamma*(grad_v+2*mu*vj),bv-gamma*(grad_bv+2*mu*bv)))
                  
def adaptY(joinedRDD,gamma,theta,N):
    ''' return updated implicit profile Y
    '''
    return joinedRDD.map(lambda (i,j,ui,vj,bu,bv,rij,Nu,yj,Zu,delta): (j, (yj,gradient_y(delta,Nu,vj)))) \
                    .reduceByKey(custom_add_gradients_y, numPartitions = N) \
                    .mapValues(lambda (yj, grad_y): (yj-gamma*(grad_y+2*theta*yj)))
         
def SE(joinedRDD):
    ''' return scalar: square error
    '''
    return joinedRDD.map(lambda x: x[-1]**2).reduce(add)
 
def normSq_U(profileRDD,param):
    ''' return scalar: norm square error
    '''
    return param * profileRDD.map(lambda (i, (ui,bu,Nu)): np.dot(ui.T,ui)+bu**2).reduce(add)
    
def normSq_V(profileRDD,param):
    ''' return scalar: norm square error
    '''
    return param * profileRDD.map(lambda (j, (vj,bv)): np.dot(vj.T,vj)+bv**2).reduce(add)
    
def normSq_Y(profileRDD,param):
    ''' return scalar: norm square error
    '''
    return param * profileRDD.map(lambda (j, yj): np.dot(yj.T,yj)).reduce(add)

  
def train(args, folds, latent, lam, mu, theta):
    cross_val_rmses = []
    time_list = []
    for k in folds:
        train_folds = [folds[j] for j in folds if j is not k ]

        if len(train_folds)>0:
            train = train_folds[0]
            for fold in  train_folds[1:]:
                train=train.union(fold)
            train.repartition(args.N).cache()
            test = folds[k].repartition(args.N).cache()
            Mtrain=train.count()
            Mtest=test.count() 
            print("Initiating fold %d with %d train samples and %d test samples" % (k,Mtrain,Mtest) )
        else:
            train = folds[k].repartition(args.N).cache()
            test = train
            Mtrain=train.count()
            Mtest=test.count()
            print("Running single training over training set with %d train samples. Test RMSE computes RMSE on training set" % Mtrain )
            
        i = 0
        change = 1.e99
        obj = 1.e99
        
        #Generate user profiles
        U = generateUserProfiles(train,int(latent),args.seed,sc,args.N).repartition(args.N).cache()
        V = generateItemProfiles(train,int(latent),args.seed,sc,args.N).repartition(args.N).cache()
        Y = generateImplicitProfiles(train,int(latent),args.seed,sc,args.N).repartition(args.N).cache() 
        M = generateSumY(train,Y,args.N).repartition(args.N).cache()
        TrainRDD = train.map(lambda (i,j,rij): (i, (rij,j))).repartition(args.N).cache()
        TestRDD = test.map(lambda (i,j,rij): (i, (rij,j))).repartition(args.N).cache()
        Rmean = globalMean(train)

        train.unpersist()
        test.unpersist()
        
        numItems = V.count()
        numUsers = U.count()

        print "Training set contains %d users and %d items" %(numUsers,numItems)
        
        start = time()
        gamma = args.gain

        while i<args.maxiter and change > args.epsilon:
            i += 1
            
            joinedRDD = joinAndPredictAll(TrainRDD,U,V,Y,M,Rmean,args.N).cache()
            joinedRDD_test = joinAndPredictAll(TestRDD,U,V,Y,M,Rmean,args.N).cache()
            
            trainRMSE = np.sqrt(1.*SE(joinedRDD)/Mtrain) 
            testRMSE = np.sqrt(1.*SE(joinedRDD_test)/Mtest)   
            #obj += normSq_U(U,lam) + normSq_V(V,mu) + normSq_Y(Y,theta)   
            #change = np.abs(obj-oldObjective) 
            
            U.unpersist()
            V.unpersist()
            Y.unpersist()
            
            gamma = args.gain / i**args.power
            U = adaptU(joinedRDD,gamma,lam,args.N).cache()
            V = adaptV(joinedRDD,gamma,mu,args.N).cache()
            Y = adaptY(joinedRDD,gamma,theta,args.N).cache()

            now = time()-start
            #if i % 10 == 0:
            print "Iteration: %d\tTime: %f\tTranRMSE: %f\tTestRMSE: %f" % (i,now,trainRMSE,testRMSE)
            joinedRDD.unpersist()
            joinedRDD_test.unpersist()
          
        time_list.append(now)  
        cross_val_rmses.append(testRMSE)
        
        TrainRDD.unpersist()
        TestRDD.unpersist()
        
    return cross_val_rmses, time_list, U, V, Y   

def prediction(Rmean,vj,bu,bv,Zu):
    return Rmean + (bu-Rmean) + (bv-Rmean) + vj.T.dot(Zu)

def test(sparkContext, args):
    print "loading training data to estimate Global Mean..."
    try:
        population = readRatings(args.data, sparkContext)
        Rmean = globalMean(population)
        print "Success estimating Global Mean"
    except:
        print "Fail estimating Global Mean"
        return None
        
    print "loading model..."
    try:
        U = sparkContext.textFile(args.modelDir+"model_SVDppU", use_unicode=False).map(eval)\
                        .map(lambda (i,(ui,bu,Nu)): (i, (np.array(ui),bu,Nu))).partitionBy(args.N).cache()
        V = sparkContext.textFile(args.modelDir+"model_SVDppV", use_unicode=False).map(eval)\
                        .map(lambda (j,(vj,bv)): (j, (np.array(vj),bv))).partitionBy(args.N).cache()
        Y = sparkContext.textFile(args.modelDir+"model_SVDppY", use_unicode=False).map(eval)\
                        .map(lambda (j,yj): (j,np.array(yj))).partitionBy(args.N).cache()
        M = generateSumY(population,Y,args.N).repartition(args.N).cache()
        print "Success loading model"
    except:
        print "Fail loading model"
        return None  
    
    if args.predict == 'user':
        userprofile = U.join(M)\
                       .map(lambda (i, ((ui,bu,Nu),sumY)): (1,(i,ui,bu,Nu,sumY)) if i == args.id else None)\
                       .filter(lambda x: x is not None).repartition(args.N)
        
        itemprofile = V.join(Y)\
                       .map(lambda (j, ((vj,bv),yj)): (1,(j,vj,bv,yj))).repartition(args.N)
        joinedprofile = itemprofile.join(userprofile)\
                                   .map(lambda (c, ((j,vj,bv,yj),(i,ui,bu,Nu,sumY))): (prediction(Rmean,vj,bu,bv,ui+Nu*sumY),j))\
                                   .sortByKey(ascending=False)
        print "Recommend user %s Top 10 items: " % args.id
        print joinedprofile.take(10)
        
    elif args.predict == 'item':
        itemprofile = V.join(Y)\
                       .map(lambda (j, ((vj,bv),yj)): (1,(j,vj,bv,yj)) if j == args.id else None)\
                       .filter(lambda x: x is not None).repartition(args.N)
        userprofile = U.join(M)\
                       .map(lambda (i, ((ui,bu,Nu),sumY)): (1,(i,ui,bu,Nu,sumY))).repartition(args.N)
        joinedprofile = userprofile.join(itemprofile)\
                                   .map(lambda (c, ((i,ui,bu,Nu,sumY),(j,vj,bv,yj))): (prediction(Rmean,vj,bu,bv,ui+Nu*sumY),i))\
                                   .sortByKey(ascending=False)
        print "Recommend item %s to Top 10 uers: " % args.id
        print joinedprofile.take(10)
    
if __name__=="__main__":
        
    parser = argparse.ArgumentParser(description = 'Parallele Matrix Factorization.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data',help = 'Directory containing folds. The folds should be named fold0, fold1, ..., foldK.')
    parser.add_argument('folds',type = int,help = 'Number of folds')
    parser.add_argument('--mode',default='train',help ="Default is training mode, others will be test mode")
    parser.add_argument('--predict',default='user',help ="Default is to recommend user items, keyin 'item' will recommend item to users")
    parser.add_argument('--id',default=1,help ="user or item id")
    
    parser.add_argument('--gain',default=0.001,type=float,help ="Gain")
    parser.add_argument('--power',default=0.2,type=float,help ="Gain Exponent")
    parser.add_argument('--epsilon',default=1.e-99,type=float,help ="Desired objective accuracy")
    parser.add_argument('--lam',default=1.0,type=float,help ="Regularization parameter for user features")
    parser.add_argument('--mu',default=1.0,type=float,help ="Regularization parameter for item features")
    parser.add_argument('--d',default=10,type=int,help ="Number of latent features")
    parser.add_argument('--maxiter',default=20,type=int, help='Maximum number of iterations')
    parser.add_argument('--N',default=40,type=int, help='Parallelization Level')
    parser.add_argument('--seed',default=1234567,type=int, help='Seed used in random number generator')
    
    parser.add_argument('--cv',default=None, help='Do cross validation. If is none, only do training')
    parser.add_argument('--output',default=None, help='Output the best U and V')
    parser.add_argument('--latents', type=float, nargs='+', help='Regularization parameter latent dimensions List')
    parser.add_argument('--regul', type=float, nargs='+', help='Regularization parameters List, lambda and mu')
    parser.add_argument('--plotCSV', default='plotCSV_pp', help='File that write latent dimensions corresponding CV RMSE')
    
    parser.add_argument('--modelDir', default='./', help='Enter root folder containing saved parameters eg. ./model_SVDppU')

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    sc = SparkContext(appName='Parallel MF with SVD++')
    
    if not args.verbose :
        sc.setLogLevel("ERROR")        

    folds = {}

    if args.cv is not None:
        for k in range(args.folds):
            folds[k] = readRatings(args.data+"/part-0000"+str(k),sc)
    else:
        folds[0] = readRatings(args.data,sc)

    if args.mode == 'train' or args.mode == 'Train':
        csvData = list()
        minrmse = 9999
        bestlatent = -1
        bestre = -1
        for latent in args.latents: 
            for re in args.regul:
                cv_rmses, time_list, U, V, Y = train(args, folds, latent, re, re, re)        
                cur_kfold_rmse = np.mean(cv_rmses)
                timecost = np.sum(time_list)
                print "Latent %d, regularization %f, average error is: %f, total time cost %f" % (latent, re, cur_kfold_rmse, timecost)
                
                if minrmse > cur_kfold_rmse:
                    minrmse = cur_kfold_rmse
                    best_U = U
                    best_V = V
                    best_Y = Y
                    bestlatent = latent
                    bestre = re
                    
                U.unpersist()
                V.unpersist()
                Y.unpersist()
                storeplot = [latent, re, cur_kfold_rmse, timecost]
                csvData.append(storeplot)
                
        print "Best Latent %d, Best Regul %d, RMSE %f" % (bestlatent, bestre, minrmse)
        with open(args.plotCSV, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)
        csvFile.close()
        
        if args.output is not None:
            print "Saving U, V, Y, bu and bv RDDs"
            best_U = best_U.map(lambda (i, (ui,bu,Nu)): (i, (list(ui),bu,Nu)))
            best_V = best_V.map(lambda (j, (vj,bv)): (j, (list(vj),bv)))
            best_Y = best_Y.map(lambda (j,yj): (j, list(yj)))
            best_U.saveAsTextFile(args.output+'_SVDppU')
            best_V.saveAsTextFile(args.output+'_SVDppV')
            best_Y.saveAsTextFile(args.output+'_SVDppY')
    elif args.mode == 'test' or args.mode == 'Test':
        test(sc, args)
    else:
        print "Pleas try --mode train or --mode test"