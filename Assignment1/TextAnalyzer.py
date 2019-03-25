import sys
import argparse
import numpy as np
from pyspark import SparkContext

def toLowerCase(s):
    """ Convert a sting to lowercase. E.g., 'BaNaNa' becomes 'banana'
    """
    return s.lower()

def stripNonAlpha(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """
    return ''.join([c for c in s if c.isalpha()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text Analysis through TFIDF computation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('mode', help='Mode of operation',choices=['TF','IDF','TFIDF','SIM','TOP']) 
    parser.add_argument('input', help='Input file or list of files.')
    parser.add_argument('output', help='File in which output is stored')
    parser.add_argument('--master',default="local[20]",help="Spark Master")
    parser.add_argument('--idfvalues',type=str,default="idf", help='File/directory containing IDF values. Used in TFIDF mode to compute TFIDF')
    parser.add_argument('--other',type=str,help = 'Score to which input score is to be compared. Used in SIM mode')
    args = parser.parse_args()
  
    sc = SparkContext(args.master, 'Text Analysis')


    if args.mode=='TF':
        # Read text file at args.input, compute TF of each term,
        # and store result in file arg.output. All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings, i.e., ""
        # are also removed
        textFile = sc.textFile(args.input)
        stringRDD = textFile.flatMap(lambda line: line.split()).map(stripNonAlpha).map(toLowerCase).filter(lambda x: x!="").map(lambda word: (word, 1))\
        .reduceByKey(lambda x,y: x+y)
	stringRDD.saveAsTextFile(args.output)
        #kv = stringRDD.collectAsMap()
        #print(kv["round"])
        
        
    if args.mode=='TOP':
        # Read file at args.input, comprizing strings representing pairs of the form (TERM,VAL), 
        # where TERM is a string and VAL is a numeric value. Find the pairs with the top 20 values,
        # and store result in args.output
        textFile = sc.textFile(args.input, use_unicode=False).map(eval)
        stringRDD = textFile.sortBy(lambda pair: -pair[1])
	stringRDD.saveAsTextFile(args.output)
	#kv = stringRDD.collectAsMap()
	#print(kv["round"])	


    if args.mode=='IDF':
        # Read list of files from args.input, compute IDF of each term,
        # and store result in file args.output.  All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings ""
        # are removed
        textFile = sc.wholeTextFiles(args.input)
        docnum = textFile.keys().count()
	stringRDD = textFile.flatMapValues(lambda line:line.split()).mapValues(stripNonAlpha).mapValues(toLowerCase).filter(lambda x: x[1]!="")\
	.distinct().map(lambda comb: (comb[1],1)).reduceByKey(lambda x,y: x+y).mapValues(lambda value: np.log(1.0 * docnum / value))
	stringRDD.saveAsTextFile(args.output)
	#kv = stringRDD.collectAsMap()
	#print(kv["round"])


    if args.mode=='TFIDF':
        # Read  TF scores from file args.input the IDF scores from file args.idfvalues,
        # compute TFIDF score, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase letter-only string and VAL is a numeric value. 
        TFile = sc.textFile(args.input, use_unicode=False).map(eval).partitionBy(100)
        IDFile = sc.textFile(args.idfvalues, use_unicode=False).map(eval).partitionBy(100).cache()
	stringRDD = IDFile.join(TFile).mapValues(lambda x: 1.0*x[0]*x[1]).sortBy(lambda pair: -pair[1])
	stringRDD.saveAsTextFile(args.output)
	#kv = stringRDD.collectAsMap()
	#print(kv["round"])
  

    if args.mode=='SIM':
        # Read  scores from file args.input the scores from file args.other,
        # compute the cosine similarity between them, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL), 
        # where TERM is a lowercase, letter-only string and VAL is a numeric value. 
        TFIDF1 = sc.textFile(args.input, use_unicode=False).map(eval).partitionBy(100).cache()
        TFIDF2 = sc.textFile(args.other, use_unicode=False).map(eval).partitionBy(100)
	cossim = TFIDF1.join(TFIDF2).mapValues(lambda x: 1.0*x[0]*x[1]).values().sum()\
	/ np.sqrt(TFIDF1.mapValues(lambda x: 1.0*x*x).values().sum() * TFIDF2.mapValues(lambda x: 1.0*x*x).values().sum())
	print("Similariy: "+ str(cossim))
	simRDD = sc.parallelize([cossim])
	simRDD.saveAsTextFile(args.output)
