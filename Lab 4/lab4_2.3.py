import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
from time import time
from sklearn import metrics
from sklearn import preprocessing



crimeTypes = []
crime = []
crimeBin = []
crimeCount = []
location = []

counter = 0

with open('NIJ2016_AUG01_AUG31_USE_All_0915.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        counter = counter + 1
        XYtemp = []
#        if True == True: # all
#        if (counter<1000) or (counter >3344 and counter <3444) or (counter>19092 ): # selected all
#        if (counter<3344): # STREET CRIMES
#        if (counter >3343 and counter <19092): # OTHER
#        if (counter >19091 and counter <19290): # MOTOR VEHICLE THEFT
        if (counter >19289 ): # BURGLARY        
#            print row[0]
#            raw_input()
            if row[0] != 'CATEGORY':
                for i in range (0,len(row)):
                    if i==5 or i ==6:
                        XYtemp.append(int(row[i]))
                    if i == 0:
                        crime.append(row[i])
                        if row[i] not in crimeTypes:
                            crimeTypes.append(row[i])
                            crimeCount.append(0)
                        for ii in range (0,len(crimeTypes)):
                            if row[i] == crimeTypes[ii]:
                                crimeBin.append(int(ii))
                                crimeCount[ii] = crimeCount[ii]+1
                location.append(XYtemp)
print crimeCount
print crimeTypes
#print crime[3341]
#raw_input()
'''
print len(location), len(crime), len(crimeBin)

print location[0], crime[0], crimeBin[0] 
print location[7200], crime[7200], crimeBin[7200] 
print location[19110], crime[19110], crimeBin[19110] 
print location[19322], crime[19322], crimeBin[19322] 

print crimeTypes
raw_input()
'''


print(__doc__)

import numpy as np


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs



# #############################################################################
# Generate sample data
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.01, random_state=0)
#X = StandardScaler().fit_transform(X)


#location = StandardScaler().fit_transform(location)
location = np.asarray(location)
#print type(location[0])
#raw_input()

#raw_input()
# #############################################################################
# Compute DBSCAN
#db = DBSCAN(eps=0.4, min_samples=15).fit(X)
#db = DBSCAN(eps=0.5, min_samples=15).fit(location)




X = location
#X_scaled = preprocessing.scale(X)

clus_num = 2
print ""
print "number of clusters: ", clus_num
estimator = KMeans(n_clusters=clus_num, random_state=None,  tol=0.00001, precompute_distances=True)
y_pred = estimator.fit_predict(X)
for i in range(0,clus_num):
    cur_data = []
    count = 0;
    count_t = 0;

    min_x = 99999999
    max_x = 0
    min_y = 99999999
    max_y = 0

    min_x_t = 99999999
    max_x_t = 0
    min_y_t = 99999999
    max_y_t = 0


    for ii in range(0,len(X)):
        count_t = count_t+1
        if min_x_t>X[ii][0]:
            min_x_t=X[ii][0]
          
        if min_y_t>X[ii][1]:
            min_y_t=X[ii][1]

        if max_x_t<X[ii][0]:
            max_x_t=X[ii][0]

        if max_y_t<X[ii][1]:
            max_y_t=X[ii][1]
        
        if y_pred[ii] == i:
            if min_x>X[ii][0]:
                min_x=X[ii][0]
            
            if min_y>X[ii][1]:
                min_y=X[ii][1]

            if max_x<X[ii][0]:
                max_x=X[ii][0]

            if max_y<X[ii][1]:
                max_y=X[ii][1]

#            cur_data.append(X_scaled[ii])
            cur_data.append(X[ii])
            count = count + 1
    a = float(abs((min_x-max_x)*(min_y-max_y)))
    A = float(abs((min_x_t-max_x_t)*(min_y_t-max_y_t)))

    print "Count:", count , " -- current area: ", a/1000000, " -- PAI:" , (float(count)/float(count_t))/(float(a)/float(A))
    #cur_data = np.asarray(cur_data)
    #plt.scatter(cur_data[:, 0], cur_data[:, 1])
    #plt.title("standard dev = "+ str(np.std(cur_data)) + "   count = " +  str(count) + "  strDev/count = " + str(float(np.std(cur_data))/float(count)))
    #plt.show()
    #print np.std(cur_data), count
plt.figure(figsize=(12, 12))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("2 cluster Kmeans")
plt.show()







clus_num = 3
print ""
print "number of clusters: ", clus_num
estimator = KMeans(n_clusters=clus_num, random_state=None,  tol=0.00001, precompute_distances=True)
y_pred = estimator.fit_predict(X)
for i in range(0,clus_num):
    cur_data = []
    count = 0;
    count_t = 0;

    min_x = 99999999
    max_x = 0
    min_y = 99999999
    max_y = 0

    min_x_t = 99999999
    max_x_t = 0
    min_y_t = 99999999
    max_y_t = 0


    for ii in range(0,len(X)):
        count_t = count_t+1
        if min_x_t>X[ii][0]:
            min_x_t=X[ii][0]
          
        if min_y_t>X[ii][1]:
            min_y_t=X[ii][1]

        if max_x_t<X[ii][0]:
            max_x_t=X[ii][0]

        if max_y_t<X[ii][1]:
            max_y_t=X[ii][1]
        
        if y_pred[ii] == i:
            if min_x>X[ii][0]:
                min_x=X[ii][0]
            
            if min_y>X[ii][1]:
                min_y=X[ii][1]

            if max_x<X[ii][0]:
                max_x=X[ii][0]

            if max_y<X[ii][1]:
                max_y=X[ii][1]

#            cur_data.append(X_scaled[ii])
            cur_data.append(X[ii])
            count = count + 1
    a = float(abs((min_x-max_x)*(min_y-max_y)))
    A = float(abs((min_x_t-max_x_t)*(min_y_t-max_y_t)))

    print "Count:", count , " -- current area: ", a/1000000, " -- PAI:" , (float(count)/float(count_t))/(float(a)/float(A))
    #cur_data = np.asarray(cur_data)
    #plt.scatter(cur_data[:, 0], cur_data[:, 1])
    #plt.title("standard dev = "+ str(np.std(cur_data)) + "   count = " +  str(count) + "  strDev/count = " + str(float(np.std(cur_data))/float(count)))
    #plt.show()
    #print np.std(cur_data), count
plt.figure(figsize=(12, 12))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("3 cluster Kmeans")
plt.show()








clus_num = 4
print ""
print "number of clusters: ", clus_num
estimator = KMeans(n_clusters=clus_num, random_state=None,  tol=0.00001, precompute_distances=True)
y_pred = estimator.fit_predict(X)
for i in range(0,clus_num):
    cur_data = []
    count = 0;
    count_t = 0;

    min_x = 99999999
    max_x = 0
    min_y = 99999999
    max_y = 0

    min_x_t = 99999999
    max_x_t = 0
    min_y_t = 99999999
    max_y_t = 0


    for ii in range(0,len(X)):
        count_t = count_t+1
        if min_x_t>X[ii][0]:
            min_x_t=X[ii][0]
          
        if min_y_t>X[ii][1]:
            min_y_t=X[ii][1]

        if max_x_t<X[ii][0]:
            max_x_t=X[ii][0]

        if max_y_t<X[ii][1]:
            max_y_t=X[ii][1]
        
        if y_pred[ii] == i:
            if min_x>X[ii][0]:
                min_x=X[ii][0]
            
            if min_y>X[ii][1]:
                min_y=X[ii][1]

            if max_x<X[ii][0]:
                max_x=X[ii][0]

            if max_y<X[ii][1]:
                max_y=X[ii][1]

#            cur_data.append(X_scaled[ii])
            cur_data.append(X[ii])
            count = count + 1
    a = float(abs((min_x-max_x)*(min_y-max_y)))
    A = float(abs((min_x_t-max_x_t)*(min_y_t-max_y_t)))

    print "Count:", count , " -- current area: ", a/1000000, " -- PAI:" , (float(count)/float(count_t))/(float(a)/float(A))
    #cur_data = np.asarray(cur_data)
    #plt.scatter(cur_data[:, 0], cur_data[:, 1])
    #plt.title("standard dev = "+ str(np.std(cur_data)) + "   count = " +  str(count) + "  strDev/count = " + str(float(np.std(cur_data))/float(count)))
    #plt.show()
    #print np.std(cur_data), count
plt.figure(figsize=(12, 12))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("4 cluster Kmeans")
plt.show()






clus_num = 5
print ""
print "number of clusters: ", clus_num
estimator = KMeans(n_clusters=clus_num, random_state=None,  tol=0.00001, precompute_distances=True)
y_pred = estimator.fit_predict(X)
for i in range(0,clus_num):
    cur_data = []
    count = 0;
    count_t = 0;

    min_x = 99999999
    max_x = 0
    min_y = 99999999
    max_y = 0

    min_x_t = 99999999
    max_x_t = 0
    min_y_t = 99999999
    max_y_t = 0


    for ii in range(0,len(X)):
        count_t = count_t+1
        if min_x_t>X[ii][0]:
            min_x_t=X[ii][0]
          
        if min_y_t>X[ii][1]:
            min_y_t=X[ii][1]

        if max_x_t<X[ii][0]:
            max_x_t=X[ii][0]

        if max_y_t<X[ii][1]:
            max_y_t=X[ii][1]
        
        if y_pred[ii] == i:
            if min_x>X[ii][0]:
                min_x=X[ii][0]
            
            if min_y>X[ii][1]:
                min_y=X[ii][1]

            if max_x<X[ii][0]:
                max_x=X[ii][0]

            if max_y<X[ii][1]:
                max_y=X[ii][1]

#            cur_data.append(X_scaled[ii])
            cur_data.append(X[ii])
            count = count + 1
    a = float(abs((min_x-max_x)*(min_y-max_y)))
    A = float(abs((min_x_t-max_x_t)*(min_y_t-max_y_t)))

    print "Count:", count , " -- current area: ", a/1000000, " -- PAI:" , (float(count)/float(count_t))/(float(a)/float(A))
    #cur_data = np.asarray(cur_data)
    #plt.scatter(cur_data[:, 0], cur_data[:, 1])
    #plt.title("standard dev = "+ str(np.std(cur_data)) + "   count = " +  str(count) + "  strDev/count = " + str(float(np.std(cur_data))/float(count)))
    #plt.show()
    #print np.std(cur_data), count
plt.figure(figsize=(12, 12))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("5 cluster Kmeans")
plt.show()




clus_num = 6
print ""
print "number of clusters: ", clus_num
estimator = KMeans(n_clusters=clus_num, random_state=None,  tol=0.00001, precompute_distances=True)
y_pred = estimator.fit_predict(X)
for i in range(0,clus_num):
    cur_data = []
    count = 0;
    count_t = 0;

    min_x = 99999999
    max_x = 0
    min_y = 99999999
    max_y = 0

    min_x_t = 99999999
    max_x_t = 0
    min_y_t = 99999999
    max_y_t = 0


    for ii in range(0,len(X)):
        count_t = count_t+1
        if min_x_t>X[ii][0]:
            min_x_t=X[ii][0]
          
        if min_y_t>X[ii][1]:
            min_y_t=X[ii][1]

        if max_x_t<X[ii][0]:
            max_x_t=X[ii][0]

        if max_y_t<X[ii][1]:
            max_y_t=X[ii][1]
        
        if y_pred[ii] == i:
            if min_x>X[ii][0]:
                min_x=X[ii][0]
            
            if min_y>X[ii][1]:
                min_y=X[ii][1]

            if max_x<X[ii][0]:
                max_x=X[ii][0]

            if max_y<X[ii][1]:
                max_y=X[ii][1]

#            cur_data.append(X_scaled[ii])
            cur_data.append(X[ii])
            count = count + 1
    a = float(abs((min_x-max_x)*(min_y-max_y)))
    A = float(abs((min_x_t-max_x_t)*(min_y_t-max_y_t)))

    print "Count:", count , " -- current area: ", a/1000000, " -- PAI:" , (float(count)/float(count_t))/(float(a)/float(A))
    #cur_data = np.asarray(cur_data)
    #plt.scatter(cur_data[:, 0], cur_data[:, 1])
    #plt.title("standard dev = "+ str(np.std(cur_data)) + "   count = " +  str(count) + "  strDev/count = " + str(float(np.std(cur_data))/float(count)))
    #plt.show()
    #print np.std(cur_data), count
plt.figure(figsize=(12, 12))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("6 cluster Kmeans")
plt.show()





clus_num = 7
print ""
print "number of clusters: ", clus_num
estimator = KMeans(n_clusters=clus_num, random_state=None,  tol=0.00001, precompute_distances=True)
y_pred = estimator.fit_predict(X)
for i in range(0,clus_num):
    cur_data = []
    count = 0;
    count_t = 0;

    min_x = 99999999
    max_x = 0
    min_y = 99999999
    max_y = 0

    min_x_t = 99999999
    max_x_t = 0
    min_y_t = 99999999
    max_y_t = 0


    for ii in range(0,len(X)):
        count_t = count_t+1
        if min_x_t>X[ii][0]:
            min_x_t=X[ii][0]
          
        if min_y_t>X[ii][1]:
            min_y_t=X[ii][1]

        if max_x_t<X[ii][0]:
            max_x_t=X[ii][0]

        if max_y_t<X[ii][1]:
            max_y_t=X[ii][1]
        
        if y_pred[ii] == i:
            if min_x>X[ii][0]:
                min_x=X[ii][0]
            
            if min_y>X[ii][1]:
                min_y=X[ii][1]

            if max_x<X[ii][0]:
                max_x=X[ii][0]

            if max_y<X[ii][1]:
                max_y=X[ii][1]

#            cur_data.append(X_scaled[ii])
            cur_data.append(X[ii])
            count = count + 1
    a = float(abs((min_x-max_x)*(min_y-max_y)))
    A = float(abs((min_x_t-max_x_t)*(min_y_t-max_y_t)))

    print "Count:", count , " -- current area: ", a/1000000, " -- PAI:" , (float(count)/float(count_t))/(float(a)/float(A))
    #cur_data = np.asarray(cur_data)
    #plt.scatter(cur_data[:, 0], cur_data[:, 1])
    #plt.title("standard dev = "+ str(np.std(cur_data)) + "   count = " +  str(count) + "  strDev/count = " + str(float(np.std(cur_data))/float(count)))
    #plt.show()
    #print np.std(cur_data), count
plt.figure(figsize=(12, 12))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("7 cluster Kmeans")
plt.show()






clus_num = 8
print ""
print "number of clusters: ", clus_num
estimator = KMeans(n_clusters=clus_num, random_state=None,  tol=0.00001, precompute_distances=True)
y_pred = estimator.fit_predict(X)
for i in range(0,clus_num):
    cur_data = []
    count = 0;
    count_t = 0;

    min_x = 99999999
    max_x = 0
    min_y = 99999999
    max_y = 0

    min_x_t = 99999999
    max_x_t = 0
    min_y_t = 99999999
    max_y_t = 0


    for ii in range(0,len(X)):
        count_t = count_t+1
        if min_x_t>X[ii][0]:
            min_x_t=X[ii][0]
          
        if min_y_t>X[ii][1]:
            min_y_t=X[ii][1]

        if max_x_t<X[ii][0]:
            max_x_t=X[ii][0]

        if max_y_t<X[ii][1]:
            max_y_t=X[ii][1]
        
        if y_pred[ii] == i:
            if min_x>X[ii][0]:
                min_x=X[ii][0]
            
            if min_y>X[ii][1]:
                min_y=X[ii][1]

            if max_x<X[ii][0]:
                max_x=X[ii][0]

            if max_y<X[ii][1]:
                max_y=X[ii][1]

#            cur_data.append(X_scaled[ii])
            cur_data.append(X[ii])
            count = count + 1
    a = float(abs((min_x-max_x)*(min_y-max_y)))
    A = float(abs((min_x_t-max_x_t)*(min_y_t-max_y_t)))

    print "Count:", count , " -- current area: ", a/1000000, " -- PAI:" , (float(count)/float(count_t))/(float(a)/float(A))
    #cur_data = np.asarray(cur_data)
    #plt.scatter(cur_data[:, 0], cur_data[:, 1])
    #plt.title("standard dev = "+ str(np.std(cur_data)) + "   count = " +  str(count) + "  strDev/count = " + str(float(np.std(cur_data))/float(count)))
    #plt.show()
    #print np.std(cur_data), count
plt.figure(figsize=(12, 12))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("8 cluster Kmeans")
plt.show()





clus_num = 9
print ""
print "number of clusters: ", clus_num
estimator = KMeans(n_clusters=clus_num, random_state=None,  tol=0.00001, precompute_distances=True)
y_pred = estimator.fit_predict(X)
for i in range(0,clus_num):
    cur_data = []
    count = 0;
    count_t = 0;

    min_x = 99999999
    max_x = 0
    min_y = 99999999
    max_y = 0

    min_x_t = 99999999
    max_x_t = 0
    min_y_t = 99999999
    max_y_t = 0


    for ii in range(0,len(X)):
        count_t = count_t+1
        if min_x_t>X[ii][0]:
            min_x_t=X[ii][0]
          
        if min_y_t>X[ii][1]:
            min_y_t=X[ii][1]

        if max_x_t<X[ii][0]:
            max_x_t=X[ii][0]

        if max_y_t<X[ii][1]:
            max_y_t=X[ii][1]
        
        if y_pred[ii] == i:
            if min_x>X[ii][0]:
                min_x=X[ii][0]
            
            if min_y>X[ii][1]:
                min_y=X[ii][1]

            if max_x<X[ii][0]:
                max_x=X[ii][0]

            if max_y<X[ii][1]:
                max_y=X[ii][1]

#            cur_data.append(X_scaled[ii])
            cur_data.append(X[ii])
            count = count + 1
    a = float(abs((min_x-max_x)*(min_y-max_y)))
    A = float(abs((min_x_t-max_x_t)*(min_y_t-max_y_t)))

    print "Count:", count , " -- current area: ", a/1000000, " -- PAI:" , (float(count)/float(count_t))/(float(a)/float(A))
    #cur_data = np.asarray(cur_data)
    #plt.scatter(cur_data[:, 0], cur_data[:, 1])
    #plt.title("standard dev = "+ str(np.std(cur_data)) + "   count = " +  str(count) + "  strDev/count = " + str(float(np.std(cur_data))/float(count)))
    #plt.show()
    #print np.std(cur_data), count
plt.figure(figsize=(12, 12))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("9 cluster Kmeans")
plt.show()





clus_num = 10
print ""
print "number of clusters: ", clus_num
estimator = KMeans(n_clusters=clus_num, random_state=None,  tol=0.00001, precompute_distances=True)
y_pred = estimator.fit_predict(X)
for i in range(0,clus_num):
    cur_data = []
    count = 0;
    count_t = 0;

    min_x = 99999999
    max_x = 0
    min_y = 99999999
    max_y = 0

    min_x_t = 99999999
    max_x_t = 0
    min_y_t = 99999999
    max_y_t = 0


    for ii in range(0,len(X)):
        count_t = count_t+1
        if min_x_t>X[ii][0]:
            min_x_t=X[ii][0]
          
        if min_y_t>X[ii][1]:
            min_y_t=X[ii][1]

        if max_x_t<X[ii][0]:
            max_x_t=X[ii][0]

        if max_y_t<X[ii][1]:
            max_y_t=X[ii][1]
        
        if y_pred[ii] == i:
            if min_x>X[ii][0]:
                min_x=X[ii][0]
            
            if min_y>X[ii][1]:
                min_y=X[ii][1]

            if max_x<X[ii][0]:
                max_x=X[ii][0]

            if max_y<X[ii][1]:
                max_y=X[ii][1]

#            cur_data.append(X_scaled[ii])
            cur_data.append(X[ii])
            count = count + 1
    a = float(abs((min_x-max_x)*(min_y-max_y)))
    A = float(abs((min_x_t-max_x_t)*(min_y_t-max_y_t)))

    print "Count:", count , " -- current area: ", a/1000000, " -- PAI:" , (float(count)/float(count_t))/(float(a)/float(A))
    #cur_data = np.asarray(cur_data)
    #plt.scatter(cur_data[:, 0], cur_data[:, 1])
    #plt.title("standard dev = "+ str(np.std(cur_data)) + "   count = " +  str(count) + "  strDev/count = " + str(float(np.std(cur_data))/float(count)))
    #plt.show()
    #print np.std(cur_data), count
plt.figure(figsize=(12, 12))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("10 cluster Kmeans")
plt.show()



"""
y_pred = KMeans(n_clusters=3, random_state=None,  tol=0.00001, precompute_distances=True).fit_predict(X)
plt.subplot(222)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("3 cluster Kmeans")


y_pred = KMeans(n_clusters=4, random_state=None,  tol=0.00001, precompute_distances=True).fit_predict(X)
plt.subplot(223)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("4 cluster Kmeans")


y_pred = KMeans(n_clusters=5, random_state=None,  tol=0.00001, precompute_distances=True).fit_predict(X)
plt.subplot(224)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("5 cluster Kmeans")


plt.show()


y_pred = KMeans(n_clusters=6, random_state=None,  tol=0.00001, precompute_distances=True).fit_predict(X)
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("6 cluster Kmeans")



y_pred = KMeans(n_clusters=7, random_state=None,  tol=0.00001, precompute_distances=True).fit_predict(X)
plt.subplot(222)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("7 cluster Kmeans")


y_pred = KMeans(n_clusters=8, random_state=None,  tol=0.00001, precompute_distances=True).fit_predict(X)
plt.subplot(223)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("8 cluster Kmeans")


y_pred = KMeans(n_clusters=9, random_state=None,  tol=0.00001, precompute_distances=True).fit_predict(X)
plt.subplot(224)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("9 cluster Kmeans")

plt.show()
"""












