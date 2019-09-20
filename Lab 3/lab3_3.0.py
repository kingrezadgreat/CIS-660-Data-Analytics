"""
game prediction project 0

"""
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math



def normalize(target, data):

    targetCol = 0
    for j in range(0,Y):
        if header[j] == target:
            targetCol = j

    maxTar = 0
    minTar = 9999999
    for i in range(0,X):
        if int(data[i][targetCol])<minTar:
            minTar = int(data[i][targetCol])
        if int(data[i][targetCol])>maxTar:
            maxTar = int(data[i][targetCol])

    for i in range(0,X):
        data[i][targetCol] = (float(data[i][targetCol])-float(minTar))/(float(maxTar) - float(minTar))
    
    return data
    


def mean(target, data, header):
    
    targetCol = 0
    for j in range(0,Y):
        if header[j] == target:
            targetCol = j
    
    targetArray = []    
    for i in range(0,X):
        targetArray.append(data[i][targetCol])
    
    meanVal = np.array(targetArray).astype(np.float)
    meanVal = np.mean(meanVal)
    print 'Mean of ', target, ':', meanVal
    return meanVal


def mode(target, data, header):
    
    targetCol = 0
    for j in range(0,Y):
        if header[j] == target:
            targetCol = j
    
    targetArray = []    
    for i in range(0,X):
        targetArray.append(data[i][targetCol])
    
    modeVal = np.array(targetArray).astype(np.float)
    modeVal = stats.mode(modeVal)
    print 'Mode of ', target, ':', modeVal[0][0]
    return modeVal[0][0]

def standardDev(target, data, header):
    
    targetCol = 0
    for j in range(0,Y):
        if header[j] == target:
            targetCol = j
    
    targetArray = []    
    for i in range(0,X):
        targetArray.append(data[i][targetCol])
    
    stdVal = np.array(targetArray).astype(np.float)
    stdVal = np.std(stdVal)
    print 'STD of ', target, ':', stdVal
    return stdVal


def variance(target, data, header):
    
    targetCol = 0
    for j in range(0,Y):
        if header[j] == target:
            targetCol = j
    
    targetArray = []    
    for i in range(0,X):
        targetArray.append(data[i][targetCol])
    
    varVal = np.array(targetArray).astype(np.float)
    varVal = np.var(varVal)
    print 'Variance of ', target, ':', varVal
    print ''
    return varVal


def zScore(data, header, numData, binLen):
    for i in range(0,len(data)):
        binCount = 0
        for j in range(0,len(data[0])):
            if header[j] in numData:
                r = math.floor(data[i][j]/(1.0/binLen[binCount]))+1
                if data[i][j]==1:
                    r = math.floor(data[i][j]/(1.0/binLen[binCount]))
                z = (r-1)/(binLen[binCount]-1)
                data[i][j] = z
                binCount = binCount + 1
        
    return data


def histogramPrint(data, header, numData, binLen):
    dataSet = []
    for j in range(0,len(data[0])):
        temp = []
        if header[j] in numData:
            for i in range(0,len(data)):
                temp.append(data[i][j]) 
            
            dataSet.append(temp)

    for i in range(0, len(numData)):
        numDSet = i
        hist, bin_edges = np.histogram(dataSet[numDSet], binLen[i])
        wid = (1.0/binLen[i])-0.01
        plt.bar(bin_edges[:-1], hist, width = wid)
        plt.xlim(min(bin_edges), max(bin_edges))
        plt.title(numData[i])
        plt.show()   

def binerize(header, headerB, data):
    variableList = []
    for j in range(0,len(data[0])):
        if header[j] in headerB:
            temp = []
            for i in range(0,len(data)):
                if data[i][j] not in temp:
                    temp.append(data[i][j])
            variableList.append(temp)
    

    
    for i in range(0,len(data)):
        count = 0
        for j in range(0,len(data[0])):
            if header[j] in headerB:
                temp = []
                for jj in range(0, len(variableList[count])):
                   if data[i][j] == variableList[count][jj]:
                       temp.append(1)
                   else:
                       temp.append(0)
                data[i][j] = temp
                count=count+1
            
            

def binMedian(data, header, numData, binLen, dataOrg):
    dataSort = []
    dataSortOrg = []
    
    for j in range(0,len(data[0])):
        temp = []
        tempOrg = []
        if header[j] in numData:
            for i in range(0,len(data)):
                temp.append(data[i][j])
                tempOrg.append(float(dataOrg[i][j]))
            temp.sort()
            tempOrg.sort()
            dataSort.append(temp)
            dataSortOrg.append(tempOrg)
    
    for i in range(0,len(dataSort)):
       count = 1
       dx = 1.0/binLen[i]
       temp = []
       for j in range(0,len(dataSort[0])):
           temp.append(dataSortOrg[i][j])
           if dataSort[i][j]>count*dx:
               
               med = int(math.floor(len(temp)/2))
               print "Median of bin #", count, " of ", numData[i], " is ", temp[med]
               count = count + 1
	       temp = []
	            
       
       med = int(math.floor(len(temp)/2))
       print "Median of bin #", count, " of ", numData[i], " is ", temp[med]
       print ''

def simData(data,dataType, dataTarget):

   c1 = 0
   c2 = 2

   

   print 'DataSet 11000 :', data[c1]
   print 'DataSet 11012 :', data[c2]

   simArray = []
   distance = []
   for i in range(0,len(dataType)):
       if dataType[i] == 'b':
           cell1 = data[c1][i]
           cell2 = data[c2][i]
           temp = 0
           for j in range(0,len(data[0][i])):
               if cell1[j] == cell2[j]:
                   temp = temp + 1
           temp = 1.0*temp/len(data[0][i])
       else: 
           
           temp = abs(data[c1][i] - data[c2][i])
             
       distance.append(temp)
   
   hamming = sum(distance)*1.0/len(data[0])
   print 'hamming: ', hamming
   


   simArray = []
   distance = []
   for i in range(0,len(dataType)):
       if dataType[i] == 'b':
           cell1 = data[c1][i]
           cell2 = data[c2][i]
           temp = 0
           countZero = 0
           for j in range(0,len(data[0][i])):
               if cell1[j] == cell2[j] and cell1[j] == 1 :
                   temp = temp + 1
               if cell1[j] == cell2[j] and cell1[j] == 0 :
                   countZero = countZero + 1
           temp = 1.0*temp/(len(data[0][i])-countZero)
       else: 
           
           temp = abs(data[c1][i] - data[c2][i])
             
       distance.append(temp)
   
   jaccard = sum(distance)*1.0/len(data[0])
   print 'jaccard: ', jaccard
   
   
   simArray1 = []
   simArray2 = []
   distance = []
   A = 0
   B = 0
   B1 = 0
   B2 = 0
   for i in range(0,len(dataType)):
       if dataType[i] == 'b':
           
           cell1 = data[c1][i]
           cell2 = data[c2][i]
           tempA = 0
           tempB1 = 0
           tempB2 = 0
           for j in range(0,len(data[0][i])):
               if cell1[j] == 1 and cell2[j]==1:
                   tempA = tempA + 1
               if cell1[j] == 1:
                   tempB1 = tempB1 + 1
               if cell2[j] == 1:
                   tempB2 = tempB2 + 1
               
           A = A + tempA
           B1 = B1 + tempB1*tempB1
           B2 = B2 + tempB2*tempB2
           
       else: 
           
           A = A + data[c1][i]*data[c2][i]
           B1 = B1 + data[c1][i]*data[c1][i]
           B2 = B2 + data[c2][i]*data[c2][i]
       
       
   cosine = 1.0*A/(math.sqrt(B1)*math.sqrt(B2))
   print 'cosine:  ', cosine
   


###############################################################
###############################################################

plat = []
data = []
with open('vTargetMailCustomer.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)
        if row [16] not in plat and row [16]!='EnglishEducation':
            plat.append(row [16])

header = data.pop(0)

X = len(data)	 # number of tuples
Y = len(data[0]) # number of attributes

classList = []
for i in range (0,len(data)):
    classList.append(data[i][31])
    

'''
print "data is ", X, "by", Y
print ''
print 'Old Attr:', header
print ''
print 'Old data:', data[0]
print '' 
'''

####################
'''
PART 1.

    here is alist of all atributes and its type. I will gather all these data in 
    a new data matrix:
    
    1. Marital status:			Discrete Binary 
    2. Gender: 				Discrete Binary 
    3. Yearly income: 			Continuous Interval 
    4. Total Children: 			Discrete Interval 
    5. Number of children at home: 	Discrete Interval  
    6. English education: 		Discrete Ordinal
    7. House owner flag: 		Discrete Binary 
    8. number of cars owned: 		Discrete Interval 
    10. Region: 			Discrete Nominal
    11. Age: 				Discrete Interval 
    
    
    At the enf of this section header holds all target atributes and data holds 
    tuple data
'''
header1 = ['MaritalStatus','Gender', 'YearlyIncome', 'TotalChildren', 'NumberChildrenAtHome', 'EnglishEducation', 'HouseOwnerFlag', 'NumberCarsOwned', 'Region', 'Age']
data1 = []

for i in range(0,X):
    row = []
    for j in range(0,Y):
        if header[j] in header1:
            row.append(data[i][j])
    data1.append(row)
    
X = len(data1)	 # number of tuples
Y = len(data1[0]) # number of attributes

data = []

for i in range(0,X):
    row = []
    for j in range(0,Y):
        row.append(data1[i][j])
    data.append(row)
    



'''
print 'New Arrt:  ', header1
print ''
print 'New data:', data[0]
raw_input()
'''

header = []
for i in range(0,len(header1)):
    header.append(header1[i])

from copy import deepcopy
dataOrg = deepcopy(data)


    

####################        
'''        
PART 2.1: 
handling null values:

   all values that are null are replaced by unknown however, not 
   all null values are valuable. Here is a list of parameters with 
   null values:
   	Title:		18383
   	MiddleName: 	7830
   	Suffix: 	18481
   	AddressLine2:	18172
   
   By just looking at the numbers it is evident that most over %98 of data
   for Title, Suffix, and AddressLine2 are NULL values. It can be deductaed
   that either it is normal to have null values for these parameters or they 
   are terribly corrupt. We will go throught them one by one: 
   
   1. Title has to replaced by unknown. It should be Mr, Mrs or niether. As 
      most of them are NULL, and it is just a title probably user did not provide
      it. In any case this data would not be a source of analysis and is not 
      reliable. All null value titles are replaced by unknown.
   2. MiddleNames can be replaced by unknown too as some people do not have 
      middle names. However, this might be a fault too. %42 of middle names are
      null. Middle names are replaced by unknown
   3. Suffix is replaced by unknown
   4. AddressLine2 is replaced by unknown
   
   As the null values of all of the parameters mentioned above are more than %10,
   I assume none of them as a good source of analysis.
   
    
'''

count = 0
for j in range(0,Y):
    count = 0
    for i in range(0,X):
        if data[i][j] == 'NULL':
            count = count + 1
            data[i][j] = 'unknown'

####################
'''
PART 2.1: 
duplication:
   
   This pice of code captures if there is any duplications. It checks the following 
   parameters:
      1. First name
      2. Middle name
      3. Last name
      4. Age
   
   By checking these 4 parameters one shoudl be able to say if there is any duplication 
   if there is no duplicate you may proceed. Ottherwise, you may take action and remove 
   the tuple.   
    
'''
'''
for i in range(0,X):
    for j in range(0,X):
        if i != j:
            if data[i][4] == data[j][4] and data[i][5] == data[j][5] and data[i][6] == data[j][6] and data[i][30] == data[j][30]:
                print i, j , "duplicate", data[i][4], data[i][5], data[i][6], data[j][4], data[j][5], data[j][6]
             
'''
####################
'''
PART 2.2: 
Random sampling:
    
    This set of data is already sampled so there is not need to sample. In care
    there is a need to random sample data you can use the following piece of code

''' 
'''
import random         
import math
sampleData = []    
sampleNum = 100
for i in range(0,sampleNum):
    randomData = random.uniform(0, 1)
    randomData = int(math.floor(randomData*X))
    sampleData.append(randomData)

print sampleData
'''

####################
'''
PART 2.3: Mean/Variance/Standard Deviation for Ordinal Numeric attributes 
'''

numData = ['YearlyIncome', 'TotalChildren', 'NumberChildrenAtHome', 'NumberCarsOwned', 'Age']


'''
print 'Mean, Mode, Variance, Standard Deviation:'
meanData = []
modeData = []
STDData = []
varData = []
for i in range(0,len(numData)):
    meanData.append(mean(numData[i], data, header))
    modeData.append(mode(numData[i], data, header))
    STDData.append(standardDev(numData[i], data, header))
    varData.append(variance(numData[i], data, header))

raw_input()
'''
####################
'''
PART 2.4: Normalization
    In this section data is normalized

'''
#print 'Normalization:'
#print 'Before:    ' , data[0]

data = normalize('YearlyIncome',data)
data = normalize('TotalChildren',data)
data = normalize('NumberChildrenAtHome',data)
data = normalize('NumberCarsOwned',data)
data = normalize('Age',data)

#print 'After     :' , data[0]
#print ''
#raw_input()


####################
'''
PART 2.5: Standardization
   calculating z-score
'''
binLen = [10, 6, 6, 5, 15]
#print 'Standardization: '
#print 'Before: ' , data[0]
data = zScore(data, header, numData, binLen)
#print 'After:  ', data[0]
#raw_input()


####################
'''
PART 2.6: Discretization: Histogram
   calculating bins and plotting histogram
'''

#histogramPrint(data, header, numData, binLen)



####################
'''
PART 2.7: Getting median of bins
'''
'''
binMedian(data, header, numData, binLen, dataOrg)
raw_input()
'''

####################
'''
PART 2.8: Binerization

    1. Marital status:			Discrete Binary 
    2. Gender: 				Discrete Binary 
    3. Yearly income: 			Continuous Interval 
    4. Total Children: 			Discrete Interval 
    5. Number of children at home: 	Discrete Interval  
    6. English education: 		Discrete Ordinal
    7. House owner flag: 		Discrete Binary 
    8. number of cars owned: 		Discrete Interval 
    10. Region: 			Discrete Nominal
    11. Age: 				Discrete Interval 

'''
headerB = ['MaritalStatus', 'Gender', 'EnglishEducation', 'HouseOwnerFlag', 'Region']
'''
print 'Attributes: ', headerB
print 'Standardization: '
print 'Before(0)  : ', data[0]
print 'Before(300): ', data[300]
'''

binerize(header, headerB, data)

'''
print 'After(0)  :  ', data[0]
print 'After(300):  ', data[300]
print ''
raw_input()
'''
                
                
                
####################
'''
PART 3: Similarity



'''                

dataType = ['b', 'b','n','n','n','b','b','n','b','n']      
dataTarget = [0,1]
#print data[0]
#simData(data,dataType, dataTarget)


####################
'''
PART 4: kNN

'''                
print ''
#print 'kNN'
divide = float(2/3)
trainingMin = 0
trainingMax = int(int(len(data))*2/3)
testMin = trainingMax + 1
testMax = int(len(data))

#print trainingMin, trainingMax, testMin ,testMax

trainSet = []
trainClass = []
for i in range (trainingMin,trainingMax):
    dataTemp = []
    for j in range(0,len(data[0])):
        if isinstance(data[i][j], list):
            for w in range(0,len(data[i][j])):
                #dataTemp.append(float(data[i][j][w])/2)
                dataTemp.append(float(data[i][j][w]))
        else:
            dataTemp.append(data[i][j])
    trainSet.append(dataTemp)
    trainClass.append(classList[i])


testSet = []
testClass = []
for i in range (testMin,testMax):
    dataTemp = []
    for j in range(0,len(data[0])):
        if isinstance(data[i][j], list):
            for w in range(0,len(data[i][j])):
                dataTemp.append(float(data[i][j][w])/2)
        else:
            dataTemp.append(data[i][j])
    testSet.append(dataTemp)
    testClass.append(classList[i])


from sklearn.neighbors import NearestNeighbors

kNNx = []
kNNy = []
kNNMin = 1
kNNMax = 20
#kNNMax = 2
for NN in range(kNNMin,kNNMax):

    neigh = NearestNeighbors(n_neighbors=NN)
    neigh.fit(trainSet) 

    countTrue = 0
    CountFalse = 0
    for xx in range (testMin,testMax):

        targetCell = xx - testMin
        result = neigh.kneighbors([testSet[int(targetCell)]])
  
        sumTemp = 0.0
        for i in range(0,len(result[0][0])):
            sumTemp = float(trainClass [result[1][0][i]]) + sumTemp

        decision = float(sumTemp)/float(len(result[0][0]))

        if abs(int(testClass[targetCell])-decision)>0.5:
            CountFalse = CountFalse + 1 

        else:
            countTrue = countTrue + 1 


    accuracy = (float(countTrue)/float(countTrue+CountFalse))
    print 'kNN Accuracy', '( # of neighbors =', NN, ') :' , accuracy

    kNNx.append(NN)
    kNNy.append(accuracy)


import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(211)

plt.plot((kNNx), kNNy, '--')
plt.xlabel('# of neighbors')
plt.ylabel('accuracy')

####################
'''
PART 5: SVM

''' 
print ''
#print 'SVM'
from sklearn import svm

SVMType = ['sigmoid','linear','rbf', 'poly']


SVMx = []
SVMy = []

for SVMLoop in range (0,len(SVMType)):
    
    #clf = svm.SVR(kernel=SVMType[SVMLoop])
    clf = svm.SVC(kernel=SVMType[SVMLoop])
    #clf = svm.NuSVC(kernel=SVMType[SVMLoop])
    #clf = svm.NuSVR(kernel=SVMType[SVMLoop])
    clf.fit(trainSet, trainClass) 

    countTrue = 0
    CountFalse = 0
    for xx in range (testMin,testMax):

        targetCell = xx - testMin
        result = clf.predict([testSet[targetCell]])

        if abs(float(result[0])-float(testClass[targetCell]))>0.5:
            CountFalse = CountFalse + 1 

        else:
            countTrue = countTrue + 1 


    accuracy = (float(countTrue)/float(countTrue+CountFalse ))
    print 'SVM Accuracy', '( kernel =',SVMType[SVMLoop],') :', accuracy    
    SVMx.append(SVMType[SVMLoop])
    SVMy.append(accuracy)


plt.subplot(212)

plt.plot(SVMx, SVMy,'o')
plt.xlabel('SVM type')
plt.ylabel('accuracy')
plt.show()














