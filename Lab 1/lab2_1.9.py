"""
game prediction project 0

"""
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math



def normalize(target, data):

    #target = 'YearlyIncome'
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
#        data[i][targetCol] = float(data[i][targetCol])
    
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
    #print 'Mean of ', target, ':', meanVal
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
    #print 'Mode of ', target, ':', modeVal[0][0]
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
    #print 'STD of ', target, ':', stdVal
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
    #print 'Variance of ', target, ':', varVal
    return varVal


def zScore(data, header, numData, binLen):
    #print data[1]
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
    #print binLen, len(binLen)
    #print numData, len(numData)
    #raw_input()     
    #'''

    for i in range(0, len(numData)):
        numDSet = i
        #hist, bin_edges = np.histogram(dataSet[numDSet], bins = 'auto')
        hist, bin_edges = np.histogram(dataSet[numDSet], binLen[i])
        #wid = math.floor((max(dataSet[numDSet]) - min(dataSet[numDSet]))/len(bin_edges))
        #wid = ((max(dataSet[numDSet]) - min(dataSet[numDSet]))/len(bin_edges))
        #wid = 0.09
        wid = (1.0/binLen[i])-0.01
        plt.bar(bin_edges[:-1], hist, width = wid)
        plt.xlim(min(bin_edges), max(bin_edges))
        plt.show()   
    #'''

def binerize(header, headerB, data):
    variableList = []
    for j in range(0,len(data[0])):
        if header[j] in headerB:
            temp = []
            for i in range(0,len(data)):
                if data[i][j] not in temp:
                    temp.append(data[i][j])
            variableList.append(temp)
    
    #
    #print variableList
    #print headerB
    #selec = 300
    #print data[selec]

    
    for i in range(0,len(data)):
        count = 0
        for j in range(0,len(data[0])):
            if header[j] in headerB:
                temp = []
                for jj in range(0, len(variableList[count])):
                   #print '  ', data[i][j] , variableList[count][jj]
                   if data[i][j] == variableList[count][jj]:
                       temp.append(1)
                   else:
                       temp.append(0)
                #print temp
                #raw_input()
                data[i][j] = temp
                count=count+1
            
    
    #print ''
    #print data[selec]  
    #print header[2]  
            

def binMedian(data, header, numData, binLen, dataOrg):
    dataSort = []
    dataSortOrg = []
    #print len(data)
    
    #print dataOrg[0]
    #raw_input()

    #tempOrg = map(int, tempOrg)
            
    
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
            #print tempOrg 
            #raw_input()
            
    #print len(dataSort), len(dataSort[0])
    #print dataSortOrg[0]
    #print len(dataSort), len(dataSort[0])
    #print len(dataSortOrg), len(dataSortOrg[0])
    #print (dataSort[0][0])
    #print (dataSortOrg[0][0])
    
    for i in range(0,len(dataSort)):
       count = 1
       dx = 1.0/binLen[i]
       temp = []
       for j in range(0,len(dataSort[0])):
           temp.append(dataSortOrg[i][j])
           if dataSort[i][j]>count*dx:
               
               med = int(math.floor(len(temp)/2))
               #print count*dx
               #print len(temp), med, float(temp[med])
               print "Median of bin #", count, " of ", numData[i], " is ", temp[med]
               #print np.mean(temp)
               #raw_input()
               count = count + 1
               #print ''
	       #print temp
	       #print ''  
	       #raw_input()
	       temp = []
	            
       
       med = int(math.floor(len(temp)/2))
       #print temp[0], med
       print "Median of bin #", count, " of ", numData[i], " is ", temp[med]
       print ''
       #print temp
       #print ''  
       #raw_input()     


def simData(data,dataType, dataTarget):
   c1 = 0
   c2 = 1
   #print data[c1]
   #print data[c2]
   simArray = []
   distance = []
   for i in range(0,len(dataType)):
       if dataType[i] == 'b':
           cell1 = data[c1][i]
           cell2 = data[c2][i]
           temp = 0
           for j in range(0,len(data[0][i])):
               #print cell1[j], cell2[j] 
               if cell1[j] == cell2[j]:
                   temp = temp + 1
           #print temp, len(data[0][i])
           temp = 1.0*temp/len(data[0][i])
           #print temp      
           #print ''
       else: 
           
           temp = abs(data[c1][i] - data[c2][i])
           #print data[c1][i] , data[c2][i]
           #print '   ' , temp
             
       distance.append(temp)
       
       #print data[c1][i], data[c2][i], temp
   
   hamming = sum(distance)*1.0/len(data[0])
   print hamming
   
   


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

#print header
X = len(data)	 # number of tuples
Y = len(data[0]) # number of attributes
print "data is ", X, "by", Y


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
#dataOrg = []

for i in range(0,X):
    row = []
    for j in range(0,Y):
        row.append(data1[i][j])
    data.append(row)
#    dataOrg.append(row)

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
        #if data[i][j] == '':
        if data[i][j] == 'NULL':
            count = count + 1
            data[i][j] = 'unknown'
    #print count , " : ", header[j]

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
PART 2.4: Normalization
    In this section data is normalized

'''
#print dataOrg[0]

data = normalize('YearlyIncome',data)
data = normalize('TotalChildren',data)
data = normalize('NumberChildrenAtHome',data)
data = normalize('NumberCarsOwned',data)
data = normalize('Age',data)



####################
'''
PART 2.3: Mean/Variance/Standard Deviation for Ordinal Numeric attributes 
'''

numData = ['YearlyIncome', 'TotalChildren', 'NumberChildrenAtHome', 'NumberCarsOwned', 'Age']
meanData = []
modeData = []
STDData = []
varData = []
for i in range(0,len(numData)):
    meanData.append(mean(numData[i], data, header))
    modeData.append(mode(numData[i], data, header))
    STDData.append(standardDev(numData[i], data, header))
    varData.append(variance(numData[i], data, header))


####################
'''
PART 2.6: Discretization: Histogram
   calculating bins and plotting histogram
'''
binLen = [10, 6, 6, 5, 15]
#histogramPrint(data, header, numData, binLen)


####################
'''
PART 2.5: Standardization
   calculating z-score
'''

data = zScore(data, header, numData, binLen)


####################
'''
PART 2.7: Getting median of bins
'''

binMedian(data, header, numData, binLen, dataOrg)



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
binerize(header, headerB, data)
#print data[0]

                
                
                
####################
'''
PART 3: Similarity



'''                

dataType = ['b', 'b','n','n','n','b','b','n','b','n']      
dataTarget = [0,1]
#print len(data[0])
#print data[0]
#print data[1]

simData(data,dataType, dataTarget)

    
    
              
            






















