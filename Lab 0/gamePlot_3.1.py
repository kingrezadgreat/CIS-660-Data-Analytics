"""
game prediction project 0

"""
import csv
import matplotlib.pyplot as plt
    
select = 1
region_num = 6 #6:NA, 7:EU, 8:jpn, 9:OTH, 10:GLB

plat = []
with open('videogamdata.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        if row [2] not in plat and row [2]!='Platform':
            plat.append(row [2])
print "Company length is: " ,  len(plat)
print plat            
print " "

platNew = []
for p in plat:
    year = []
    with open('videogamdata.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if row [3] not in year and row [2]==p and row [3]!="N/A" and int(row [3])<=2017:
                year.append(row [3])

    year.sort()


    if ('2016' in year) or ('2015' in year) or ('2014' in year):
        platNew.append(p)
   


for p in plat:
    year = []
    with open('videogamdata.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if row [3] not in year and row [2]==p and row [3]!="N/A" and int(row [3])<=2017:
                year.append(row [3])

    year.sort()
    #print ""
    #print ""
    #print 'Years are:  ', year            
    

    sumList = []
    sumTemp = 0 
    for y in year:
        with open('videogamdata.csv', 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[3] == y and row[2]==p:
                    sumTemp = sumTemp + float(row[region_num])
        sumList.append(sumTemp)
        sumTemp = 0

    #print 'Sales are:  ', sumList
    plt.xlim(1980, 2018)   # set the xlim to xmin, xmax
    
    if select == 0:
        if ('2016' in year) or ('2015' in year) or ('2014' in year):
            stri1 = [int(e) for e in year]
            stri2 = [int(e) for e in sumList]
            plt.plot(stri1, stri2)    
            plt.legend(platNew)
    
    else:    
        stri1 = [int(e) for e in year]
        stri2 = [int(e) for e in sumList]
        plt.plot(stri1, stri2)    
        plt.legend(plat)

#plt.xlim(1980, 2018)   # set the xlim to xmin, xmax
plt.xlabel('Year')
plt.ylabel('Sale')
plt.show()    


