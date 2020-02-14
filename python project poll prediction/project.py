
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm


df=pd.DataFrame()

for f in['C:\\Users\\Mahesh\\Desktop\\python\\datasets\\election2009.xlsx','C:\\Users\\Mahesh\\Desktop\\python\\datasets\\election2014.xlsx']:
    data=pd.read_excel(f,'Sheet1')#this loop mergwes the two excel sheets
    df=df.append(data)
    
df.to_excel('C:\\Users\\Mahesh\\Desktop\\python\\datasets\\combineddata.xlsx')#resultantdata
data=pd.read_excel('C:\\Users\\Mahesh\\Desktop\\python\\datasets\\combineddata.xlsx')
print(data)

plt.figure(figsize=(20,10))
plt.scatter(data['PARTYNAME'],data['SEATS'],c='blue')
plt.xlabel("PARTY")
plt.ylabel("SEATS WON")
plt.xticks(rotation=90)
plt.show()

final_Data={}
for i in data["PARTYNAME"]:
    x=i
    t1=(data[(data.PARTYNAME==x)&(data.YEAR==2009)].SEATS).tolist()
    t2=(data[(data.PARTYNAME==x)&(data.YEAR==2014)].SEATS).tolist()
    t3=t1+t2
    print("----------------------")
    print("Name of thr party=",i)
    print("NUMBER OF SEATS=",int(sum(t3)))
    final_Data.update({i:int(sum(t3))})
    
plt.bar(final_Data.keys(),final_Data.values(),color='green')
plt.xlabel("Party")
plt.ylabel("Seats Won")
plt.xticks(rotation=90)

print("the changes in performance is as follows")
print("-******************************************************************-")

for i in data['PARTYNAME']:
    x=i
    t2=(data[(data.PARTYNAME==x)&(data.YEAR==2014)].SEATS).tolist()
    t1=(data[(data.PARTYNAME==x)&(data.YEAR==2009)].SEATS).tolist()
    diff=sum(t1[0:])-sum(t2[0:])
    if diff>0:
        print("PARTY NAME:",x)
        print("loss from 2009 to 2014=",int(diff))
    else:
        if(diff<0):
            print("PARTY NAME:",x)
            print("Gain from 2009 t0 2014=",abs(int(diff)))
        else:
            print("PARTY NAME:",x)
            print("NO CHANGE FROM 2009 to 2014=",int(diff))
    
    print("--------------------------------------------------")
    
    
    
print("PRIDICTION FOR BJP___*************************************")
data=pd.read_excel("C:\\Users\\Mahesh\\Desktop\\python\\datasets\\BJP.xlsx")#reading data
data.head()


plt.figure(figsize=(20,15))
plt.scatter(data['year'],data['seats'],c='blue')


plt.xlabel("YEARS(BJP)")
plt.ylabel("SEATS (OUT OF 451)")
plt.show()



X=data['year'].values.reshape(-1,1)
y=data['seats'].values.reshape(-1,1)
reg=LinearRegression()
reg.fit(X,y)

print("the linear model is:Y={:.5}X+{:.5}".format(reg.coef_[0][0],reg.intercept_[0]))


predictions=reg.predict(X)
plt.figure(figsize=(16,8))
plt.scatter(data['year'],data['seats'],c='black')

plt.plot(data['year'],predictions,c='red',linewidth=3)
plt.xlabel("YEARS")
plt.ylabel("SEATS WON (OUT OF 451)")
plt.show()

X=data['year']
y=data['seats']
X2=sm.add_constant(X)


print("ENTER THE YEAR")
p=int(input())
seats=p*reg.coef_[0][0]+reg.intercept_[0]
print("THE SEATS ARE :",seats)


  
print("PRIDICTION FOR CONGRESS___*************************************")
data=pd.read_excel("C:\\Users\\Mahesh\\Desktop\\python\\datasets\\CONGRES.xlsx")#reading data
data.head()


plt.figure(figsize=(20,15))
plt.scatter(data['year'],data['seats'],c='blue')
f         

plt.xlabel("YEARS(CONGRESS)")
plt.ylabel("SEATS (OUT OF 451)")
plt.show()



X=data['year'].values.reshape(-1,1)
y=data['seats'].values.reshape(-1,1)
reg=LinearRegression()
reg.fit(X,y)

print("the linear model is:Y={:.5}X+{:.5}".format(reg.coef_[0][0],reg.intercept_[0]))


predictions=reg.predict(X)
plt.figure(figsize=(16,8))
plt.scatter(data['year'],data['seats'],c='black')

plt.plot(data['year'],predictions,c='red',linewidth=3)
plt.xlabel("YEARS")
plt.ylabel("SEATS WON (OUT OF 451)")
plt.show()

X=data['year']
y=data['seats']
X2=sm.add_constant(X)
'''
est=sm.OLS(y,X2)
est2=est.fit()
print(est2.summary())
'''

print("ENTER THE YEAR")
p=int(input())
seats=p*reg.coef_[0][0]+reg.intercept_[0]
print("THE SEATS ARE :",seats)