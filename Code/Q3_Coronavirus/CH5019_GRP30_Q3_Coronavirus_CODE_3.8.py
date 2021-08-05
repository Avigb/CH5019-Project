import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("covid_19_india.csv")
l =[]
l = list(df['Date'].unique())
case = []

for j in range(len(l)):
    val = l[j]
    c = 0
    for i in range(len(df)):
#        if(df.loc[i]['State/UnionTerritory']=='Kerala'):
#ADD THE ABOVE LINE FOR GETTING THE CURVE FOR KERALA        
            if(val == df.loc[i]['Date']):
                c = c + df.loc[i]['Confirmed']-df.loc[i]['Deaths']-df.loc[i]['Cured']
    case.append(c)
eday = []

date = []
for i in range(31,72):
    cpday = (case[i])
    eday.append(cpday)
    date.append(l[i])
print(eday)
plt.plot(date,eday)  
plt.xticks(rotation = 90)
plt.title('NO FLATTENING IN CURVE - INDIA')
plt.xlabel('day of the month')
plt.ylabel('Active cases on a day')
         