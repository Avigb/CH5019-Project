import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import re
from matplotlib import rc

#Q1


agegr = pd.read_csv('AgeGroupDetails.csv')
plt.bar(agegr['AgeGroup'],agegr['TotalCases'])
print(agegr[agegr['TotalCases']==agegr['TotalCases'].max()].AgeGroup)


#Q2

cv = pd.read_csv('covid_19_india.csv')
cv = cv.rename(columns={'State/UnionTerritory': 'State'})


for i in range(cv['Date'].count()):
    cv['Date'][i] = datetime.strptime(cv.loc[i].Date, '%d/%m/%y').date()

cvg = pd.DataFrame()
cvg['Confirmed'] = cv.groupby('Date')['Confirmed'].sum().sort_values()
cvg['Cured'] = cv.groupby('Date')['Cured'].sum().sort_values()
cvg['Deaths'] = cv.groupby('Date')['Deaths'].sum().sort_values()
cvg['Confpday'] = cvg['Confirmed']
cvg['Deathspday'] = cvg['Deaths']
cvg['Curedpday'] = cvg['Cured']
for i in range(1,len(cvg.index)):
    cvg['Confpday'][i] = cvg['Confirmed'][i] - cvg['Confirmed'][i-1]
    cvg['Deathspday'][i] = cvg['Deaths'][i] - cvg['Deaths'][i-1]
    cvg['Curedpday'][i] = cvg['Cured'][i] - cvg['Cured'][i-1]

plt.figure()
plt.title('India')
ln = plt.plot(cvg.index,cvg[['Confpday','Curedpday','Deathspday']])
plt.legend(ln[:3], ('Confirmed', 'Deaths', 'Cured'))

plt.show()

#plt.plot(cvg.index,cvg[])
labl = pd.DataFrame(columns = ['Date','State'])
a = []
b = []
c = []
d = []
e = []
cvs = pd.DataFrame()
#cvs = cv.groupby(['State','Date'])['State','Confirmed'].sum()
cvs['Confirmed'] = cv.groupby(['State','Date'])['Confirmed'].sum()
cvs['Deaths'] = cv.groupby(['State','Date'])['Deaths'].sum()
cvs['Cured'] = cv.groupby(['State','Date'])['Cured'].sum()
for i in range(len(cvs.index)):
    a.append(cvs.index[i][1])
    b.append(cvs.index[i][0])
    c.append(cvs['Confirmed'][i])
    d.append(cvs['Cured'][i])
    e.append(cvs['Deaths'][i])
labl['State'] = b
labl['Date'] = a
labl['Confirmed'] = c
labl['Deaths'] = e
labl['Cured'] = d  

labl['Confpday'] = labl['Confirmed']
labl['Deathspday'] = labl['Deaths']
labl['Curedpday'] = labl['Cured']

labl['Confirmed'][434]=128
labl.drop(labl.index[655:659],inplace = True)
labl.reset_index(inplace = True)

start = 0
for st in labl.State.unique():
    #print(st)
    for i in range(start+1,start+len(labl[labl.State==st])):
        if(labl['State'][i]==st):
            labl['Confpday'][i] = labl['Confirmed'][i] - labl['Confirmed'][i-1]
            labl['Deathspday'][i] = labl['Deaths'][i] - labl['Deaths'][i-1]
            labl['Curedpday'][i] = labl['Cured'][i] - labl['Cured'][i-1]  
    start = i+1
    #print(start)
    

dropin =  labl[labl.Confpday < 0].index
dropin1 = labl[labl.Deathspday < 0].index
dropin2 = labl[labl.Curedpday < 0].index
labl.drop(dropin,inplace = True)
labl.drop(dropin1,inplace = True)  
labl.drop(dropin2,inplace = True)      
#print(cvs)
#plt.plot(cvs[cvs.index == ' Kerala'][1],cvs[cvs.State == ' Kerala'].Confirmed)
#print(labl.State.nunique())
i = 1

for st in labl.State.unique():
    fig = plt.figure()
    #plt.rc('xtick',labelsize=2)
    lines = plt.plot(labl[labl.State== st].Date,labl[labl.State==st][['Confpday','Deathspday','Curedpday']])
    i+=1
    plt.legend(lines[:3], ('Confirmed', 'Deaths', 'Cured'))
    plt.title(st)

plt.show()

#print(len(labl[labl.State=='Unassigned']))



#Q3

q3 = pd.DataFrame()
pop = pd.read_csv('Population_india_census2011.csv')
pop = pop.rename(columns={'State / Union Territory': 'State'})
q3['cases'] = cv.groupby('State')['Confirmed'].max()
#plt.bar(q3.index,q3)
#plt.xticks(fontsize = 5)
print(q3['cases']['Kerala'])
q3['intensity'] = q3['cases']
q3.drop('Unassigned',inplace = True)


#print(re.findall("\d+", pop['Density'][34]))
def get_num(x):
    return float(''.join(ele for ele in x if ele.isdigit() or ele == '.'))

s = pop['Density'][34]
print(get_num(s[:5]))

for i in range(len(pop['Density'])):
    s = pop['Density'][i]
    pop['Density'][i] = get_num(s[:5])

'''      
lst = pop['Density']
subbed_list = [float(re.sub('\b\d.\d.\d\d\d','',i)) for i in lst]
'''

            
for st in q3.index:
    #print(q3['cases'][st]/pop[pop['State']== st].Density)
    q3['intensity'][st] = float(q3['cases'][st]/pop[pop['State']== st].Density)*100

plt.figure()
plt.bar(q3.index,q3['intensity'])
#street_map = gpd.read_file('Indian_States.shp')
#street_map.plot()
    
#Q6

q6dash = pd.read_csv('IndividualDetailsnew.csv')
q6 = q6dash[:7598]   
q6['notes'].fillna('Details awaited',inplace = True)
q6g = pd.DataFrame() 
q6g['Ncount'] = q6.groupby(['detected_state','notes'])['notes'].count()
q6g['Type'] = q6g['Ncount']

#primary = ['US','Italy','UK','London','Mecca','Ongle','Paris','Saudi Arabia','Stockholm','Washington','Bahrain','Dubai','Middle East','Qatar']                                      
countries = pd.read_csv('Countries.csv')
cntr = countries['country']
#q6g.reset_index()
#print(q6g[1])
sec = ['Delhi','Religious','Conference','delhi','Rajastan','rajastan','Kolkata','kolkata','P36','P35','P37']

for nstr in q6g.index:
    #print(q6g['Type'][nstr[0]])
    if any(ext in nstr[1] for ext in cntr):
       q6g['Type'][nstr] = 'Primary'
       #print(nstr[1])
    elif any(ext in nstr[1] for ext in sec):
        q6g['Type'][nstr] = 'Secondary'
    else:
        q6g['Type'][nstr] = 'Tertiary'    

q6g['State'] = q6g['Type']

for i in range(len(q6g.index)):
    q6g['State'][i] = q6g.index[i][0]


print(q6g[q6g.State=='Kerala'].Type.count())

q6gg = q6g.groupby(['State','Type'])['Ncount'].sum()
print(q6gg['Kerala']['Primary'])

ts = ['Maharashtra','Tamil Nadu','Delhi','Kerala','Telangana']

plt.figure()
rc('font', weight='bold')
# Values of each group
bars1 = [q6gg[ts[0]]['Primary'],q6gg[ts[1]]['Primary'],q6gg[ts[2]]['Primary'],q6gg[ts[3]]['Primary'],q6gg[ts[4]]['Primary']]
bars2 = [q6gg[ts[0]]['Secondary'],q6gg[ts[1]]['Secondary'],q6gg[ts[2]]['Secondary'],q6gg[ts[3]]['Secondary'],q6gg[ts[4]]['Secondary']]
bars3 = [q6gg[ts[0]]['Tertiary'], q6gg[ts[1]]['Tertiary'], q6gg[ts[2]]['Tertiary'], q6gg[ts[3]]['Tertiary'], q6gg[ts[4]]['Tertiary']]
 
# Heights of bars1 + bars2
bars = np.add(bars1, bars2).tolist()
 
# The position of the bars on the x-axis
r = [0,2,4,6,8]
 
# Names of group and bar width
names = ['Maharashtra','TN','Delhi','Kerala','Telangana']
barWidth = 1
 
# Create brown bars
b1 = plt.bar(r, bars1, color='firebrick', edgecolor='white', width=barWidth)
# Create green bars (middle), on top of the firs ones
b2 = plt.bar(r, bars2, bottom=bars1, color='salmon', edgecolor='white', width=barWidth)
# Create green bars (top)
b3 = plt.bar(r, bars3, bottom=bars, color='peachpuff', edgecolor='white', width=barWidth)

bf = []
bf.append(b1)
bf.append(b2)
bf.append(b3) 
# Custom X axis
plt.xticks(r, names, fontweight='bold')
plt.xlabel("State")
plt.ylabel('No.of Cases')
plt.legend(bf[:3],('Primary','Secondary','Tertiary'))
 
# Show graphic
plt.show()




#Q4

q6['status_change_date'].fillna(q6['diagnosed_date'][567],inplace = True)
for i in range(len(q6['diagnosed_date'])):
    temp = q6['diagnosed_date'][i]
    temp1 = q6['status_change_date'][i]
    q6['diagnosed_date'][i] = temp[0:8]
    q6['status_change_date'][i] = temp1[0:8]

for i in range(q6['diagnosed_date'].count()):
    q6['diagnosed_date'][i] = datetime.strptime(q6.loc[i].diagnosed_date, '%d/%m/%y').date()
    q6['status_change_date'][i] = datetime.strptime(q6.loc[i].status_change_date, '%d/%m/%y').date()

#q6['detected_district'].fillna('Unknown',inplace = True)

for i in range(len(q6['detected_district'])):
    if(pd.isnull(q6['detected_district'][i])):
        q6['detected_district'][i] = q6.loc[i].detected_state
        #print('yes')
q4 = pd.DataFrame()
q4 = q6.groupby(['diagnosed_date','detected_district','current_status','status_change_date'])['status_change_date'].count()



q4.sort_index(inplace = True)

print('Active Hotspots as of 04.04.20 is' )
test = q4[q4.index[-1][0]]
for ele in test.index:
    if(test[ele]>=10):
        print(ele[0])

#q4['14/03/20'].index[1][0] = 'abc'
dt = np.array(q6.diagnosed_date.unique())
#dt.append(q6.status_change_date.unique())

dtf = np.append(dt,q6.status_change_date.unique())
status = pd.DataFrame(index = q6.detected_district.unique() ,columns = np.unique(dtf))
status.fillna(0,inplace=True)


dtt = q6.diagnosed_date.unique()
status['Cumm20'] = status[dtt[0]]
status['Cumm27'] = status[dtt[0]]
status['Cumm4'] = status[dtt[0]]
status['Cumm10'] = status[dtt[0]]
#print(q4[dtt[0]])
     
for d in dtt:
    val = q4[d]
    #print(val)
    for i in range(len(val)):
        if(val.index[i][1]=='Recovered'):
            #a = status[d][val.index[i][0]]
            #status.loc[val.index[i][0]].d = a + val[i]
            status[d][val.index[i][0]]+=val[i]
            #print(status[d][val.index[i][0]],val[i],a,val.index[i][2])
            status[val.index[i][2]][val.index[i][0]]-=val[i]
            #print(status[val.index[i][2]][val.index[i][0]],status[d][val.index[i][0]])
        else:
            status[d][val.index[i][0]]+=val[i]

print(val.index[1][0])

mst = np.unique(dtf)
for dist in status.index:
    summ20 = 0
    summ27 = 0
    summ4 = 0
    summ10 = 0
    for dat in mst:
        if(dat!='Cumm10'):
            summ10+=status[dat][dist] 
            if(dat<=mst[22]):
                summ20+=status[dat][dist]
            if(dat<=mst[29]):
                summ27+=status[dat][dist]
            if(dat<=mst[37]):
                summ4+=status[dat][dist]
    status['Cumm20'][dist] = summ20
    status['Cumm27'][dist] = summ27
    status['Cumm4'][dist] = summ4
    status['Cumm10'][dist] = summ10

threshhold = 10 
#print(status[status.Cumm4>=threshhold].index)

#Q5

hots = pd.DataFrame(index = q6.detected_state.unique(),columns = ['20_mar','27_mar','4_apr','10_apr','week1','week2','week3'])

hots.fillna(0,inplace = True)

for rict in status[status.Cumm20>=threshhold].index:
    hots['20_mar'][q6[q6.detected_district==rict].detected_state.unique()]+=1
    
for rict in status[status.Cumm27>=threshhold].index:    
    hots['27_mar'][q6[q6.detected_district==rict].detected_state.unique()]+=1

for rict in status[status.Cumm4>=threshhold].index:    
    hots['4_apr'][q6[q6.detected_district==rict].detected_state.unique()]+=1

for rict in status[status.Cumm10>=threshhold].index:    
    hots['10_apr'][q6[q6.detected_district==rict].detected_state.unique()]+=1

hots['week1'] = hots['27_mar'] - hots['20_mar']
hots['week2'] = hots['4_apr'] - hots['27_mar']
hots['week3'] = hots['10_apr'] - hots['4_apr']

print(hots['week1'].idxmax(),hots['week2'].idxmax(),hots['week3'].idxmax())
#hots = hots.rename(index={'Tamil Nadu': 'Tamil_Nadu'})
#hots.set_index()
plt.plot(hots.columns[0:4],hots.loc['Tamil Nadu'][['20_mar','27_mar','4_apr','10_apr']])
plt.plot(hots.columns[0:4],hots.loc['Kerala'][['20_mar','27_mar','4_apr','10_apr']])
plt.legend(['Tamil Nadu','Kerala'])
#print(q6[q6.detected_district=='Coimbatore'].detected_state.unique())    
