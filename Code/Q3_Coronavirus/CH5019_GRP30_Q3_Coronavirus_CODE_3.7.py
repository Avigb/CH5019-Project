import pandas as pd
df = pd.read_csv('ICMRTestingDetails.csv')

spday = []
cpday = []
spday.append('nan')
cpday.append('nan')
for i in range(1,len(df)):
    spday.append(df.loc[i]['TotalSamplesTested']-df.loc[i-1]['TotalSamplesTested'])
    cpday.append(df.loc[i]['TotalPositiveCases']-df.loc[i-1]['TotalPositiveCases'])
df['SamplesPerDay']=spday
df['CasesPerDay']=cpday

df.drop(['TotalSamplesTested','TotalPositiveCases','TotalIndividualsTested'],axis=1,inplace = True)
ratio = []
sample = []
cases = []
time = []
for i in range(23,34):
    ratio.append(df.loc[i]['SamplesPerDay']/df.loc[i]['CasesPerDay'])
    sample.append(df.loc[i]['SamplesPerDay'])
    cases.append(df.loc[i]['CasesPerDay'])
    time.append(df.loc[i]['DateTime'])
#print(ratio)
sum = 0
for val in ratio:
    sum = sum + val
movratio = sum/len(ratio)
#print(movratio)

ds = pd.DataFrame()
ds['DateTime']= time
ds['SamplesPerDay']= sample
ds['CasesPerDay']= cases
ds['ratio'] = ratio

print(ds)



    