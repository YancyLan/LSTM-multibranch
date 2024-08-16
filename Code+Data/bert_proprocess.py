import pandas as pd
data1 = pd.read_csv('textfeature2018.csv')
data2 = pd.read_csv('textfeature2019.csv')
data3 = pd.read_csv('textfeature2020.csv')
data4 = pd.read_csv('textfeature2021.csv')
data5 = pd.read_csv('textfeature2022.csv')
data = pd.concat([data1,data2,data3,data4,data5])
date = pd.read_excel('交易日.xlsx')
dd = date['交易日期'].tolist()
data = data[data['公布日期'].isin(dd)]
data.to_csv('textfeature.csv',index=0)
print(data)