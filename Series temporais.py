import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Registro de converters para uso do matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#%%

# Importando a base de dados
df = pd.read_csv('AirPassengers.csv')

#%%

# Visualização do tipo dos dados atrobuidos
print(df.dtypes)

#%%

# Conversão dos atributos que estão no formato string para formato de data: Ano-Mês
dateparse = lambda dates: datetime.strptime(dates, '%Y-%m')
df = pd.read_csv('AirPassengers.csv', parse_dates = ['Month'],
                 index_col = 'Month', date_parser = dateparse)

#%%

# Visualização do índice do dataframe (#Passengers)
df_indice = df.index

#%%

# Criação da série temporal (ts)
ts = df['#Passengers']

#%%

# Visualizando registro específico
ts_1 = ts[1]

#%%

# Visualização por ano e mês
ts_anomes = ts['1949-02']

#%%

# Visualização de data específica
ts_data = ts[datetime(1949,2,1)]

#%%

# Visualização de intervalos
ts_inter = ts['1950-01-01' : '1950-07-31']

#%%

# Visualização de intervalos sem preencher a data de início
ts_inter2 = ts[:'1950-07-31']

#%%

# Visualização por ano
ts_ano = ts['1950']

#%%

# Valores maximos
ts_max = ts.index.max()

#%%

# Valores mínimos
ts_min = ts.index.min()

#%%

# Visualização da série temporal completa
plt.plot(ts)

#%%

# Visualização por ano
ts_ano1 = ts.resample('A').sum()
plt.plot(ts_ano1)

#%%

# Visualização por mês
ts_mes1 = ts.groupby([lambda x: x.month]).sum()
plt.plot(ts_mes1)

#%%

# Visualização entre datas específicas
ts_datas = ts['1960-01-01' : '1960-12-01']
plt.plot(ts_datas)
