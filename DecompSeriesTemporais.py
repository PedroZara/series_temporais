import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime

# Registro de converters para uso do matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#%%

# Carregamento da base de dados, conversão do atributo para data e criação da série temporal (ts)
dateparse = lambda dates: datetime.strptime(dates, '%Y-%m')
df = pd.read_csv('AirPassengers.csv', parse_dates = ['Month'],
                 index_col = 'Month', date_parser = dateparse)
ts = df['#Passengers']

#%%

# Visualização da série temporal
plt.plot(ts)

#%%

# Decomposição da série temporal, criando uma variável para cada formato
decomposicao = seasonal_decompose(ts)

#%%

# Tendência
tendencia = decomposicao.trend

#%%

# Sazonalidade
sazonal = decomposicao.seasonal

#%%

# Erro
aleatorio = decomposicao.resid

#%%

# Visualização de gráfico para cad aformato da série temporal
plt.plot(sazonal)

#%%

plt.plot(tendencia)

#%%

plt.plot(aleatorio)

#%%

plt.subplot(4,1,1)
plt.plot(ts, label = 'Original')
plt.legend(loc = 'best')

# Visualização apenas da tendência
plt.subplot(4,1,2)
plt.plot(tendencia, label = 'Tendência')
plt.legend(loc = 'best')

# Visualizaão da sazonalidade
plt.subplot(4,1,3)
plt.plot(sazonal, label = 'Sazonalidade')
plt.legend(loc = 'best')

# Visualização somente do elemento aleatório
plt.subplot(4,1,4)
plt.plot(aleatorio, label = 'Aleatório')
plt.legend(loc = 'best')

plt.tight_layout()
