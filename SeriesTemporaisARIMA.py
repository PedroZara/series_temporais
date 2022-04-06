import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
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

# Criação do modelo ARIMA com os parametros p = 2, q = 1, d = 2, treinamento e visualização dos resultados
modelo = ARIMA(ts, order=(2, 1, 2), freq = ts.index.inferred_freq)
modelo_treinado = modelo.fit()
mod_treinado_sumario = modelo_treinado.summary()

#%%

# Previsões de 12 datas futuras
previsoes = modelo_treinado.forecast(steps = 12)[0]

#%%

# Criação de um eio para a série temporal completa, com adição das previsões do modelo
# lot_insample = True dados originais
eixo = ts.plot()
modelo_treinado.plot_predict('1960-01-01', '1965-01-01', ax= eixo, plot_insample = True)

#%%

# Implementação do auto arima para descoberta automática de parâmetros
modelo_auto = auto_arima(ts, m = 12, seasonal = True, trace = False)
mod_auto__sumario = modelo_auto.summary()

#%%

proximos_12 = modelo_auto.predict(n_periods = 12)
# Visualização dos próximos 12 valores
print(proximos_12)
plt.plot(proximos_12)

