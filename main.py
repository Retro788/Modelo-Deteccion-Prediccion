import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import signal

f = open("code\XB.ELYSE.02.BHV.2021-05-02HR01_evid0017.csv")
dataraw = f.readlines()
wsize = 50
data = []
xaxis = []
yaxis = []
yaxis_abs = []
yaxis_abs_wavg = [] # window z abs
xaxis_abs_wavg = []
yaxis_tempo = [] #pochodna
xaxis_tempo = []
avg_tempo = 0 #średnia z pochodnej
yaxis_above_avg_tempo = []  #dodaję tutaj te, które są większe od średniej pochodnej
yaxis_above_avg_tempo_wavg = []
xaxis_above_avg_tempo_wavg = []
for x in dataraw:
    data.append(x.split(','))
print(data[3])
for x in data[1:]:
    xaxis.append(float(x[1]))
    yaxis.append(float(x[2]))


for x in data[2:]:
    yaxis_abs.append(abs(float(x[2])))
    #print(x[1],x[2],end='')
    #print(float(x[1])/3600)
for i in range(0,math.ceil((len(xaxis)-wsize)/wsize)):
    #print(i,xaxis[i*wsize])
    yaxis_abs_wavg.append(sum((yaxis[i*wsize:(i+1)*wsize]))/wsize)
    xaxis_abs_wavg.append(i*wsize)

for i in range(0,len(yaxis)-1):
    yaxis_tempo.append(abs(yaxis[i+1] - yaxis[i]))
    xaxis_tempo.append(i)

avg_tempo = sum(yaxis_tempo)/len(yaxis_tempo)*2

print("2x avg tempo = ", avg_tempo)



print(len(yaxis),len(yaxis_tempo))
for i in range(len(yaxis_tempo)):
    if(yaxis_tempo[i] > avg_tempo):
        yaxis_above_avg_tempo.append(abs(yaxis[i]))
    else:
        yaxis_above_avg_tempo.append(0)

for i in range(len(yaxis_above_avg_tempo)//wsize):
    yaxis_above_avg_tempo_wavg.append(max(yaxis_above_avg_tempo[i*wsize:i*wsize+wsize]) / wsize)
    xaxis_above_avg_tempo_wavg.append(i)
    print(i)


fig, axs = plt.subplots(4,figsize=(19,5))

axs[0].plot(xaxis, yaxis)
axs[0].set(xlabel='time (s)', ylabel='amplitude (eee?)',title='original_data')
axs[0].grid()

axs[1].plot(xaxis_tempo, yaxis_tempo)
axs[1].set(xlabel='ticks', ylabel='amplitude change',title='tempo')
axs[1].grid()
axs[1].axhline(y = avg_tempo, color = 'r', linestyle = '-') 

axs[2].plot(xaxis_tempo, yaxis_above_avg_tempo)
axs[2].set(xlabel='ticks', ylabel='amplitude (eee?)',title='tempo')
axs[2].grid()

axs[3].plot(xaxis_above_avg_tempo_wavg, yaxis_above_avg_tempo_wavg)
axs[3].set(xlabel='ticks', ylabel='amplitude (eee?)',title='tempo')
axs[3].grid()

value = 100 # random values

for i in range(1,len(yaxis_above_avg_tempo_wavg)):
    if(abs(yaxis_above_avg_tempo_wavg[i]-yaxis_above_avg_tempo_wavg[i-1])>value):
        axs[3].axvline(x=i, color = 'r', linestyle = '-')




fig.savefig("test.png",dpi = 300)
