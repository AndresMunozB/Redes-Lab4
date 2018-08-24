# -*- coding: utf-8 -*-
"""
Created on Sat May  5 18:06:02 2018
Integrantes:
    Diego Mellis - 18.663.454-3
    Andrés Muñoz - 19.646.487-5
"""

from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt
from numpy import arange, linspace, cos, pi
from pylab import savefig
from scipy.fftpack import fft, ifft, fftshift
import warnings 
from scipy import integrate
from math import pi
warnings.filterwarnings('ignore')

#==============================================================================
# Función: En base a los datos que entrega handel.wav se obtiene			   
# los datos de la señal, la cantidad de datos que esta tiene, y el tiempo que  
# que dura el audio.														   
# Parámetros de entrada: Matriz con los datos de la amplitud del audio.
# Parámetros de salida: Vector con la señal a trabajar, el largo de la señal y 
# un vector con los tiempos de la señal, su frecuencia de muestro y su tiempo.
#==============================================================================
def getData(nameFile):
	rate_signal,y_signal=read(nameFile)
	return y_signal, rate_signal


def getBits(y_signal):
	maximum = max(y_signal)
	maxbin = format(maximum,"b")
	return len(maxbin)+1 #PARA AÑADIR UN BIT DE SIGNO NEGATIVO

def binarize(value, bits):
	binvalue = format(value,"b").zfill(bits)
	if(binvalue[0]=="-"):
		binvalue = "1"+binvalue[1:]
	return binvalue

def getArrayBin(y_signal):
	signalBin = []
	maxbits = getBits(y_signal)
	for value in y_signal:
		binvalue = binarize(value,maxbits)
		for bit in binvalue:
			signalBin.append(int(bit))
	return signalBin

def OOKModulation(signalBin):

	A = 4
	B = 5
	bp = 1
	br = 1/bp
	f = br*10
	#t = np.arange(0,10,1)
	t = np.arange(bp/100,bp + bp/100,bp/100)
	ss = len(t)
	modulated = []
	count = 0
	for bit in signalBin[:5000]:
		count +=1
		if bit==1:
			modulated = np.concatenate((modulated,A*cos(2*pi*f*t)))
		else:
			modulated = np.concatenate((modulated,0*cos(2*pi*f*t)))
	t2 = np.arange(bp/100,bp*count + bp/100, bp/100)
	return modulated, t2



y_signal, rate_signal = getData("handel.wav")
binarySignal = getArrayBin(y_signal)
mod, t = OOKModulation(binarySignal)
print(len(mod))
print(len(t))
plt.plot(t[:5000],mod[:5000])
plt.show()