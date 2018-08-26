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
	len_signal = len(y_signal)
	time_signal = len_signal/rate_signal
	return y_signal, rate_signal, time_signal


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


def digitalGraph(signalBin, time_signal):
	visualBin = []
	for bit in signalBin:
		for i in range(10):
			visualBin.append(bit)
	new_time = np.linspace(0, time_signal, len(visualBin))
	plt.plot(new_time[:1000], visualBin[:1000])
	plt.ylim(-0.2,1.2)
	plt.xlabel("Tiempo")
	plt.ylabel("Amplitud")
	plt.title("Señal Digital")
	plt.grid(True)
	savefig("senal_digital")
	plt.show()

def OOKModulation(signalBin):

	A = 4
	bp = 0.001 #Periodo de bit
	br = 1/bp   #Bit rate
	f = br*10    #Cuantas ondas habrán en un tiempo de bit
	time = np.arange(bp/100,bp + bp/100,bp/100)
	modulated = []
	count = 0
	for bit in signalBin[:5000]:
		count +=1
		if bit==1:
			modulated = np.concatenate((modulated,A*cos(2*pi*f*time)))
		else:
			modulated = np.concatenate((modulated,0*cos(2*pi*f*time)))
	time2 = np.arange(bp/100,bp*count + bp/100, bp/100)
	
	#Aquí se comienza a graficar la señal modulada
	plt.ylim(-5,5)
	plt.xlim(0, 0.05)
	plt.xlabel("Tiempo")
	plt.ylabel("Amplitud")
	plt.title("Señal Modulación OOK")
	plt.grid(True)
	plt.plot(time2[:5000],modulated[:5000])
	savefig("modulacion_OOK")
	plt.show()




print("Inicio del programa")
y_signal, rate_signal, time_signal = getData("handel.wav")
binarySignal = getArrayBin(y_signal)
digitalGraph(binarySignal,time_signal)
OOKModulation(binarySignal)

