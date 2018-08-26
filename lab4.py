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


def digitalGraph(signalBin, title, figura):
	
	visualBin = []
	bp = 0.001
	for bit in signalBin:
		for i in range(100):
			visualBin.append(bit)
	new_time = np.arange(bp/100, bp*len(visualBin) + bp/100, bp/100)
	plt.plot(new_time[:10000], visualBin[:10000])
	plt.ylim(-0.2,1.2)
	plt.xlabel("Tiempo")
	plt.ylabel("Amplitud")
	plt.title(title)
	plt.grid(True)
	savefig(figura)
	plt.show()

def OOKModulation(signalBin):

	print("2.- Comenzando la modulación OOK, espere un momento...")
	print("\tLa señal tiene ", len(signalBin), "bits")
	print("\tSe ha escogido un bit rate de 100 bits por segundo\n")
	A = 4
	bp = 0.001  #Periodo de bit
	br = 1/bp   #Bit rate
	f = br*10   #Cuantas ondas habrán en un tiempo de bit
	time = np.arange(bp/100,bp + bp/100,bp/100)
	modulated = []
	count = 0
	for bit in signalBin[:10000]:
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
	plt.plot(time2[:10000],modulated[:10000])
	savefig("modulacion_OOK")
	plt.show()
	return modulated, time, f

def OOKdemodulation(signalModulated,time, f):

	print("3.- Comenzando demodulación, espere un momento...\n")
	demodulated = []
	bp = 0.001
	len_modulated = len(signalModulated)
	aux = len(time)
	carrier = cos(2*pi*f*time)
	for i in range(aux, len_modulated + aux, aux):
		#Esto es para obtener los 100 bits que hay en un segundo para multiplicarlos por el carrier
		#cumpliendo la condición de ser vectores de igual tamaño.
		dm_aux = carrier*signalModulated[i-aux:i] 
		integral = integrate.trapz(dm_aux,time)
		integral2 = round((2*integral/bp))
		if(integral2 > 2): #Al ser OOK (A + 0)/2 = 2
			bit = 1
		else:
			bit = 0

		demodulated.append(bit)
	return demodulated



print("******Inicio del programa******")
y_signal, rate_signal, time_signal = getData("handel.wav")
binarySignal = getArrayBin(y_signal)

# Para temas de visualización y tiempo de ejecución
# se decide mostrar solo 5000 datos, ya que si quisiera 
# mostrar todos los datos tomaría mucho tiempo.
print("1.- Digitalizando la señal, espero un momento...\n")
digitalGraph(binarySignal[:10000],"Señal Original","senal_original")
modulated, time, f = OOKModulation(binarySignal)
demodulated = OOKdemodulation(modulated,time,f)
print("4.- Digitalizando la señal demodulada, espero un momento...\n")
digitalGraph(demodulated,"Señal al demodular","senal_demodulada")
print("******Fin del programa******")

