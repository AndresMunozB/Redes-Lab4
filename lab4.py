# -*- coding: utf-8 -*-
"""==============================================================================
Created on Sat May  5 18:06:02 2018
Integrantes:
    Diego Mellis - 18.663.454-3
    Andrés Muñoz - 19.646.487-5
=============================================================================="""

from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt
from numpy import arange, linspace, cos, pi,random
from pylab import savefig
from scipy.fftpack import fft, ifft, fftshift
import warnings 
from scipy import integrate
from math import pi
from matplotlib.ticker import NullFormatter
warnings.filterwarnings('ignore')


"""==============================================================================
Función: En base a los datos que entrega el un archivo con el nombre "nameFile" 
		 se obtiene los datos de la señal, la cantidad de datos que esta tiene, 
 		 y el tiempo que dura el audio.
Entrada: nameFile -> Nombre del archivo 
Salida:  y_signal -> Valores de la amplitud de la señal (Eje Y) 
		 rate_signal -> frecuencia de la señal.
		 time_signal -> Valores del tiempo de la señal (Eje X)
=============================================================================="""
def getData(nameFile):
	rate_signal,y_signal=read(nameFile)
	len_signal = len(y_signal)
	time_signal = len_signal/rate_signal
	return y_signal, rate_signal, time_signal

"""==============================================================================
Función: Dada una señal, se obtiene la cantidad de bits para poder representar sus datos.
Entrada: y_signal -> Señal de la cual se obtiene la cantidad maxima de bits.
Salida:  len_bin -> Cantidad de bits necesesarios para representar todos los datos.
=============================================================================="""
def getBits(y_signal):
	maximum = max(y_signal)
	maxbin = format(maximum,"b")
	len_bin =  len(maxbin)+1
	return len_bin #PARA AÑADIR UN BIT DE SIGNO NEGATIVO


"""==============================================================================
Función: Función que rellena de ceros y modifica el formato de un número negativo binario.
Entrada: value -> valor que se desea modificar
		 bits  -> Cantidad de bits que debe tener el nuevo número
Salida:  binvalue -> valor resultante al remplazar el "-" por el "1" y rellenar con  
					 ceros hasta obtener un largo igual a "bits".
=============================================================================="""
def binarize(value, bits):
	binvalue = format(value,"b").zfill(bits)
	if(binvalue[0]=="-"):
		binvalue = "1"+binvalue[1:]
	return binvalue

"""==============================================================================
Función: Función que se encarga de transformar la señal a una señal digital.
Entrada: y_signal -> Señal que se transformará a una señal digital.
Salida : signalBin -> Señal digitalizada, es decir, tiene solo valores 0 ó 1.
=============================================================================="""
def getArrayBin(y_signal):
	signalBin = []
	maxbits = getBits(y_signal)
	for value in y_signal:
		binvalue = binarize(value,maxbits)
		for bit in binvalue:
			signalBin.append(int(bit))
	return signalBin

"""==============================================================================
Función: Función que grafica y guarda una señal digital.
Entrada: signalBin -> Señal digital que se desea graficar.
		 title     -> Título del gráfico.
		 figura    -> Nombre del archivo que se guardará.
		 datos	   -> Cantidad de bits a mostrar
=============================================================================="""
def digitalGraph(signalBin, title, figura,datos):
	
	visualBin = []
	bp = 0.001
	for bit in signalBin:
		for i in range(100):
			visualBin.append(bit)
	new_time = np.arange(bp/100, bp*len(visualBin) + bp/100, bp/100)
	plt.plot(new_time[:datos], visualBin[:datos])
	plt.ylim(-0.2,1.2)
	plt.xlabel("Tiempo")
	plt.ylabel("Amplitud")
	plt.title(title)
	plt.grid(True)
	savefig(figura)
	plt.show()

"""==============================================================================
Función: Función que grafica y guarda una una señal modulada con o sin ruido.
Entrada: time2 	   -> Eje x del gráfico.
		 modulated -> Señal modulada que se desea graficar.
		 title     -> Título del gráfico.
		 figura    -> Nombre del archivo que se guardará.
=============================================================================="""
def modulatedGraph(time2,modulated,title,figura,datos):
	#Aquí se comienza a graficar la señal modulada
	plt.ylim(-5,5)
	plt.xlabel("Tiempo")
	plt.ylabel("Amplitud")
	plt.title(title)
	plt.grid(True)
	plt.plot(time2[:datos],modulated[:datos])
	savefig(figura)
	plt.show()


def transformData(signalBin,bp,datos):
	visualBin = []
	bp = 0.001
	for bit in signalBin[:datos]:
		for i in range(100):
			visualBin.append(bit)
	new_time = np.arange(bp/100, bp*len(visualBin) + bp/100, bp/100)
	return new_time,visualBin



def graphOriginalAndDemodulate(binarySignal,demodulated,bp,datos, title, figura):


	graph1_x, graph1_y = transformData(binarySignal,bp,datos)
	graph4_x, graph4_y = transformData(demodulated,bp,datos)

	plt.subplot(2, 1, 1)
	plt.ylim(-0.2,1.2)
	plt.grid(True)
	plt.plot(graph1_x[:datos], graph1_y[:datos],linewidth=0.4)
	plt.title(title)
	plt.ylabel('Amplitud')

	plt.subplot(2, 1, 2)
	plt.ylim(-0.2,1.2)
	plt.grid(True)
	plt.plot(graph4_x[:datos], graph4_y[:datos],linewidth=0.4)
	plt.xlabel('Tiempo')
	plt.ylabel('Amplitud')
	savefig(figura)
	plt.show()



def graphModulated(time2,modulated,modulatedWithNoise,datos, title, figura):


	plt.subplot(2, 1, 1)
	plt.plot(time2[:datos], modulated[:datos],linewidth=0.4)
	plt.title(title)
	plt.ylabel('Amplitud')

	plt.subplot(2, 1, 2)
	plt.plot(time2[:datos], modulatedWithNoise[:datos],linewidth=0.4)
	plt.xlabel('Tiempo')
	plt.ylabel('Amplitud')
	savefig(figura)
	plt.show()

"""==============================================================================
Función: Función que agrega ruido a una señal.
Entrada: signal -> Señal a la cual se le agregará ruido.
		 snr    -> Relación entre la señal y el ruido.
Salida : signal_with_noise -> Señal con ruido agregado.
		 noise -> ruido para la señal.
=============================================================================="""
def addNoise(signal, snr):
    noise = random.normal(0.0, 1.0/snr, len(signal))
    signal_with_noise = signal + noise
    return signal_with_noise, noise


"""==============================================================================
Función: Función que modula una señal digital con el método OOK.
Entrada: signalBin -> Señal que se modulará.
Salida : modulated -> Señal modulada. (Eje Y)
		 time      -> Intervalo de tiempo en un periodo de bit.
		 f         -> Frecuencia de modulación.
		 time2     -> Intervalo de valores de tiempo para la nueva señal modulada.
=============================================================================="""
def OOKModulation(signalBin,datos):

	
	A = 4
	bp = 0.001  #Periodo de bit
	br = 1/bp   #Bit rate
	f = br*10   #Cuantas ondas habrán en un tiempo de bit
	time = np.arange(bp/100,bp + bp/100,bp/100)
	modulated = []
	count = 0
	for bit in signalBin[:datos]:
		count +=1
		if bit==1:
			modulated = np.concatenate((modulated,A*cos(2*pi*f*time)))
		else:
			modulated = np.concatenate((modulated,0*cos(2*pi*f*time)))
	time2 = np.arange(bp/100,bp*count + bp/100, bp/100)
	return modulated, time, f,time2

"""==============================================================================
Función: Función que demodula una señal digital modulada.
Entrada: signalModulated -> Señal que se modulará.
		 time -> Intervalo de tiempo en un periodo de bit.
		 f         -> Frecuencia de modulación.
Salida : demodulated -> Señal demodulada. 
=============================================================================="""
def OOKdemodulation(signalModulated,time, f):

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
		if(integral2 >= 2.0): #Al ser OOK (A + 0)/2 = 2
			bit = 1
		else:
			bit = 0

		demodulated.append(bit)
	return demodulated

def obtainError(signalBin, demodulated):
	count = 0
	for i in range(len(signalBin)):
		if signalBin[i] != demodulated[i]:
			count +=1
	return (count*100)/len(signalBin)


def graphSNR(SNRList, errors, figura):
	plt.plot(SNRList,errors)
	plt.ylabel("Tasa de error (%)")
	plt.xlabel("SNR")
	plt.title("Tasa de error vs SNR")
	savefig(figura)
	plt.show()

"""==============================================================================
Función: Que imprime las opciones del menú.
=============================================================================="""
def printMenu():
	print("		Menu\n")
	print("1) Mostrar Señal Digital Original")
	print("2) Mostrar Señal Digital Modulada")
	print("3) Mostrar estudio con SNR = 0.5")
	print("4) Mostrar estudio con SNR = 0.1")
	print("5) Mostrar estudio con SNR = 0.05")
	print("6) Mostrar estudio con SNR = 0.025")
	print("7) Mostrar estudio con SNR = 0.0125")
	print("8) Mostrar gráfico de tasas de errores según SNR")
	

	print("9) Salir\n\n")



#BLOQUE PRINCIPAL

# Para temas de visualización y tiempo de ejecución
# se decide mostrar solo 5000 datos, ya que si quisiera 
# mostrar todos los datos tomaría mucho tiempo.
datos = 10000
bp = 0.001
print("******Iniciando programa******")
#  0) Se cargan los datos de la señal que se desea modular y demodular.
y_signal, rate_signal, time_signal = getData("handel.wav")
binarySignal = getArrayBin(y_signal)
snr = 10

#  1) Se modula la señal 
print("1.- Comenzando la modulación OOK, espere un momento...")
print("\tLa señal tiene ", len(binarySignal), "bits")
print("\tSe ha escogido un bit rate de 100 bits por segundo\n")
modulated, time, f,time2 = OOKModulation(binarySignal,datos)

#  2) Se añade ruido a la señal
print("2.- Calculando errores, espere un momento...")
print("\tLos SNR usados son: 0.5, 0.1, 0.05, 0.025 y 0.0125")
SNRList = [0.2, 0.1, 0.05, 0.025, 0.0125]
errors = []
for snr in SNRList:
	modulatedWithNoise, noise = addNoise(modulated, snr)
	demodulated = OOKdemodulation(modulatedWithNoise,time,f)
	error = obtainError(binarySignal[:datos], demodulated)
	errors.append(error)







#  3) Aquí se muestra un menú para graficar los resultados obtenidos.

menu = "0"
printMenu()
while(True):
	
	menu = str(input("Ingrese una opcion: "))
	if(menu == "1"):
		digitalGraph(binarySignal[:datos],"Señal Original","senal_original",datos)
	elif(menu == "2"):
		modulatedGraph(time2,modulated,"Señal Modulación OOK","modulacion_OOK" ,datos)
	elif(menu == "3"):
		modulatedWithNoise, noise = addNoise(modulated, 0.2)
		demodulated = OOKdemodulation(modulatedWithNoise,time,f)
		graphModulated(time2,modulated,modulatedWithNoise,datos,"Señal Modulada y Señal con Ruido SNR = 0.2","Ruido02")
		graphOriginalAndDemodulate(binarySignal,demodulated,bp,datos, "Señal Original y Demodulada con SNR = 0.2","02SNR")
		
	elif(menu == "4"):
		modulatedWithNoise, noise = addNoise(modulated, 0.1)
		demodulated = OOKdemodulation(modulatedWithNoise,time,f)
		graphModulated(time2,modulated,modulatedWithNoise,datos,"Señal Modulada y Señal con Ruido SNR = 1","Ruido1")
		graphOriginalAndDemodulate(binarySignal,demodulated,bp,datos, "Señal Original y Demodulada con SNR = 0.1","01SNR")
		
	elif(menu == "5"):
		modulatedWithNoise, noise = addNoise(modulated, 0.05)
		demodulated = OOKdemodulation(modulatedWithNoise,time,f)
		graphModulated(time2,modulated,modulatedWithNoise,datos,"Señal Modulada y Señal con Ruido SNR = 0.05","Ruido005")
		graphOriginalAndDemodulate(binarySignal,demodulated,bp,datos, "Señal Original y Demodulada con SNR = 0.05","005SNR")
		
	elif(menu == "6"):
		modulatedWithNoise, noise = addNoise(modulated, 0.025)
		demodulated = OOKdemodulation(modulatedWithNoise,time,f)
		graphModulated(time2,modulated,modulatedWithNoise,datos,"Señal Modulada y Señal con Ruido SNR = 0.025","Ruido0025")
		graphOriginalAndDemodulate(binarySignal,demodulated,bp,datos, "Señal Original y Demodulada con SNR = 0.025","0025SNR")
		
	elif(menu == "7"):
		modulatedWithNoise, noise = addNoise(modulated, 0.0125)
		demodulated = OOKdemodulation(modulatedWithNoise,time,f)
		graphModulated(time2,modulated,modulatedWithNoise,datos,"Señal Modulada y Señal con Ruido SNR = 0.0125","Ruido00125")	
		graphOriginalAndDemodulate(binarySignal,demodulated,bp,datos, "Señal Original y Demodulada con SNR = 0.0125","00125SNR")
		
	elif(menu == "8"):
		graphSNR(SNRList,errors,"SNRErrors")		
	elif(menu == "9"):	
		break
