from numpy import *
import pyaudio
import matplotlib.pyplot as plt
import time
from scipy import signal
from scipy.fftpack import fft, ifft, ifftshift
import matplotlib as mpl
import pandas as pd
mpl.style.use('ggplot')


# Define all note frequencies
notes_current = pd.DataFrame((('C1',32.70),('C1#',34.65),('D1',36.71),('D1#',38.89),('E1',41.20),('F1',43.65),('F1#',46.25),('G1',49.00),('G1#',51.91),('A1',55.00),('A1#',58.27),('B1',61.74)))
notes_current.columns = ['note','freq']
all_notes = notes_current.copy()
for k in range(2,7):
	notes_current['note'] = [x[:1]+str(k)+x[2:] for x in notes_current['note'].values]
	notes_current['freq'] = [2*x for x in notes_current['freq'].values]
	all_notes = all_notes.append(notes_current)


freq1 = all_notes[all_notes['note']=='A4']['freq'].values[0]
freq2 = all_notes[all_notes['note']=='F4']['freq'].values[0]




x = linspace(0,.03,1000)
y1 = sin(2*pi*freq1*x)
y2 = sin(2*pi*freq2*x)

temp1 = y1
temp2 = fft(list(temp1)+list(zeros(len(temp1)-1)))
temp3 = real(ifft(abs(temp2)))
autocorr = temp3[:len(temp1)//2]/(arange(len(temp1)//2)[::-1]+len(temp1)//2)	
autocorr = autocorr/max(autocorr)	
plt.plot(autocorr)
plt.show()

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(1,1,1)
plt.plot(x,y1+y2)
plt.show()