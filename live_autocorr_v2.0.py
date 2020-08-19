from numpy import *
import pyaudio
import matplotlib.pyplot as plt
import time
from scipy import signal
from scipy.fftpack import fft, ifft, ifftshift
import matplotlib as mpl
import pandas as pd
mpl.style.use('ggplot')

# Set program parameters
sampling_rate = 44100
window_length = 1/16
noise_db_threshold = 25

update_period = 1/1000
raw_signal_range = 6000
fft_signal_range = 1
potentail_freq_range = [30,2000]


# Define fast autocorrelation function
def autocorr(series):
	temp2 = fft(list(series)+list(zeros(len(series)-1)))
	temp3 = real(ifft(abs(temp2)))
	autocorr = temp3[:len(series)//2]/(arange(len(series)//2)[::-1]+len(series)//2)	
	return autocorr/max(autocorr)

# Define all note frequencies
notes_current = pd.DataFrame((('C1',32.70),('C1#',34.65),('D1',36.71),('D1#',38.89),('E1',41.20),('F1',43.65),('F1#',46.25),('G1',49.00),('G1#',51.91),('A1',55.00),('A1#',58.27),('B1',61.74)))
notes_current.columns = ['note','freq']
all_notes = notes_current.copy()
for k in range(2,7):
	notes_current['note'] = [x[:1]+str(k)+x[2:] for x in notes_current['note'].values]
	notes_current['freq'] = [2*x for x in notes_current['freq'].values]
	all_notes = all_notes.append(notes_current)

# First gauge room noise, will only detect signals 
audio = pyaudio.PyAudio()
noise_level = 10*log10(9**2)

# Set up actual audio stream for tuning
buffer_size = int(window_length*sampling_rate)
stream = audio.open(
	format=pyaudio.paInt16, 
	channels=1, 
	rate=sampling_rate, 
	input=True,
	frames_per_buffer=buffer_size
	)

# Set up base plot
fig = plt.figure(figsize=(14,4), dpi=100)
x1 = 1000*linspace(0,window_length,buffer_size)
x2 = arange(buffer_size//2)
y1 = random.randn(buffer_size)
y2 = random.randn(buffer_size//2)

ax1 = fig.add_subplot(1,3,1)
ax1.set_ylim([-raw_signal_range,raw_signal_range])
ax1.set_title('Raw Audio Signal')
ax1.set_yticks([])
line1, = ax1.plot(x1,y1,'k-')
ax1.set_xticklabels([str(int(x))+'ms' for x in ax1.get_xticks()])

ax2 = fig.add_subplot(1,3,2)
ax2.set_ylim([-fft_signal_range,fft_signal_range])
ax2.set_title('Signal Autocorrelation')
line2, = ax2.plot(x2,y2,'k--')
line3, = ax2.plot(x2,y2,'b-')
ax2.set_xlim(0,len(x2))
ax2.set_xticklabels([str(int(1000*x/sampling_rate))+'ms' for x in ax2.get_xticks()])
point1, = ax2.plot(-5,5,'rs')
test1 = ax2.text(0.4, 0.85, 'Main Frequency: --\nMax Degree of fit: ', horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes, 
	bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

ax3 = fig.add_subplot(1,3,3)
line4, = ax3.plot(arange(200),arange(200),'r-')
ax3.set_xlim([-2,0])
ax3.set_ylim([0,100])

fig.tight_layout()
# plt.pause(update_period)


running_signal_volume = []
for k in range(20):

	# Read in audio data
	last_audio_chunk = array(frombuffer(stream.read(buffer_size, exception_on_overflow=False), int16), float)
	chunk_volume = 10*log10(mean(last_audio_chunk**2))
	line1.set_ydata(last_audio_chunk)

	# Transform raw audio data and plot
	signal_autocorr = autocorr(last_audio_chunk)
	line2.set_ydata(signal_autocorr)

	# Update running list of sound magnitudes
	curr_time = time.time()
	num_split = 2
	split_audio_chunks = [last_audio_chunk[i:i + int(floor(buffer_size/num_split))] for i in range(0, buffer_size, int(floor(buffer_size/num_split)))]
	running_signal_volume += [(curr_time-(i/num_split)*buffer_size/sampling_rate,10*log10(mean(x**2))) for x,i in zip(split_audio_chunks,arange(num_split-1,-1,-1))]
	running_signal_volume = [x for x in running_signal_volume if x[0]-curr_time>-2]
	num_udpate = min(len(running_signal_volume),len(line4.get_xdata()))
	line4.set_xdata([x[0]-curr_time for x in running_signal_volume[-num_udpate:]]+(len(running_signal_volume)-num_udpate)*[0])
	line4.set_ydata([x[1] for x in running_signal_volume[-num_udpate:]]+(len(running_signal_volume)-num_udpate)*[0])

	# If there is more than just noise, analyze signal
	if chunk_volume-noise_db_threshold>noise_level:

		# Determine primary frequency and note using peak method
		first_maximum_peak = argmax(signal_autocorr[50:])+50
		main_freq = round(sampling_rate/first_maximum_peak)		
		curr_note = all_notes['note'].values[argmin(abs(all_notes['freq'].values-main_freq))]

		# Find best fitting theoretical wave
		wave_sine = autocorr(sin(main_freq*(x1/1000)*2*pi))
		wave_square = autocorr(signal.square(main_freq*(x1/1000)*2*pi))
		wave_sawtooth = autocorr(signal.sawtooth(main_freq*(x1/1000)*2*pi))
		best_wave, best_fit = max([(x,corrcoef(x,signal_autocorr)[0,1]) for x in [wave_sine,wave_square,wave_sawtooth]], key=lambda t: t[1])

		# Update graph
		point1.set_xdata(first_maximum_peak)
		point1.set_ydata(signal_autocorr[first_maximum_peak])
		test1.set_text(f'Main Frequency: {main_freq}Hz ({curr_note})\nMax Degree of fit: {int(100*best_fit)}%')
		line3.set_ydata(best_wave)


	else:
		line3.set_ydata(-1000*ones(len(y2)))
		point1.set_xdata(-100)
		point1.set_ydata(0)
		test1.set_text(f'Main Frequency: --\nMax Degree of fit: --')

	# plt.pause(update_period)

temp = [x[0]-time.time() for x in running_signal_volume]
