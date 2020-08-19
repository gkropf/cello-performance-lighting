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
noise_db_threshold = 2

update_period = 1/1000
raw_signal_range = 6000
fft_signal_range = 2
potentail_freq_range = [30,2000]

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
print('Gauging noise level of the room. Please remain quiet.')
# temp_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sampling_rate, input=True, frames_per_buffer=1*sampling_rate)
# noise_sample = frombuffer(temp_stream.read(1*sampling_rate, exception_on_overflow=False), int16)
noise_level = 9#mean(abs(noise_sample))
# temp_stream.close()


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
fig = plt.figure(figsize=(12,4), dpi=100)
x1 = 1000*linspace(0,window_length,buffer_size)
x2 = arange(buffer_size//2)
y1 = random.randn(buffer_size)
y2 = random.randn(buffer_size//2)

ax1 = fig.add_subplot(1,2,1)
ax1.set_ylim([-raw_signal_range,raw_signal_range])
ax1.set_title('Audio Signal (Time Domain)')
ax1.set_yticks([])
line1, = ax1.plot(x1,y1,'k-')
ax1.set_xticklabels([str(int(x))+'ms' for x in ax1.get_xticks()])

ax2 = fig.add_subplot(1,2,2)
ax2.set_ylim([-fft_signal_range,fft_signal_range])
ax2.set_title('Signal Autocorrelation')
line2, = ax2.plot(x2,y2,'k--')
ax2.set_xlim(0,1000)
ax2.set_xticklabels([str(int(1000*x/sampling_rate))+'ms' for x in ax2.get_xticks()])
point2, = ax2.plot(-5,5,'rs')
text2 = ax2.text(0.4, 0.85, 'Main Frequency: --', horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes, 
	bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

fig.tight_layout()
plt.pause(update_period)


for k in range(40*40):

	# Read in audio data
	last_audio_chunk = frombuffer(stream.read(buffer_size, exception_on_overflow=False), int16)
	line1.set_ydata(last_audio_chunk)

	# Transform raw audio data
	temp1 = ifftshift((last_audio_chunk-mean(last_audio_chunk))/std(last_audio_chunk))
	temp2 = fft(list(last_audio_chunk)+list(zeros(buffer_size-1)))
	temp3 = real(ifft(abs(temp2)))
	autocorr = temp3[:buffer_size//2]/(arange(buffer_size//2)[::-1]+buffer_size//2)	
	autocorr = autocorr/max(autocorr)	
	line2.set_ydata(autocorr)

	# If there is more than just noise, analyze signal
	if mean(abs(last_audio_chunk))>(2**noise_db_threshold)*noise_level:
		# Determine primary frequency.
		first_maximum_peak = argmax(autocorr[50:])+50
		main_freq = round(sampling_rate/first_maximum_peak)
		point2.set_xdata(first_maximum_peak)
		point2.set_ydata(autocorr[first_maximum_peak])

		# Determine note.
		curr_note = all_notes['note'].values[argmin(abs(all_notes['freq'].values-main_freq))]
		text2.set_text(f'Main Frequency: {main_freq}Hz ({curr_note})')

	else:
		point2.set_xdata(-100)
		point2.set_ydata(0)
		text2.set_text(f'Main Frequency: --')

	plt.pause(update_period)

