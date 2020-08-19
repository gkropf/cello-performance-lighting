from numpy import *
import pyaudio
import matplotlib.pyplot as plt
import time
from scipy import signal
import matplotlib as mpl
mpl.style.use('ggplot')

# Set program parameters
sampling_rate = 44100
window_length = 1/16

update_period = 1/1000
raw_signal_range = 6000
fft_signal_range = 20000
potentail_freq_range = [0,2000]
method = ['windowed_fft','autocorr'][1]

# Define base frequencies (frequencies for all notes in first octave, with A4==440 reference)
base_freq = {
	'C':32.70,
	'C#':34.65,
	'D':36.71,
	'D#':38.89,
	'E':41.20,
	'F':43.65,
	'F#':46.25,
	'G':49.00,
	'G#':51.91,
	'A':55.00,
	'A#':58.27,
	'B':61.74}
octave_range = int(ceil(log(potentail_freq_range[1]/base_freq['C'])))

# Set up audio stream
buffer_size = int(window_length*sampling_rate)
audio = pyaudio.PyAudio()
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
x2 = linspace(sampling_rate/buffer_size,sampling_rate/2,int(floor(buffer_size/2))+1)
y1 = random.randn(buffer_size)
y2 = random.randn(int(floor(buffer_size/2))+1)

ax1 = fig.add_subplot(1,2,1)
ax1.set_ylim([-raw_signal_range,raw_signal_range])
ax1.set_title('Audio Signal (Time Domain)')
ax1.set_yticks([])
line1, = ax1.plot(x1,y1)
ax1.set_xticklabels([str(int(x))+'ms' for x in ax1.get_xticks()])

ax2 = fig.add_subplot(1,2,2)
ax2.set_ylim([0,fft_signal_range])
ax2.set_title('Audio Signal (Freq. Domain) - Raw FFT')
line2, = ax2.plot(x2,y2)
ax2.set_xlim(potentail_freq_range)
ax2.set_xticklabels([str(int(x))+'Hz' for x in ax2.get_xticks()])
ax2.set_yticklabels([str(int(x))+'dB' for x in ax2.get_yticks()])

fig.tight_layout()
plt.pause(update_period)


for k in range(30*30):

	# Read in audio data
	last_audio_chunk = frombuffer(stream.read(buffer_size, exception_on_overflow=False), int16)

	# Transform raw audio data
	fft_signal = 10*log10(abs(fft.rfft(last_audio_chunk)))
	fft_clean_signal = signal.periodogram(last_audio_chunk,sampling_rate,return_onesided=True, window='hamming')[1]

	# Update plots
	line1.set_ydata(last_audio_chunk)
	line2.set_ydata(fft_clean_signal)

	plt.pause(update_period)



