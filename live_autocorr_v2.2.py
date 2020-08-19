from numpy import *
import pyaudio
import matplotlib.pyplot as plt
import time
from scipy import signal
from scipy.fftpack import fft, ifft, ifftshift
import matplotlib as mpl
import pandas as pd
# from threading import Thread
# from queue import Queue
from multiprocessing import Process, Queue
import sys

start_time = time.time()

# Set audio processing parameters
sampling_rate = 44100
window_length = 1/4
noise_db_threshold = 10
window_subdivide = 2

# Set gui display parameters
raw_signal_range = 6000
fft_signal_range = 1
potentail_freq_range = [30,2000]
running_window_length = 2



# Define fast autocorrelation function
def autocorr(series):
	temp2 = fft(list(series)+list(zeros(len(series)-1)))
	temp3 = real(ifft(abs(temp2)))
	autocorr = temp3[:len(series)//2]/(arange(len(series)//2)[::-1]+len(series)//2)	
	return autocorr/max(autocorr)


# --------------------------------------------- #
# This functions only purpose is to maintain
# a live in-time queue of audio chunks
# --------------------------------------------- #
def record_audio_thread(output_queue):
	audio = pyaudio.PyAudio()
	window_size = window_subdivide*int(window_length*sampling_rate//window_subdivide)
	stream = audio.open(
		format=pyaudio.paInt16, 
		channels=1, 
		rate=sampling_rate, 
		input=True,
		frames_per_buffer=window_size
		)
	while time.time()-start_time<10:
		last_audio_chunk = array(frombuffer(stream.read(window_size, exception_on_overflow=True), int16), float)
		audio_stream_queue.put(last_audio_chunk)

# --------------------------------------------- #
# This function holds the core audio processing
# code and continuously updates the shared data.
# --------------------------------------------- #
def audio_processing_thread(input_queue,output_queue):
	all_recorded_audio = []
	sample_volumes = []
	while time.time()-start_time<10:
		# Update running audio file
		latest_audio_chunk = input_queue.get()
		window_size = len(latest_audio_chunk)
		all_recorded_audio += list(latest_audio_chunk)

		# Do all freq analysis



		# Split latest chunk into subdivisions for volume analysis
		split_chunks = [latest_audio_chunk[i:i + window_size//window_subdivide] for i in range(0, window_size, window_size//window_subdivide)]
		sample_volumes += [10*log10(mean(x**2)) for x in split_chunks]

		# Output processed audio features for display
		output_queue.put(sample_volumes)


	# Save the entire recording
	# processed_audio_data_queue.put(all_recorded_audio)


# ---------------------------------------------------- #
# This function produces simple plots using matplotlib
# based on the currently processed audio data.
# ---------------------------------------------------- #
def GUI_display_thread(input_queue):

	# Set up the base plots
	fig = plt.figure(figsize=(14,4), dpi=100)

	ax3 = fig.add_subplot(1,3,3)
	x3 = linspace(-running_window_length, 0, window_subdivide*int(running_window_length/window_length))
	y3 = arange(window_subdivide*int(running_window_length/window_length))
	line4, = ax3.plot(x3,y3,'k--')
	ax3.set_xlim([-running_window_length,0])
	ax3.set_ylim([0,100])
	ax3.set_yticklabels([str(int(x))+'dB' for x in ax3.get_yticks()])
	plt.pause(.000001)

	while True:
		sample_volumes = input_queue.get()
		if len(sample_volumes)>len(y3):
			line4.set_ydata(sample_volumes[:len(y3)])
			plt.pause(.000001)

audio_stream_queue = Queue()
processed_audio_data_queue = Queue()


process1 = Process(target=record_audio_thread, args=[audio_stream_queue])
process2 = Process(target=audio_processing_thread, args=[audio_stream_queue,processed_audio_data_queue])
process3 = Process(target=GUI_display_thread, args=[processed_audio_data_queue])
process1.start()
process2.start()
process3.start()


# D = processed_audio_data_queue.get()
# p = pyaudio.PyAudio()
# stream = p.open(
# 	format=pyaudio.paFloat32,
# 	channels=1,
# 	rate=sampling_rate,
# 	output=True,
# )
# stream.write(array(D))


