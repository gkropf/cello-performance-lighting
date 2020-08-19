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


start_time = time.time()
# Set audio processing parameters
sampling_rate = 44100
window_length = 1/32
noise_db_threshold = 10

# Set gui display parameters
raw_signal_range = 6000
fft_signal_range = 1
potentail_freq_range = [30,2000]



# Define fast autocorrelation function
def autocorr(series):
	temp2 = fft(list(series)+list(zeros(len(series)-1)))
	temp3 = real(ifft(abs(temp2)))
	autocorr = temp3[:len(series)//2]/(arange(len(series)//2)[::-1]+len(series)//2)	
	return autocorr/max(autocorr)


# --------------------------------------------- #
# This function holds the core audio processing
# code and continuously updates the shared data.
# --------------------------------------------- #
def audio_processing_thread(output_queue):

	# Initialize variables
	first_max_peak = 0
	main_freq = 0
	curr_note = 0
	best_wave = 0
	best_fit = 0

	# Define all note frequencies
	notes_current = pd.DataFrame((('C1',32.70),('C1#',34.65),('D1',36.71),('D1#',38.89),('E1',41.20),('F1',43.65),('F1#',46.25),('G1',49.00),('G1#',51.91),('A1',55.00),('A1#',58.27),('B1',61.74)))
	notes_current.columns = ['note','freq']
	all_notes = notes_current.copy()
	for k in range(2,7):
		notes_current['note'] = [x[:1]+str(k)+x[2:] for x in notes_current['note'].values]
		notes_current['freq'] = [2*x for x in notes_current['freq'].values]
		all_notes = all_notes.append(notes_current)

	# Set up audio stream for tuning
	audio = pyaudio.PyAudio()
	window_size = int(window_length*sampling_rate)
	stream = audio.open(
		format=pyaudio.paInt16, 
		channels=1, 
		rate=sampling_rate, 
		input=True,
		frames_per_buffer=window_size
		)

	# Get noise magnitude 
	print('\nGauging ambient noise level of the room, please be quiet.')
	noise_sample = array([],float)
	for k in range(int(4*sampling_rate/window_size)):
		noise_sample = concatenate((noise_sample,array(frombuffer(stream.read(window_size, exception_on_overflow=False), int16), float)))
	noise_level = 10*log10(mean(noise_sample**2))
	print(f'Recorded noise: {noise_level}dB')

	# Start main loop to continually pull and analyze audio data
	running_signal_volume = []
	while True:
		# Read in audio data and transform
		last_audio_chunk = array(frombuffer(stream.read(window_size, exception_on_overflow=False), int16), float)
		chunk_volume = 10*log10(mean(last_audio_chunk**2))
		signal_autocorr = autocorr(last_audio_chunk)

		# Update running list of sound magnitudes
		curr_time = time.time()
		num_split = 2
		split_audio_chunks = [last_audio_chunk[i:i + int(floor(window_size/num_split))] for i in range(0, window_size, int(floor(window_size/num_split)))]
		running_signal_volume += [(curr_time-(i/num_split)*window_size/sampling_rate,10*log10(mean(x**2))) for x,i in zip(split_audio_chunks,arange(num_split-1,-1,-1))]
		running_signal_volume = [x for x in running_signal_volume if x[0]-curr_time>-2]

		# If there is more than just noise, analyze signal
		chunk_has_signal = chunk_volume-noise_db_threshold>noise_level
		if chunk_has_signal:

			# Determine primary frequency and note using peak method
			first_max_peak = argmax(signal_autocorr[50:])+50
			main_freq = round(sampling_rate/first_max_peak)		
			curr_note = all_notes['note'].values[argmin(abs(all_notes['freq'].values-main_freq))]

			# Find best fitting theoretical wave
			signal_x_times = 1000*linspace(0,window_length,window_size)
			wave_sine = autocorr(sin(main_freq*(signal_x_times/1000)*2*pi))
			wave_square = autocorr(signal.square(main_freq*(signal_x_times/1000)*2*pi))
			wave_sawtooth = autocorr(signal.sawtooth(main_freq*(signal_x_times/1000)*2*pi))
			best_wave, best_fit = max([(x,corrcoef(x,signal_autocorr)[0,1]) for x in [wave_sine,wave_square,wave_sawtooth]], key=lambda t: t[1])

		# Update queue with latest value
		if not output_queue.empty():
			output_queue.get()

		output_queue.put([last_audio_chunk,chunk_volume,signal_autocorr,running_signal_volume,chunk_has_signal,first_max_peak,main_freq,curr_note,best_wave,best_fit,time.time()])
		# print(time.time()-start_time)


# ---------------------------------------------------- #
# This function produces simple plots using matplotlib
# based on the currently processed audio data.
# ---------------------------------------------------- #
def GUI_display_thread(input_queue):

	window_size = int(window_length*sampling_rate)

	# Set up the base plots
	fig = plt.figure(figsize=(14,4), dpi=100)
	x1 = 1000*linspace(0,window_length,window_size)
	x2 = arange(window_size//2)
	y1 = random.randn(window_size)
	y2 = random.randn(window_size//2)

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
	ax3.set_yticklabels([str(int(x))+'dB' for x in ax3.get_yticks()])

	fig.tight_layout()
	plt.pause(.00001)

	# Now continuously update plots based on latest processed audio
	while True:

		# Get all needed variables from queue and raw signal graph
		last_audio_chunk,chunk_volume,signal_autocorr,running_signal_volume,chunk_has_signal,first_max_peak,main_freq,curr_note,best_wave,best_fit,curr_time= input_queue.get()

		line1.set_ydata(last_audio_chunk)
		line2.set_ydata(signal_autocorr)
		num_udpate = min(len(running_signal_volume),len(line4.get_xdata()))
		line4.set_xdata([x[0]-curr_time for x in running_signal_volume[-num_udpate:]]+(len(running_signal_volume)-num_udpate)*[0])
		line4.set_ydata([x[1] for x in running_signal_volume[-num_udpate:]]+(len(running_signal_volume)-num_udpate)*[0])

		# If there is more than just noise, update note
		if chunk_has_signal:
			point1.set_xdata(first_max_peak)
			point1.set_ydata(signal_autocorr[first_max_peak])
			test1.set_text(f'Main Frequency: {main_freq}Hz ({curr_note})\nMax Degree of fit: {int(100*best_fit)}%')
			line3.set_ydata(best_wave)
		else:
			line3.set_ydata(-1000*ones(len(y2)))
			point1.set_xdata(-100)
			point1.set_ydata(0)
			test1.set_text(f'Main Frequency: --\nMax Degree of fit: --')

		plt.pause(.00001)

main_queue = Queue()
process1 = Process(target=audio_processing_thread, args=[main_queue])
# process2 = Process(target=GUI_display_thread, args=[main_queue])
process1.start()
# process2.start()
GUI_display_thread(main_queue)
# audio_processing_thread(main_queue)



