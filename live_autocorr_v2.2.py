from numpy import *
import pyaudio
import time
from scipy import signal
from scipy.fftpack import fft, ifft, ifftshift, helper
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
# from threading import Thread
# from queue import Queue
from multiprocessing import Process, Queue
import sys
from scipy.io.wavfile import write

set_printoptions(threshold=sys.maxsize)
start_time = time.time()

# Set audio processing parameters
sampling_rate = 44100
visual_refresh_rate = 60#Hz, determines audio buffer size
noise_db_threshold = 10#dB, sets threshold for volume to trigger audio analysis
freq_window_length = 1/32#s, length of running window used to compute current frequency
volume_window_length = 1/32#s, length of running window used to compute current volume
running_record_length = 5#s, length of running window of audio to store


# Set gui display parameters
spectrum_num_bands = 14
raw_signal_range = 8000
fft_signal_range = 1
potentail_freq_range = [30,2000]
running_window_length = 2

run_time = 1000


# --------------------------------------------- #
# This functions only purpose is to maintain
# a live in-time queue of audio chunks
# --------------------------------------------- #
def record_audio_thread(output_queue):
	audio = pyaudio.PyAudio()
	buffer_size = int(sampling_rate/visual_refresh_rate)
	stream = audio.open(
		format=pyaudio.paInt16, 
		channels=1, 
		rate=sampling_rate, 
		input=True,
		frames_per_buffer=buffer_size
		)
	while time.time()-start_time<run_time:
		last_audio_chunk = array(frombuffer(stream.read(buffer_size, exception_on_overflow=True), int16), float)
		audio_stream_queue.put(last_audio_chunk)


# --------------------------------------------- #
# This function holds the core audio processing
# code and continuously updates the shared data.
# --------------------------------------------- #
def audio_processing_thread(input_queue,output_queue,save_queue):

	# Define fast autocorrelation function
	def autocorr(series):
		temp2 = fft(list(series)+list(zeros(len(series)-1)))
		temp3 = real(ifft(abs(temp2)))
		autocorr = temp3[:len(series)//2]/(arange(len(series)//2)[::-1]+len(series)//2)	
		return autocorr/max(autocorr)


	# Define all note frequencies
	notes_current = pd.DataFrame([('C1',32.70),('C1#',34.65),('D1',36.71),('D1#',38.89),('E1',41.20),('F1',43.65),
		('F1#',46.25),('G1',49.00),('G1#',51.91),('A1',55.00),('A1#',58.27),('B1',61.74)])
	notes_current.columns = ['note','freq']
	all_notes = notes_current.copy()
	for k in range(2,7):
		notes_current['note'] = [x[:1]+str(k)+x[2:] for x in notes_current['note'].values]
		notes_current['freq'] = [2*x for x in notes_current['freq'].values]
		all_notes = all_notes.append(notes_current)

	# Get noise magnitude of room
	# print('\nGauging ambient noise level of the room, please be quiet.')
	# noise_sample = array([],float)
	# for k in range(int(1*sampling_rate/window_size)):
	# 	noise_sample = concatenate((noise_sample,array(frombuffer(stream.read(window_size, exception_on_overflow=False), int16), float)))
	# noise_level = 10*log10(mean(noise_sample**2))
	# print(f'Recorded noise: {noise_level}dB')
	noise_level = 52



	all_recorded_audio = []
	sample_volumes = []
	loop_time = 60*[10]
	loop_time2 = 60*[10]
	chunk_counter = 

	while time.time()-start_time<run_time:

		# Update running audio variables
		latest_audio_chunk = input_queue.get()
		ctime = time.time()
		while not input_queue.empty():
			latest_audio_chunk = input_queue.get()
		window_size = len(latest_audio_chunk)
		all_recorded_audio += list(latest_audio_chunk)
		all_recorded_audio = all_recorded_audio[-int(3*sampling_rate):]
		freq_audio_chunk = all_recorded_audio[-int(freq_window_length*sampling_rate):]
		volume_audio_chunk = all_recorded_audio[-int(volume_window_length*sampling_rate):]
		signal_autocorrelation = autocorr(freq_audio_chunk)
		signal_fft = abs(fft(hanning(len(freq_audio_chunk))*freq_audio_chunk))**2

		ctime2 = time.time()
		# Check if there is an actual signal present (based simply on volume)
		curr_freq_volume = 10*log10(mean(array(freq_audio_chunk)**2))
		is_signal = (curr_freq_volume > noise_level + noise_db_threshold)

		# Determine primary frequency and note using peak method
		first_max_peak = argmax(signal_autocorrelation[50:])+50
		main_freq = round(sampling_rate/first_max_peak)		
		curr_note = all_notes['note'].values[argmin(abs(all_notes['freq'].values-main_freq))]

		# Find best fitting theoretical wave
		signal_x_times = arange(len(freq_audio_chunk))/sampling_rate
		wave_sine = autocorr(sin(main_freq*signal_x_times*2*pi))
		wave_square = autocorr(signal.square(main_freq*signal_x_times*2*pi))
		wave_sawtooth = autocorr(signal.sawtooth(main_freq*signal_x_times*2*pi))
		best_wave, best_fit = max([(x,corrcoef(x,signal_autocorrelation)[0,1]) for x in [wave_sine,wave_square,wave_sawtooth]], key=lambda t: t[1])

		# Output processed audio features for display
		output_queue.put([freq_audio_chunk, volume_audio_chunk, signal_fft, signal_autocorrelation, \
			is_signal, first_max_peak, main_freq, curr_note, best_wave, best_fit])

		loop_time = loop_time[1:]+[ctime2-ctime]
		loop_time2 = loop_time2[1:]+[time.time()-ctime2]
		# print(mean(loop_time),mean(loop_time2))
		# if curr_freq_volume>60:
		# 	print('------ blip ------')

	# Save the entire recording
	save_queue.put(all_recorded_audio)
	save_queue.put(signal_fft)

# ---------------------------------------------------- #
# This function produces simple plots using matplotlib
# based on the currently processed audio data.
# ---------------------------------------------------- #
def GUI_display_thread(input_queue):

	# Set up the base plots
	fig = plt.figure(figsize=(28,20), dpi=100)

	ax1 = fig.add_subplot(2,2,1)
	x = 1000*arange(int(freq_window_length*sampling_rate))/sampling_rate
	line1, = ax1.plot(x,0*x,'k-')
	current_ticks = ax1.get_xticks()
	print(current_ticks)
	ax1.set_xticklabels([str(int(x))+'ms' for x in current_ticks])
	ax1.set_ylim([-raw_signal_range,raw_signal_range])
	#ax1.set_xlim([min(x),max(x)])
	ax1.set_title('Raw Audio Signal')
	ax1.set_yticks([])

	ax2 = fig.add_subplot(2,2,3)
	x = arange(len(x)//2)
	point1, = ax2.plot(-5,5,'rs')
	line2, = ax2.plot(x,0*x,'k--')
	line3, = ax2.plot(x,0*x,'b-')
	ax2.set_ylim([-fft_signal_range,fft_signal_range])
	ax2.set_title('Signal Autocorrelation')
	ax2.set_xticklabels([str(int(1000*x/sampling_rate))+'ms' for x in ax2.get_xticks()])
	#ax2.set_xlim([min(x),max(x)])
	text1 = ax2.text(0.4, 0.85, 'Main Frequency: --\nMax Degree of fit: ', horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes, 
	bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))	

	ax3 = fig.add_subplot(2,2,2)
	x = arange(spectrum_num_bands)
	y = arange(len(x))
	line4 = ax3.bar(x,height=y)
	ax3.set_ylim([0,2*10**11])

	# Prep for band updates
	fft_signal_freqs = sampling_rate*helper.fftfreq(int(freq_window_length*sampling_rate))
	band_indices = [argmin(abs(fft_signal_freqs-x)) for x in linspace(potentail_freq_range[0],potentail_freq_range[1],spectrum_num_bands+1)]
	fig.tight_layout()
	# plt.plot()
	plt.pause(.00001)
	fig_counter = 0


	while time.time()-start_time<run_time:
		# Read processed audio from queue
		freq_audio_chunk, volume_audio_chunk, signal_fft, signal_autocorrelation, \
		is_signal, first_max_peak, main_freq, curr_note, best_wave, best_fit = input_queue.get()
		while not input_queue.empty():
			freq_audio_chunk, volume_audio_chunk, signal_fft, signal_autocorrelation, \
			is_signal, first_max_peak, main_freq, curr_note, best_wave, best_fit = input_queue.get()

		fig_counter += 1
		print(f'------ fake fig_counter {fig_counter}')
		# if 10*log10(mean(array(freq_audio_chunk)**2))>60:
		# 	print('------ blip ------')


		# Update raw audio plot
		line1.set_ydata(freq_audio_chunk)

		# Update spectrum plot
		# fft_bands = [mean(signal_fft[band_indices[k]:band_indices[k+1]]) for k in range(spectrum_num_bands)]
		# for rect, h in zip(line4,fft_bands):
		# 	rect.set_height(h)

		# Update freq content
		line2.set_ydata(signal_autocorrelation)
		if is_signal:
			point1.set_xdata(first_max_peak)
			point1.set_ydata(signal_autocorrelation[first_max_peak])
			# text1.set_text(f'Main Frequencyz: {main_freq}Hz ({curr_note})\nMax Degree of fit: {int(100*best_fit)}%')
			line3.set_ydata(best_wave)
		else:
			# text1.set_text(f'Main Frequencyz: --Hz (--)\nMax Degree of fit: --%')
			line3.set_ydata(len(best_wave)*[-3])
			point1.set_ydata([-3])

		text1.set_text(f'Im on figure {fig_counter}')
		plt.pause(.000001)
		# plt.show()

audio_stream_queue = Queue()
processed_audio_data_queue = Queue()
final_audio = Queue()

process1 = Process(target=record_audio_thread, args=[audio_stream_queue])
process2 = Process(target=audio_processing_thread, args=[audio_stream_queue,processed_audio_data_queue,final_audio])
process3 = Process(target=GUI_display_thread, args=[processed_audio_data_queue])
process1.start()
process2.start()
process3.start()


# process3.start()
# write("temp.wav",sampling_rate,array(final_audio.get(), dtype=int16))



D = final_audio.get()

