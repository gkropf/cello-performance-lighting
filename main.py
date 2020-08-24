from numpy import *
import pyaudio
import time
from scipy import signal
from scipy.fftpack import fft, ifft, ifftshift, helper
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.style.use('ggplot')

import pandas as pd
# from threading import Thread
# from queue import Queue
from multiprocessing import Process, Queue, Manager
from queue import Empty
import sys
from scipy.io.wavfile import write
import readline

set_printoptions(threshold=sys.maxsize)
start_time = time.time()

# Set audio processing parameters
sampling_rate = 44100
visual_refresh_rate = 30#Hz, determines audio buffer size
noise_db_threshold = 10#dB, sets threshold for volume to trigger audio analysis
freq_window_length = 1/32#s, length of window used to compute current frequency
volume_window_length = 1/64#s, length of window used to compute current volume
running__beat_analysis = 2#s, must be less than running record length
running_record_length = 5#s, length of running window of audio to store


num_max_width = 2

# Set gui display parameters
spectrum_num_bands = 14
raw_signal_range = 8000
fft_signal_range = 1
volume_range = 80
potentail_freq_range = [30,1200]


# --------------------------------------------- #
# This functions only purpose is to maintain
# a live in-time queue of audio chunks
# --------------------------------------------- #
def record_audio_thread(output_queue,program_state):
	audio = pyaudio.PyAudio()
	buffer_size = int(sampling_rate/visual_refresh_rate)
	stream = audio.open(
		format=pyaudio.paInt16, 
		channels=1, 
		rate=sampling_rate, 
		input=True,
		frames_per_buffer=buffer_size
		)
	while program_state.value<=4:
	# while time.time()-start_time<=.5:
		last_audio_chunk = array(frombuffer(stream.read(buffer_size, exception_on_overflow=True), int16), float)
		audio_stream_queue.put([last_audio_chunk,program_state.value])

	# This makes sure that the next thread gets properly shutdown
	audio_stream_queue.put([[],program_state.value])
	print('1 Finished')


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
	print('\nGauging ambient noise level of the room, please be quiet.')
	# noise_sample = array([],float)
	# for k in range(int(1*sampling_rate/window_size)):
	# 	noise_sample = concatenate((noise_sample,array(frombuffer(stream.read(window_size, exception_on_overflow=False), int16), float)))
	# noise_level = 10*log10(mean(noise_sample**2))
	# print(f'Recorded noise: {noise_level}dB')
	noise_level = 40


	all_recorded_audio = []
	all_recorded_volumes = []
	volume_is_beat = []
	last_beat = 0
	last_beat_volume = 100

	while True:

		# Update running audio variable
		latest_audio_chunk, program_state = input_queue.get()
		window_size = len(latest_audio_chunk)
		all_recorded_audio += list(latest_audio_chunk)
		if program_state>4:
			break

		# Check if there is an actual signal present (based simply on volume)
		freq_audio_chunk = all_recorded_audio[-int(freq_window_length*sampling_rate):]
		curr_freq_volume = 10*log10(mean(array(freq_audio_chunk)**2))
		is_signal = (curr_freq_volume > noise_level + noise_db_threshold)

		# Determine primary frequency and note using autocorr peak method
		signal_autocorrelation = autocorr(freq_audio_chunk)
		signal_fft = abs(fft(hanning(len(freq_audio_chunk))*freq_audio_chunk))**2
		first_max_peak = argmax(signal_autocorrelation[50:])+50
		main_freq = round(sampling_rate/first_max_peak)		
		curr_note = all_notes['note'].values[argmin(abs(all_notes['freq'].values-main_freq))]

		# Find best fitting theoretical wave
		signal_x_times = arange(len(freq_audio_chunk))/sampling_rate
		wave_sine = autocorr(sin(main_freq*signal_x_times*2*pi))
		wave_square = autocorr(signal.square(main_freq*signal_x_times*2*pi))
		wave_sawtooth = autocorr(signal.sawtooth(main_freq*signal_x_times*2*pi))
		best_wave, best_fit = max([(x,corrcoef(x,signal_autocorrelation)[0,1]) for x in [wave_sine,wave_square,wave_sawtooth]], key=lambda t: t[1])

		# Update volumes list
		volume_chunk_size = int(sampling_rate*volume_window_length)
		if len(all_recorded_audio) > (len(all_recorded_volumes)+1)*volume_chunk_size:
			need_vol_analy = all_recorded_audio[(len(all_recorded_volumes))*volume_chunk_size:]
			volume_chunks = array([need_vol_analy[i*volume_chunk_size:(i+1)*volume_chunk_size] for i in range(len(need_vol_analy)//volume_chunk_size)])
			all_recorded_volumes += [10*log10(mean(chunk**2)) for chunk in volume_chunks]

		# Check new additions for possible beats. In order to be a beat the volume chunk must be
		# 1. Above threshold for noise
		# 2. A local maximum
		# 3. There must be atleast a 20% fall in volume between two beats
		for k in range(len(volume_is_beat),len(all_recorded_volumes)-num_max_width):

			# Check value is worth investigating
			beat_local_max = all_recorded_volumes[k]>max(
				max(all_recorded_volumes[(k-num_max_width):k], default=1000),
				max(all_recorded_volumes[(k+1):(k+1+num_max_width)],default=1000),
				noise_db_threshold+noise_level)

			# If beat is local minimum and above noise threshold, then check that there has been a dip in noise
			# between this and last beat (prevents multy counting continuous notes).
			if beat_local_max:
				min_between_beats = min(all_recorded_volumes[last_beat:])
				min_beats = min(all_recorded_volumes[k],last_beat_volume)
				if (min_between_beats-noise_level)<.7*(min_beats-noise_level): 
					last_beat = k
					last_beat_volume = all_recorded_volumes[k]
					volume_is_beat.append(True)
				else:
					volume_is_beat.append(False)
			else:
				volume_is_beat.append(False)
	



		# Output processed audio features for display
		output_queue.put([freq_audio_chunk, signal_fft, signal_autocorrelation, is_signal, first_max_peak, \
			main_freq, curr_note, best_wave, best_fit, all_recorded_volumes[-int(running_record_length/volume_window_length):],
			volume_is_beat[-(int(running_record_length/volume_window_length)-num_max_width):]+num_max_width*[False], program_state])

	# Clean up and save recording
	output_queue.put(11*[[]]+[program_state])
	save_queue.put(all_recorded_audio)
	print('2 finished')

# ---------------------------------------------------- #
# This function produces simple plots using matplotlib
# based on the currently processed audio data.
# ---------------------------------------------------- #
def GUI_display_thread(input_queue):

	# Init
	program_state = 1
	fig = plt.figure(figsize=(20,10), dpi=100)

	while program_state<=4:

		# Read processed audio from queue
		freq_audio_chunk, signal_fft, signal_autocorrelation, is_signal, first_max_peak, main_freq, \
		curr_note, best_wave, best_fit, past_volumes, past_beats, program_state = input_queue.get()

		# Call get method until we arrive at the latest item (use exception to prevent locking)
		try:
			while input_queue.qsize()>0:
				freq_audio_chunk, signal_fft, signal_autocorrelation, is_signal, first_max_peak, main_freq, \
				curr_note, best_wave, best_fit, past_volumes, past_beats, program_state = input_queue.get()
		except Empty:
			pass

		if program_state>4:
			break

		# Update/make raw audio plot
		if program_state>=1:
			try:
				line1.set_ydata(freq_audio_chunk)
			except:
				ax1 = fig.add_subplot(2,2,1)
				x = 1000*arange(int(freq_window_length*sampling_rate))/sampling_rate
				line1, = ax1.plot(x,0*x,'k-')
				current_ticks = ax1.get_xticks()
				print(current_ticks)
				ax1.set_xlim([min(x),max(x)])
				ax1.set_xticks([int((max(x)-min(x))*i) for i in linspace(.05,.95,5)])
				ax1.set_xticklabels([str(int(x))+'ms' for x in ax1.get_xticks()])
				ax1.set_ylim([-raw_signal_range,raw_signal_range])
				ax1.set_title('Raw Audio Signal')
				ax1.set_yticklabels([])

		# Update/make spectrum plot
		if program_state>=2:
			try:
				fft_bands = [mean(signal_fft[band_indices[k]:band_indices[k+1]]) for k in range(spectrum_num_bands)]
				for rect, h in zip(line4,fft_bands):
					rect.set_height(h)
			except:
				ax3 = fig.add_subplot(2,2,2)
				ax3.set_yticklabels([])
				fft_signal_freqs = sampling_rate*helper.fftfreq(int(freq_window_length*sampling_rate))
				band_indices = [argmin(abs(fft_signal_freqs-x)) for x in linspace(potentail_freq_range[0],potentail_freq_range[1],spectrum_num_bands+1)]
				x = arange(spectrum_num_bands)
				y = arange(len(x))
				line4 = ax3.bar(x,height=y)
				ax3.set_ylim([0,2*10**11])
				ax3.set_xticks(linspace(0,spectrum_num_bands-1,6))
				ax3.set_xticklabels([str(int(k))+'Hz' for k in linspace(potentail_freq_range[0],potentail_freq_range[1],6)])
				ax3.set_title('Fourier Transform of Signal')


		# Update/make freq content
		if program_state>=3:
			try:
				line2.set_ydata(signal_autocorrelation)
				if is_signal:
					point1.set_xdata(first_max_peak)
					point1.set_ydata(signal_autocorrelation[first_max_peak])
					text1.set_text(f'Main Frequencyz: {main_freq}Hz ({curr_note})\nMax Degree of fit: {int(100*best_fit)}%')
					line3.set_ydata(best_wave)
				else:
					text1.set_text(f'Main Frequencyz: --Hz (--)\nMax Degree of fit: --%')
					line3.set_ydata(len(best_wave)*[-3])
					point1.set_ydata([-3])
			except:
				ax2 = fig.add_subplot(2,2,3)
				x = arange(int(freq_window_length*sampling_rate)//2)
				point1, = ax2.plot(-5,5,'rs')
				line2, = ax2.plot(x,0*x,'k--')
				line3, = ax2.plot(x,0*x,'b-')
				ax2.set_ylim([-fft_signal_range,fft_signal_range])
				ax2.set_title('Signal Autocorrelation')
				ax2.set_xticks([int((max(x)-min(x))*i) for i in linspace(.05,.95,5)])
				ax2.set_xticklabels([str(int(1000*x/sampling_rate))+'ms' for x in ax2.get_xticks()])
				ax2.set_xlim([min(x),max(x)])
				ax2.set_title('Signal Autocorrelation')
				text1 = ax2.text(0.4, 0.85, 'Main Frequency: --\nMax Degree of fit: ', horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes, 
				bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))	

		# Update/make volume plot
		if program_state>=4:
			try:
				past_volumes = (int(running_record_length/volume_window_length)-len(past_volumes))*[0]+past_volumes
				past_beats = (int(running_record_length/volume_window_length)-len(past_beats))*[False]+past_beats
				beat_plot_update = [past_volumes[i] if past_beats[i] else -1 for i in range(len(past_beats))]
				line5.set_ydata(past_volumes)
				line6.set_ydata(beat_plot_update)
			except:
				ax4 = fig.add_subplot(2,2,4)
				x = arange(int(running_record_length/volume_window_length))*volume_window_length
				line5, = ax4.plot(x,0*x,'k--')
				line6, = ax4.plot(x,0*x-1,'rs')
				ax4.set_ylim([0,volume_range])
				ax4.set_title('Signal Volume')
				ax4.set_yticklabels([])

		# Display all changes
		fig.tight_layout()
		plt.pause(.000001)
	print('4 finished')





audio_stream_queue = Queue()
processed_audio_data_queue = Queue()
final_audio = Queue()
manager = Manager()
program_state = manager.Value('int',1)

process1 = Process(target=record_audio_thread, args=[audio_stream_queue,program_state])
process2 = Process(target=audio_processing_thread, args=[audio_stream_queue,processed_audio_data_queue,final_audio])
process3 = Process(target=GUI_display_thread, args=[processed_audio_data_queue])
process1.start()
process2.start()
process3.start()

while program_state.value<=4:
	input('Press any key to advance state')
	program_state.value += 1

full_recording = final_audio.get()
time.sleep(.1)
write("main_recording.wav",sampling_rate,array(full_recording, dtype=int16))

