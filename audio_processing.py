import pandas as pd
import pyaudio
from numpy import *
from scipy.fftpack import fft, ifft, ifftshift, helper
from scipy import signal
import time


# Set audio processing parameters
sampling_rate = 44100
visual_refresh_rate = 30#Hz, determines audio buffer size
noise_db_threshold = 20#dB, sets threshold for volume to trigger audio analysis
freq_window_length = 1/32#s, length of window used to compute current frequency
volume_window_length = 1/32#s, length of window used to compute current volume
running__beat_analysis = 2#s, must be less than running record length
running_record_length = 5#s, length of running window of audio to store
num_max_width = 1


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
		last_audio_chunk = array(frombuffer(stream.read(buffer_size, exception_on_overflow=False), int16), float)
		output_queue.put([last_audio_chunk,program_state.value])

	# This makes sure that the next thread gets properly shutdown
	output_queue.put([[],program_state.value])
	print('Audio recording finished')


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
	notes_current = pd.DataFrame([
		('C1',  32.70, ([0,0,255], [123,23,86])),
		('C1#', 34.65, ([0,255,0], [0,0,255])),
		('D1',  36.71, ([255,0,0], [0,255,0])),
		('D1#', 38.89, ([255,55,5], [255,0,0])),
		('E1',  41.20, ([55,55,255], [255,55,55])),
		('F1',  43.65, ([20,20,255], [55,55,255])),
		('F1#', 46.25, ([123,37,90], [20,20,255])),
		('G1',  49.00, ([21,123,120], [123,37,90])),
		('G1#', 51.91, ([223,12,0], [21,123,120])),
		('A1',  55.00, ([213,9,180], [223,12,0])),
		('A1#', 58.27, ([123,10,12], [213,9,180])),
		('B1',  61.74, ([123,23,86], [123,10,12]))])

	notes_current.columns = ['note','freq', 'color_pair']
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
	noise_level = 25


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
		curr_note = all_notes.iloc[argmin(abs(all_notes['freq'].values-main_freq)),:]

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
				if (min_between_beats-noise_level)<.6*(min_beats-noise_level): 
					last_beat = k
					last_beat_volume = all_recorded_volumes[k]
					volume_is_beat.append(True)
				else:
					volume_is_beat.append(False)
			else:
				volume_is_beat.append(False)
	

		# Output processed audio features for display
		user_params = [sampling_rate, visual_refresh_rate, noise_db_threshold, freq_window_length,\
			volume_window_length, running__beat_analysis, running_record_length, num_max_width]

		output_queue.put([freq_audio_chunk, signal_fft, signal_autocorrelation, is_signal, first_max_peak, \
			main_freq, curr_note, best_wave, best_fit, 
			all_recorded_volumes[-int(running_record_length/volume_window_length):],
			volume_is_beat[-(int(running_record_length/volume_window_length)-num_max_width):]+num_max_width*[False], 
			program_state, user_params])

	# Clean up and save recording
	output_queue.put(11*[[]]+[program_state,user_params])
	save_queue.put(all_recorded_audio)
	print('Audio anlaysis finished')
