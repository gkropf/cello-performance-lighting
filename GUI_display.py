from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.fftpack import helper
mpl.style.use('ggplot')


# Set gui display parameters
spectrum_num_bands = 14
raw_signal_range = 8000
fft_signal_range = 1
volume_range = 80
potentail_freq_range = [30,1200]


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
		curr_note, best_wave, best_fit, past_volumes, past_beats, program_state, user_params = input_queue.get()

		# Call get method until we arrive at the latest item (use exception to prevent locking)
		try:
			while input_queue.qsize()>0:
				freq_audio_chunk, signal_fft, signal_autocorrelation, is_signal, first_max_peak, main_freq, \
				curr_note, best_wave, best_fit, past_volumes, past_beats, program_state, user_params = input_queue.get()
		except Empty:
			pass

		# Set user sampling parameters
		sampling_rate, visual_refresh_rate, noise_db_threshold, freq_window_length,\
		volume_window_length, running__beat_analysis, running_record_length, num_max_width = user_params


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
	print('GUI display finished')

