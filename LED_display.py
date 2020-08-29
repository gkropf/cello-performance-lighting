import neopixel
import board

# ---------------------------------------------------- #
# This function produces simple plots using matplotlib
# based on the currently processed audio data.
# ---------------------------------------------------- #
def LED_display_thread(input_queue):

	# Init
	program_state = 1
	pixels = neopixel.NeoPixel(board.D18, 5, pixel_order=neopixel.GRBW)
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

		if is_signal:
			pixels[:] = 5*[[0,255,255]]

		else:
			pixels[:] = 5*[[0,0,0]]

