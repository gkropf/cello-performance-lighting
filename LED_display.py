import neopixel
import board
from numpy import *
#import matplotlib.pyplot as plt

# ---------------------------------------------------- #
# This function produces simple plots using matplotlib
# based on the currently processed audio data.
# ---------------------------------------------------- #
def LED_display_thread(input_queue):

	# Initilize
	program_state = 1
	speed_adjuster_state = 0
	speed = 1
	state = 1
	num_lights = 217
	cello_string = 0
	signal_strength = 1
	last_note = 'A'
	LEDcolor1, LEDcolor2 = 2*[[255,255,255]]
	pattern_direction = 1

	# Setup for streaming light pattern
	num_steps_lead = 20
	num_steps_decay = 30
	num_steps_start_next = 6
	num_states_on_full = 1
	num_states_off_lag = 5
	lead_curve_curve_power = 1
	follow_curve_power = .5
	window_decay_rate = .97
	lead_vector = [max(min(int(x),255),0) for x in 255*(linspace(0,1,num_steps_lead)**lead_curve_curve_power)]
	decay_vector = [255-max(min(int(x),255),0) for x in 255*(linspace(0,1,num_steps_decay)**follow_curve_power)]
	total_vector = lead_vector+num_states_on_full*[255]+decay_vector+num_states_off_lag*[0]

	# Set up for rainfall light pattern
	random.seed(1)
	spread_range = 25
	spread_rate = 1.08
	decay_rate = .80
	initial_height = 500
	avg_num_drops_on_sametime = 10
	num_states_before_off = int(log(.49/initial_height)/log(decay_rate))
	num_states_before_nexton = int(floor(num_states_before_off/avg_num_drops_on_sametime))
	light_drop_locs = random.randint(0,num_lights,5000)
	light_start_times = random.randint(num_states_before_nexton//2,2*num_states_before_nexton,5000)
	light_start_times[0] = 0
	light_start_times = cumsum(light_start_times)

	# Temp stuff -----------------------------------------------------------------------------
	num_lights = 217
	pixels = neopixel.NeoPixel(board.D18, num_lights+1, pixel_order=neopixel.GRBW)
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

		# Adjust speed based on beat count and update state
		# Update: Due to large number of slurs in cello music, found that using the actual beat recognition didn't work well. Instead
		# we will simply look for note changes to indicate beats.
		speed_adjuster_state = (speed_adjuster_state+1)%8
		if (last_note!=curr_note['note'] and is_signal) and (speed<8):
			speed += 1
		elif (speed_adjuster_state == 0) and (speed>1):
			speed -= 1
		last_note = curr_note['note']

		# Update current string/color based on frequency (if a signal is present). If not signal, turn off lights
		if is_signal:
			cello_string = 0 if main_freq < 65.41 else (1 if main_freq < 98 else (2 if main_freq < 146.83 else 3))
			LEDcolor1 = curr_note['color_pair'][0]
			LEDcolor2 = curr_note['color_pair'][1]
			signal_strength = min(signal_strength+.2,1)
			pattern_direction = 1 if (cello_string in [0,1]) else -1
			print(curr_note)
		else:
			signal_strength *= .8

		state += speed*pattern_direction
		if state <= 200:
			state += 1000


		# Generate LED
		chunk_length1 = 101//2+3
		LED_chunk1 = [total_vector[max(state-i*num_steps_start_next,0)%len(total_vector)] for i in range(chunk_length1)]
		LED_chunk1 = [LED_chunk1[i]*(window_decay_rate)**i for i in range(len(LED_chunk1))]
		chunk_length2 = 116//2+3
		LED_chunk2 = [total_vector[max(state-i*num_steps_start_next,0)%len(total_vector)] for i in range(chunk_length2)]
		LED_chunk2 = [LED_chunk2[i]*(window_decay_rate)**i for i in range(len(LED_chunk2))]
		LED_intensity_values = zeros(num_lights)

		# Do first origin point in both directions
		left_strip = (116-chunk_length2)*[0]+LED_chunk2[::-1]+(num_lights-116)*[0]
		right_strip = 116*[0]+LED_chunk1+(num_lights-116-chunk_length1)*[0]
		LED_intensity_values += array(left_strip)+array(right_strip)

		# Do second origin point in both directions
		left_strip = (217-chunk_length1)*[0]+LED_chunk1[::-1]
		right_strip = LED_chunk2+(num_lights-chunk_length2)*[0]
		LED_intensity_values += array(left_strip)+array(right_strip)

		# Make actual LED list of color triplets
		LED_intensity_values = (signal_strength/255)*LED_intensity_values
		LED_actual_values = [ \
			[int(intensity*x) for x in LEDcolor1] \
			if k//len(total_vector)%2==0 else \
			[int(intensity*x) for x in LEDcolor1] \
		 	for k, intensity in enumerate(LED_intensity_values)]

		# Update LEDs
		pixels[:] = [[0,0,0]] + LED_actual_values


