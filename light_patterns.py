from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
import time


# ---------------------------- #
# LIGHT PATTERN 1: LINE BURSTS #
# ---------------------------- #
def StreamingLightPattern(state, color1=[255,0,0], color2=[0,0,255]):
	num_steps_lead = 20
	num_steps_decay = 30
	num_steps_start_next = 6
	num_states_on_full = 1
	num_states_off_lag = 5
	lead_curve_curve_power = 1
	follow_curve_power = .5
	window_decay_rate = .97

	# Compute vectors
	lead_vector = [max(min(int(x),255),0) for x in 255*(linspace(0,1,num_steps_lead)**lead_curve_curve_power)]
	decay_vector = [255-max(min(int(x),255),0) for x in 255*(linspace(0,1,num_steps_decay)**follow_curve_power)]
	total_vector = lead_vector+num_states_on_full*[255]+decay_vector+num_states_off_lag*[0]

	# Generate LED 
	chunk_length1 = 101//2+3
	LED_chunk1 = [total_vector[max(state-i*num_steps_start_next,0)%len(total_vector)] for i in range(chunk_length1)]
	LED_chunk1 = [LED_chunk1[i]*(window_decay_rate)**i for i in range(len(LED_chunk1))]
	chunk_length2 = 116//2+3
	LED_chunk2 = [total_vector[max(state-i*num_steps_start_next,0)%len(total_vector)] for i in range(chunk_length2)]
	LED_chunk2 = [LED_chunk2[i]*(window_decay_rate)**i for i in range(len(LED_chunk2))]

	LED_strip_values = zeros(num_lights)

	# Do first origin point in both directions
	left_strip = (116-chunk_length2)*[0]+LED_chunk2[::-1]+(num_lights-116)*[0]
	right_strip = 116*[0]+LED_chunk1+(num_lights-116-chunk_length1)*[0]
	LED_strip_values += array(left_strip)+array(right_strip)

	# Do second origin point in both directions
	left_strip = (217-chunk_length1)*[0]+LED_chunk1[::-1]
	right_strip = LED_chunk2+(num_lights-chunk_length2)*[0]
	LED_strip_values += array(left_strip)+array(right_strip)

	return LED_strip_values




# Set up makeshift light grid
num_lights = 217
light_x_locations = list(linspace(0,-1,56))+60*[-1]+list(linspace(-1,-.5,26))+list(-1*linspace(.5**2,0,46)**(1/2))+29*[0]
light_y_locations = 56*[0]+list(linspace(0,1,60))+26*[1]+list(linspace(1,.5,46))+list(linspace(.5,0,29))
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1,1,1)
line1 = ax.scatter(light_x_locations, light_y_locations,s=num_lights*[0],c=arange(num_lights))
fig.tight_layout()
# plt.pause(.00001)


state = 0
for k in range(400):

	speed = 1
	state += speed

	spread_range = 25
	spread_rate = 1.08
	decay_rate = .80
	initial_height = 500
	avg_num_drops_on_sametime = 10

	num_states_before_off = int(log(.49/initial_height)/log(decay_rate))
	num_states_before_nexton = int(floor(num_states_before_off/avg_num_drops_on_sametime))
	LED_strip_values = zeros(num_lights)

	random.seed(1)
	light_drop_locs = random.randint(0,num_lights,1000)
	light_start_times = random.randint(num_states_before_nexton//2,2*num_states_before_nexton,1000)
	light_start_times[0] = 0
	light_start_times = cumsum(light_start_times)

	start_index = min(arange(1000)[(light_start_times+num_states_before_off)>=state])
	end_index = max(arange(1000)[light_start_times<=state])
	for k in range(start_index, end_index+1):
		sigma = spread_rate**(state-light_start_times[k])
		decay_factor = decay_rate**(state-light_start_times[k])
		curr_location = light_drop_locs[k]
		current_wave = [min(int(initial_height*decay_factor*exp(-.5*(abs(x)/sigma)**2)),255) for x in range(-spread_range, spread_range+1)]
		LED_strip_values[[x%num_lights for x in range(curr_location,(curr_location+1+2*spread_range))]] += current_wave


	# LED_strip_values = StreamingLightPattern(state)
	line1.set_sizes(LED_strip_values)
	line1.set_array(array(LED_strip_values))
	plt.pause(1/30)




