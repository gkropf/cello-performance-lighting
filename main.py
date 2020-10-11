from multiprocessing import Process, Queue, Manager
from queue import Empty
from scipy.io.wavfile import write
from audio_processing import *


# Set up queues for managing tasks.
audio_stream_queue = Queue()
processed_audio_data_queue = Queue()
final_audio = Queue()
manager = Manager()
program_state = manager.Value('int',4)

# Ask user which display method they would like to use
method = ''
while method not in ['LED','GUI']:
	method = input('Which display method would you like to use (LED/GUI)?\n')

# Start processes based on which display method was chosen
process1 = Process(target=record_audio_thread, args=[audio_stream_queue,program_state])
process2 = Process(target=audio_processing_thread, args=[audio_stream_queue,processed_audio_data_queue,final_audio])
if method == 'GUI':
	from GUI_display import *
	process3 = Process(target=GUI_display_thread, args=[processed_audio_data_queue])
elif method == 'LED':
	from LED_display import *
	process3 = Process(target=LED_display_thread, args=[processed_audio_data_queue])

process1.start()
process2.start()
process3.start()

# Use user input to advance program state
while program_state.value<=4:
	input('Press any key to advance state')
	program_state.value += 1

# Save full recording
full_recording = final_audio.get()
time.sleep(.1)
# write("main_recording.wav",sampling_rate,array(full_recording, dtype=int16))
print('Saved recording.')


