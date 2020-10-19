# Cello Performance Lighting
Since beginning cello lessons I've been amazed by both the beauty of the instrument and the patience required for daily monotonous practice sessions. This project serves to spice things up when it comes to playing basic scales by having an LED display that responds to the notes and tempo of what I play. Each note on the cello corresponds to a distinct color palette, and each string determines a different animation pattern. This repository contains two methods for visualizing audio: one for the GUI which can be used on any desktop, and for the controlling LEDs that requires a Raspberry Pi with GIPO pins.

## Using Application
To run this application, simply download the repository, extractthe contents, and start the program by running `python main.py` in a terminal (a list of required python packages is listed below). You will then be asked which display method (GUI or LED) to use; note that selecting the LED option on any device other than a Raspberry Pi will cause the application to crash. 

The `main.py` script works by first importing`audio_processing.py` and then starting a continuous loop that keeps a running queue of audio chunks that have been analyzed by doing the following.

1. The audio signals autocorrelation is computed and the firstmaximum in this function is used to determine the current frequency.
2. The closest note corresponding to the current frequency isselected from a predefined dictionary along with the correspondingcolors of that note.
3. Square, saw, and sine waves of the given frequency are comparedagainst the raw signal and the best fitting wave is chosen. This willprimarily be used for later editions when selecting two-note chordswill require wave simulations. For now, the fitted wavesautocorrelation is simply plotted over the raw signal’s.
4. Using a history of volumes the occurrence of new beats isflagged (marked by dramatic increases in volume and additionalcriteria about separation of beats).

All of this information is then inserted into an outbound queue that is to be used to update a visual display using either the GUI or LED display.

Required python packages.
For GUI display: matplotlib.
For LED display: neopixel, board.
Common: multiprocessing, queue,scipy, pandas, pyaudio, numpy, time.

### GUI Display
![alt text](https://github.com/gkropf/cello-performance-lighting/blob/master/ReadmeFiles/GUI_example.gif "")

### LED Display
![alt text](https://github.com/gkropf/cello-performance-lighting/blob/master/ReadmeFiles/LED_example.gif "")


## Hardware Setup
![alt text](ReadmeFiles/LEDSchematic.png "")
