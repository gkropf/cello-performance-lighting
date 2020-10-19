# Cello Performance Lighting
Since beginning cello lessons, I've been amazed by both the beauty of the instrument and the patience required for daily monotonous practice sessions. This project serves to spice things up when it comes to playing basic scales by having an LED display that responds to the notes and tempo of what I play. Each note on the cello corresponds to a distinct color palette, and each string determines a different animation pattern. This repository contains two methods for visualizing audio: one for the GUI which can be used on any desktop, and one for controlling LEDs that requires a Raspberry Pi with GIPO pins.

## Using Application
To run this application, simply download the repository, extract the contents, and start the program by running `python main.py` in a terminal (a list of required python packages is listed below). You will then be asked which display method (GUI or LED) to use. Note that selecting the LED option on any device other than a Raspberry Pi will cause the application to crash. 

The `main.py` script works by first importing`audio_processing.py` and then starting a continuous loop that keeps a running queue of audio chunks that have been analyzed by doing the following.

1. The audio signal's autocorrelation is computed and the first maximum in this function is used to determine the current frequency.
2. The closest note corresponding to the current frequency is selected from a predefined dictionary along with the corresponding colors of that note.
3. Square, saw, and sine waves of the given frequency are compared against the raw signal and the best-fitting wave is chosen. This will primarily be used for later editions when selecting two-note chords will require wave simulations. For now, the fitted wave's autocorrelation is simply plotted over the raw signal's autocorrelation.
4. Using a history of volumes, the occurrence of new beats is flagged (marked by dramatic increases in volume and additional criteria about the separation of beats).

All of this information is then inserted into an outbound queue that is to be used to update a visual display using either the GUI or LED display.

Required python packages: (for GUI) matplotlib; (for LED) neopixel, board; (common) multiprocessing, queue, scipy, pandas, pyaudio, numpy, time.

### GUI Display

The GUI display shows four different views of the incoming audio signal. The raw signal is shown in the top left, and the signals Fourier transform (which gives the signals spectral content) is shown on the top right. In the bottom left we show the autocorrelation of the signal along with the current frequency and note. The note and frequency are only updated when an actual signal above the noise threshold is present. Finally, on the bottom right we show a history of the signal's relative volume (plotted on the logarithmic dB scale). The occurrence of new beats are marked by red squares. 

![alt text](https://github.com/gkropf/cello-performance-lighting/blob/master/ReadmeFiles/GUI_example.gif "")

### LED Display

The LED display uses the current notes corresponding colors to create streaming light patterns. Every new beat increases the speed of the animation with a constant decay factor applied to the speed and brightness such that fade-in/fade-out effects occur in between notes. There are other light patterns written, but these haven't been incorporated into the main script yet as I'm working on smoothing out animation jumps between patterns.

![alt text](https://github.com/gkropf/cello-performance-lighting/blob/master/ReadmeFiles/LED_example.gif "")


## Hardware Setup
For this project, I used WS2812 individually addressable LEDs (often called NeoPixels due to the companies pioneering of the original strips). There is no shortage of great NeoPixel tutorials, I personally recommended starting here for a barebones walkthrough of getting started: https://learn.adafruit.com/neopixels-on-raspberry-pi/raspberry-pi-wiring.

I used logical level converter to convert the 3.3V signal coming from the Raspberry Pi 4 to the 5V signal required for the LEDs. Then I connected a 5V 40A power supply to multiple locations in my LED strips to prevent voltage drop and variable light intensity (make sure to avoid ground loops when power from multiple locations). Since I used a symmetric platform, I split the signal into both sides of the platform. This cut down the number of LEDs I needed to update (since the signal is duplicated across) sides and cut the lag of the display in half. 

Finally, I used a generic USB microphone plugged directly into the Pi which pyaudio has default support for.

![alt text](ReadmeFiles/LEDSchematic.png "")
