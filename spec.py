import numpy as np
import sounddevice as sd
from scipy import signal
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button 
import matplotlib.pyplot as plt
import soundfile as sf
import os
from datetime import datetime

# Parameters
sample_rate = 44100  # Sample rate of the microphone
duration = 2  # Duration of each audio chunk in seconds
nfft = 2048  # Number of FFT points
buffer_duration = 0.5  # Duration of audio data to collect before sending to Whisper

# Initialize plot with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))  # Increase height to accommodate two plots

# Data for the first plot
x = np.linspace(0, duration, int(sample_rate * duration))
y = np.zeros_like(x)
line1, = ax1.plot(x, y)
ax1.set_title("Full Frequency Range")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Amplitude")

# Data for the second plot (zoomed-in version)
ax2CutoffFrequency = 2500  # Cutoff frequency for the second plot
line2, = ax2.plot(x, y)
ax2.set_xlim(0, duration)
ax2.set_ylim(0, ax2CutoffFrequency)  # Limit y-axis to frequencies lower than 10k Hz
ax2.set_title("Frequencies Lower than 10k Hz")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Amplitude")

# Init recording button
record_button_ax = plt.axes([0.81, 0.01, 0.1, 0.075])
record_button = Button(record_button_ax, 'Record', color='lightgoldenrodyellow', hovercolor='0.975')

# Init recording status and buffer
recording = False
recorded_audio = [];

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)


# Callback function to update the plot
def audio_callback(indata, frames, time, status):
    global y, recorded_audio, accumulated_audio
    
    y = np.roll(y, -frames)
    y[-frames:] = indata[:, 0]
    
    # Record audio if recording is enabled
    if recording:
        recorded_audio.extend(indata[:, 0])


# Update all plots
def update_plots(frame):
    global transcription_text, new_transcription_available, recording, transcription_status
    frequencies, times, spectrogram = signal.spectrogram(y, sample_rate, nperseg=nfft, scaling='spectrum')

    log_spectrum = np.log(spectrogram)
    norm_spectrum = (log_spectrum - np.min(log_spectrum)) / (np.max(log_spectrum) - np.min(log_spectrum))

    # clear the plots
    ax1.clear()
    ax2.clear()

    # plot the full frequency range
    ax1.pcolormesh(times, frequencies, norm_spectrum)
    ax1.set_ylabel('Frequency [Hz]')
    #ax1.set_xlabel('Time [sec]')
    ax1.set_title('Full Frequency Range')

    # plot the frequencies lower than 10k Hz
    ax2.pcolormesh(times, frequencies, norm_spectrum)
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time [sec]')
    ax2.set_xlim(0, duration)
    ax2.set_ylim(0, ax2CutoffFrequency)  # Limit y-axis to frequencies lower than 10k Hz
    ax2.set_title('Frequencies Lower than ' + str(ax2CutoffFrequency) +  'Hz')
    
    # Display recording status
    if recording:
        ax1.text(0.5, 1.2, "Recording...", transform=ax1.transAxes, ha="center", va="top", fontsize=12, color="red")
        ax1.text(0.5, 1.15, "Press spacebar to stop recording or click the Record button", transform=ax1.transAxes, ha="center", va="top", fontsize=10, color="red")
    else:   
        ax1.text(0.5, 1.2, "Press spacebar to start recording or click the Record button", transform=ax1.transAxes, ha="center", va="top", fontsize=12, color="green")

    return ax1, ax2
    


# Function to start/stop recording
def toggle_recording(event=None):
    global recording, recorded_audio, transcribe, fileToTranscribe, transcription_status
    if recording:
        # Stop recording and save to file
        recording = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("output", f"recording_{timestamp}.wav")
        sf.write(filename, np.array(recorded_audio), sample_rate)
        print(f"Recording saved to {filename}")
        
        # clear recorded audio buffer
        print("Clearing recorded audio buffer")
        recorded_audio = []

    else:
        # Start recording
        recording = True
        print("Recording started")

# Bind spacebar keypress to toggle recording
def bind_spacebar_keypress():
    def on_key(event):
        if event.key == ' ':
            toggle_recording()
    fig.canvas.mpl_connect('key_press_event', on_key)


def main():
    # connect record button to toggle recording
    record_button.on_clicked(toggle_recording)
    # bind spacebar keypress to toggle recording
    bind_spacebar_keypress()


    # Start the microphone stream
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate)
    with stream:
        ani = FuncAnimation(fig, update_plots, interval=25)
        plt.show()

if __name__ == "__main__":
    main()