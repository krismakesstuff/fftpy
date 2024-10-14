import numpy as np
import sounddevice as sd
from scipy import signal
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button 
import matplotlib.pyplot as plt
import threading
import soundfile as sf
import os
from datetime import datetime
import queue
from faster_whisper import WhisperModel
#import atexit

# Parameters
sample_rate = 44100  # Sample rate of the microphone
duration = 2  # Duration of each audio chunk in seconds
nfft = 2048  # Number of FFT points
buffer_duration = 0.5  # Duration of audio data to collect before sending to Whisper

# Initialize Faster Whisper model
# model = WhisperModel("tiny.en", device="auto", compute_type="float16")
# Initialize plot with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))  # Increase height to accommodate two plots

# Data for the first plot
x = np.linspace(0, duration, int(sample_rate * duration))
y = np.zeros_like(x)
line1, = ax1.plot(x, y)
ax1.set_title("Full Frequency Range")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Amplitude")

# Data for the second plot (zoomed-in version)
ax2CutoffFrequency = 2000  # Cutoff frequency for the second plot
line2, = ax2.plot(x, y)
ax2.set_xlim(0, duration)
ax2.set_ylim(0, ax2CutoffFrequency)  # Limit y-axis to frequencies lower than 10k Hz
ax2.set_title("Frequencies Lower than 10k Hz")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Amplitude")

# Init recording button
record_button_ax = plt.axes([0.81, 0.01, 0.1, 0.075])
record_button = Button(record_button_ax, 'Record', color='lightgoldenrodyellow', hovercolor='0.975')

# Initialize plot
# fig, ax = plt.subplots()
# x = np.linspace(0, duration, int(sample_rate * duration))
# y = np.zeros_like(x)
# line, = ax.plot(x, y)

transcription_text = "No transcription"
transcription_status = ""
audio_buffer = queue.Queue()
new_transcription_available = False
recording = False
recorded_audio = []
accumulated_audio = []  # Buffer to accumulate audio data

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Declare global variables
transcribe = False
fileToTranscribe = ""
timer_thread = None

# # Function to transcribe audio from file
# def transcribe_audio_from_file(filename):
#     global transcription_text, new_transcription_available, transcription_status
#     transcription_status = "Transcribing recording..."
#     print(f"Reading file: {filename}")
#     # audio_data, sample_rate = sf.read(filename)
#     # print(f"File read successfully: {filename}")
#     # print(f"Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}")

#     segments, info = model.transcribe(filename, beam_size=3)

#     # for segment in segments:
#     #     print(segment.text)
#     #     transcription_text = segment.text
#     #     new_transcription_available = True
#     #     transcription_status = "Transcription complete"
#     #     print("Transcription: ", transcription_text)

#     transcription_text = " ".join([segment.text for segment in segments])
#     new_transcription_available = True
#     transcription_status = "Transcription complete"
#     print("Transcription: ", transcription_text)

# def transcribeIfNeeded():
#     global transcribe, fileToTranscribe
#     if transcribe:
#         print("Transcribing..." + fileToTranscribe)
#         transcribe_audio_from_file(fileToTranscribe)
#         transcribe = False  # Reset the flag after transcription

# Timer callback function
def timer_callback():
    #transcribeIfNeeded()
    # Restart the timer
    global timer_thread
    timer_thread = threading.Timer(1.0, timer_callback)
    timer_thread.start()  # Call every 1 second

# Callback function to update the plot
def audio_callback(indata, frames, time, status):
    # if status:
    #     print(status)
    global y, recorded_audio, accumulated_audio
    y = np.roll(y, -frames)
    y[-frames:] = indata[:, 0]
    
    # Record audio if recording is enabled
    if recording:
        recorded_audio.extend(indata[:, 0])
    
    # temp = indata[:, 0]

    # ythreshold = 0.25

    # for i in range(len(temp)):
    #     if temp[i] < ythreshold:
    #         temp[i] = 0

    # y[-frames:] = temp

    # Add audio chunk to buffer
    # accumulated_audio.extend(indata[:, 0])
    
    # Check if accumulated audio is long enough
    # if len(accumulated_audio) >= int(buffer_duration * sample_rate):
    #     audio_buffer.put(np.array(accumulated_audio))
    #     accumulated_audio = []  # Reset the buffer


# Function to update the plot
# def update_plot(frame):
#     global transcription_text, new_transcription_available, recording, transcription_status
#     frequencies, times, spectrogram = signal.spectrogram(y, sample_rate, nperseg=nfft)
#     ax.clear()
#     ax.pcolormesh(times, frequencies, np.log(spectrogram))
#     ax.set_ylabel('Frequency [Hz]')
#     ax.set_xlabel('Time [sec]')
    
#     # Display recording status
#     if recording:
#         ax.text(0.5, 1.2, "Recording...", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="red")
#         ax.text(0.5, 1.15, "Press spacebar to stop recording", transform=ax.transAxes, ha="center", va="top", fontsize=10, color="red")
#     else:
#         ax.text(0.5, 1.2, "Press spacebar to start recording", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="green")
    
#     # Display transcription status
#     ax.text(0.5, 1.1, transcription_status, transform=ax.transAxes, ha="center", va="top", fontsize=12, color="blue")
    
#     # Display transcription text
#     ax.text(1.05, 0.5, transcription_text, transform=ax.transAxes, ha="left", va="center", fontsize=12, color="black", wrap=True)
    
#     return ax

# Update all plots
def update_plots(frame):
    global transcription_text, new_transcription_available, recording, transcription_status
    frequencies, times, spectrogram = signal.spectrogram(y, sample_rate, nperseg=nfft)

    # clear the plots
    ax1.clear()
    ax2.clear()

    # plot the full frequency range
    ax1.pcolormesh(times, frequencies, np.log(spectrogram))
    ax1.set_ylabel('Frequency [Hz]')
    #ax1.set_xlabel('Time [sec]')
    ax1.set_title('Full Frequency Range')

    # plot the frequencies lower than 10k Hz
    ax2.pcolormesh(times, frequencies, np.log(spectrogram))
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

    # Display transcription status
    # ax1.text(0.5, 1.1, transcription_status, transform=ax1.transAxes, ha="center", va="top", fontsize=12, color="blue", bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=10))

    # Display transcription text
    # ax1.text(1.05, 0.5, transcription_text, transform=ax1.transAxes, ha="left", va="center", fontsize=12, color="black", wrap=True)

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
        
        # Set transcription flag and file
        # transcribe = True
        # fileToTranscribe = filename
        # transcription_status = "Transcribing recording..."
    else:
        # Start recording
        # transcription_status = ""
        recording = True
        print("Recording started")

# Create a button to toggle recording
# def create_record_button():
#     record_button_ax = plt.axes([0.81, 0.01, 0.1, 0.075])
#     record_button = Button(record_button_ax, 'Record', color='lightgoldenrodyellow', hovercolor='0.975')
#     record_button.on_clicked(toggle_recording)

# Bind spacebar keypress to toggle recording
def bind_spacebar_keypress():
    def on_key(event):
        if event.key == ' ':
            toggle_recording()
    fig.canvas.mpl_connect('key_press_event', on_key)

# Cleanup function to stop threads
def cleanup():
    global timer_thread
    if timer_thread is not None:
        timer_thread.cancel()
    print("Cleanup complete. Exiting...")

def main():
    #create_record_button()

    record_button.on_clicked(toggle_recording)
    bind_spacebar_keypress()

    # transcription not working on windows. Commenting out for now
    # Start the timers
    # timer_callback()

    # Not working on windows. Commenting out for now
    # Register cleanup function to be called on exit
    # atexit.register(cleanup)

    # Start the microphone stream
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate)
    with stream:
        # ani = FuncAnimation(fig, update_plot, interval=100)
        ani = FuncAnimation(fig, update_plots, interval=25)
        plt.show()

if __name__ == "__main__":
    main()