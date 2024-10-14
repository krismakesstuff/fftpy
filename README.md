# Real-Time Audio Spectrogram with faster_whisper Transcription

A python app that gives you a real-time spectrogram of a mono audio feed. Record a snippet with spacebar and it will use faster_whisper to transcribe speech-to-text. 

## Requirements

- Python 3.x
- Required Python packages:
  - numpy
  - sounddevice
  - scipy
  - matplotlib
  - soundfile
  - faster_whisper
  - atexit

## User Inputs

- **Spacebar**: Toggle recording on and off. Transcribes recorded file. 