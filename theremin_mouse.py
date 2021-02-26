import sounddevice as sd 
import numpy as np
import queue
import pyautogui
import time

FS = 44100
# base waveform of length FS to step through with base_wave_ptr
x = np.arange(FS)
base_wave = np.sin(2*np.pi*x/FS)
base_wave_ptr = 0

# Create a queue, in which input freq and amplitude will go
q = queue.Queue(maxsize=1)
# Initialise freq and amp to 0 (if queue is empty, the last values are used)
f_prev, f_prev2 = 0, 0
amp_prev, amp_prev2 = 0, 0

# Get screen size, and set freq and amp limits
screen_w, screen_h = pyautogui.size()
MIN_FREQ, MAX_FREQ = 300, 900
MIN_AMP, MAX_AMP = 0.0, 0.4

def audio_callback(outdata, frames, time, status):
    """ Gets freq and amp information from the queue and creates 
    samples to play from the base waveform """
    global base_wave_ptr, base_wave, f_prev, amp_prev, f_prev2, amp_prev2

    try:
        # Get values from the queue
        f, a = q.get_nowait()
        freq = int(f*0.5 + f_prev*0.3 + f_prev2*0.2)
        amp = a*0.5 + amp_prev*0.3 + amp_prev2*0.2
        f_prev2 = f_prev
        amp_prev2 = amp_prev
        f_prev  = f
        amp_prev = a
    except queue.Empty:
        print("Empty")
        # If queue is empty, just play the last freq and amp values
        freq = f_prev
        amp = amp_prev
    
    # Step through the base waveform in step size of desired freq
    for i in range(frames):
        outdata[i] = amp * base_wave[base_wave_ptr]
        base_wave_ptr = (base_wave_ptr + freq) % FS


with sd.OutputStream(channels=1, callback=audio_callback, samplerate=FS):
    while True:
        # Get mouse x,y position and convert to freq and amplitude
        mouse_x, mouse_y = pyautogui.position()
        freq_in = int(mouse_x/screen_w * (MAX_FREQ-MIN_FREQ) + MIN_FREQ)
        amp_in = mouse_y/screen_h * (MAX_AMP-MIN_AMP)

        # Put freq and amplitude in the queue
        q.put([freq_in, amp_in])
