import scipy
import librosa
from scipy import io
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import numpy as np
from wave import Wave_write

# y, s = librosa.load('data/wav/test.mp3', sr=8000) # Downsample 44.1kHz to 8kHz
s, y = read('data/generated/18/0.wav')
# Wave_write
# print(s)
# y_new = []
# rate = 500
# for i in range(0, len(y), rate):
#     a = y[i]
#     y_new += [a] * rate
# write('data/wav/down.wav', s, np.array(y_new))


# samplerate = 44100; fs = 100
# t = np.linspace(0., 1., samplerate)
# amplitude = np.iinfo(np.int16).max
# data = amplitude * np.sin(2. * np.pi * fs * t)
# write("data/wav/example.wav", samplerate, data)

print('')


