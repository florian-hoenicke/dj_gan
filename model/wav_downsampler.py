from scipy.io.wavfile import read
from scipy.io.wavfile import write

current_sample_rate, y = read('data/experiment/dubstep-part.wav')
target_sample_rate = 8000
sample_step_size = int(current_sample_rate / target_sample_rate)
y = y[::sample_step_size, 0]
write('data/experiment/dubstep-part-8000.wav', target_sample_rate, y)