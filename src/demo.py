import speech as sp

'''
Parameters of the algortihm
'''
# VAD
with_VAD = False
# the number of LPC coefficents to consider
lpc_order = 10
# the number of iterations to compute the Wiener filter
iterations = 5
# the lenght of our FFT
frame_size = 400
# parameter update of the sigma in sigma tracking
alpha = 0.05

# Preparing the sample
y_clean, sr = sp.data.load('../samples/speech_male/arctic_a0001.wav')

# For now : choose the SNR (Db) and add corresponding noise to the signal
SNR = 10
y = sp.data.add_noise_from_file(y_clean,sr,'../samples/noise/crowd.wav',SNR)
s = sp.process.denoise(y,frame_size,lpc_order,iterations)
s_VAD,_,_ = sp.process.denoise_with_vad(y,sr,frame_size,lpc_order,iterations,alpha)

sp.data.write(y_clean,sr,'clean_speech.wav')
sp.data.write(y,sr,'noisy_speech.wav')
sp.data.write(s,sr,'denoising.wav')
sp.data.write(s_VAD,sr,'VAD+denoising.wav')
