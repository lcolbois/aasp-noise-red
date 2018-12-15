import numpy as np
import speech.data as data

def mse(speech,recovered_speech):
    #The recovered speech my have been zero padded
    recovered_speech = recovered_speech[:len(speech)]
    return np.mean((speech-recovered_speech)**2)

def time_mse(speech,recovered_speech, frame_size):
    list_frames,speech_padded,_,_ = data.frame_split(speech,frame_size,with_overlap=False)
    recovered_speech_padded = np.concatenate((recovered_speech_truncated,np.zeros(len(speech_padded)-len(recovered_speech_truncated))))

    n_frames = len(list_frames)
    time_mse = np.zeros(n_frames)

    for frame in range(n_frames):
        idx = list_frames[frame]
        time_mse[frame] = np.mean((speech_padded[idx]-recovered_speech_padded[idx])**2)
    return time_mse

def SNR(noisy_speech,recovered_speech):
    noise = noisy_speech-recovered_speech
    return 10*np.log10(np.sum(recovered_speech**2)/np.sum(noise**2))

def time_SNR(noisy_speech,recovered_speech,frame_size):
    noise = noisy_speech-recovered_speech
    list_frames,recovered_speech_padded,_,_ = data.frame_split(recovered_speech,frame_size,with_overlap=False)
    noise_padded = np.concatenate((noise,np.zeros(len(recovered_speech_padded)-len(noise))))

    n_frames = len(list_frames)
    time_post_SNR = np.zeros(n_frames)

    for frame in range(n_frames):
        idx = list_frames[frame]
        time_post_SNR[frame] = 10*np.log10(np.sum(recovered_speech_padded[idx]**2)/np.sum(noise_padded[idx]**2))
    return time_post_SNR
