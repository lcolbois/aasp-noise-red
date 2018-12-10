import numpy as np
import speech.data as data

def mse(speech,recovered_speech):
    #The recovered speech my have been zero padded
    recovered_speech = recovered_speech[:len(speech)]
    return np.mean((speech-recovered_speech)**2)

def time_mse(speech,recovered_speech, frame_size):
    recovered_speech_truncated = recovered_speech[:len(speech)]

    list_frames,speech_padded,_,_ = data.frame_split(speech,frame_size,with_overlap=False)
    recovered_speech_padded = np.concatenate((recovered_speech_truncated,np.zeros(len(speech_padded)-len(recovered_speech_truncated))))

    n_frames = len(list_frames)
    time_mse = np.zeros(n_frames)

    for frame in range(n_frames):
        start,end = list_frames[frame]
        time_mse[frame] = np.mean((speech_padded[start:end]-recovered_speech_padded[start:end])**2)
    return time_mse
