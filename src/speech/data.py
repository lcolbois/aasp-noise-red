import numpy as np
import scipy.signal as sig
from scipy.io import wavfile

def load(filename):
    """ Load a sound sample from the path given by filename
        Normalize it and return (signal, sampling rate)
    """
    sr, s = wavfile.read(filename)
    s = s/np.std(s)
    return s, sr

def add_white_noise(x,SNR):
    """ Takes signal x and add white Gaussian noise to it to reach
        target SNR (in dB)
    """
    Ps=np.sum(x**2)
    noise_variance = Ps*10**(-SNR/10)
    noise = np.sqrt(noise_variance)*np.random.randn(len(x))
    return x+noise

def add_noise_from_file(speech,sr_speech,noise_path,SNR):
    noise_raw,sr_noise = load(noise_path)
    noise_part = noise_raw[:int(np.ceil(len(speech)*sr_noise/sr_speech))] # Keep only what will remain after resampling
    noise_resampled = sig.resample(noise_part,int(np.ceil(len(noise_part)*sr_speech/sr_noise))) # Resample to speech sampling rate
    noise = noise_resampled[:len(speech)] # Remove possible useless samples due to rounding approximations

    Ps = np.sum(speech**2)
    Pn = np.sum(noise**2)
    target_Pn = Ps/(10**(SNR/10))
    noise_corrected = noise*np.sqrt(target_Pn/Pn)
    return speech + noise_corrected

def frame_split(x,frame_size, with_overlap = True):
    """ Takes signal x and split it in frames of size frame_size
        with overlap 50% and cosine analysis and synthesis windows

        Return : list_frames -> list of start and end indices of each frame_size
                 x_padded -> original signal x padded with zeros at the end for having an integer number of frames
                 w_a, w_s -> Analysis and synthesis windows
    """
    # Overlap : the choice of analysis and synthesis windows is as in the STFT course
    # Careful that it is only consistent for a 50% overlap
    if(with_overlap):
        overlap_ratio = 0.5
        overlap = int(overlap_ratio*frame_size)
        w_a = sig.windows.cosine(frame_size)
        w_s = sig.windows.cosine(frame_size)
    else:
        overlap = 0
        w_a = np.ones(frame_size)
        w_s = np.ones(frame_size)

    # Compute the number of frames and pad the input signal at the end with zeros
    # to work only with full frames
    nframes = int(np.ceil((len(x)-overlap)/(frame_size-overlap)))

    padded_size = int((frame_size-overlap)*nframes+frame_size)
    x_padded = np.concatenate((x,np.zeros(padded_size-len(x))))


    list_frames = [[frame*(frame_size-overlap),frame*(frame_size-overlap)+frame_size] for frame in range(nframes)]

    return list_frames, x_padded, w_a, w_s
