import numpy as np
import speech as sp
def autocorrelation(s,k):
    """Compute the autocorrelation of signal s at lag k, used in both the LHS
    and the RHS of the linear system that must be solved when computing
    the filter parameters

    Inputs :
        s : signal to analyse (numpy array)
        k : required lag (int)
    """
    N = len(s)
    if(k < 0):
        raise ValueError("k should be positive")
    elif(k>=N):
        raise ValueError("k should be smaller than the signal length")
    else:
        return s[k:N]@s[0:N-k]


def lpc_analysis(s,p):
    """Compute the parameters a_k of the p-order all-pole filter by solving the linear system obtained by MAP with
    observed signal s. Following case 2 from the article

    Inputs :
        s : Observed signal (numpy array)
        p : Order of the all-pole filter (float)

    Output :
        a : Coefficients vector of the all-pole filter (numpy array)
        g : Gain factor of the excitation (periodic for voiced speech, gaussian noise for unvoiced speech) (float)

    """
    from scipy.linalg import solve, toeplitz
    N = len(s)

    # Efficient building of the linear system by using the fact that A is a symmetric toeplitz, and its diagonals, as well
    # as the RHS, just contain the autocorrelation of s for various intervals
    # 1. Compute autocorrelations
    R = np.zeros(p+1)
    for k in range(len(R)):
        R[k] = autocorrelation(s,k)
    # 2. Build LHS, RHS and solve
    A = toeplitz(R[0:p])
    b = R[1:p+1]
    a = solve(A,b)

    # 3. Compute the gain factor for the excitation. Pad s with p zeros at the beginning (assumption on the S_I)
    padded_s = np.concatenate((np.zeros(p),s))
    g2 = 0
    for n in range(0,N):
            g2 = g2 + (padded_s[n+p] - a@np.flip(padded_s[n:n+p],0))**2
    g = np.sqrt(1/N*g2)

    return a,g

from scipy import integrate
def squared_gain(a,noise_PSD, y):
    """ Estimate the squared gain of the all-pole filter assuming a noisy signal.

    Inputs:
        a : Coefficient vector of the all-pole filter (numpy array)
        noise_PSD : estimate of the noise power spectral density (float)
        y : Noisy speech signal (numpy array)
    """
    N = y.shape[0]
    def integrand(omega):
        k = np.arange(a.shape[0])
        return 1/np.abs(1-a@np.exp(-1j*k*omega))**2
    vec_integrand = np.vectorize(integrand)
    omega = np.linspace(-np.pi,np.pi,100)
    integral = integrate.trapz(vec_integrand(omega),omega)
    signal_power = np.sum(y**2)
    # The speech power is the total power minus the noise power
    g2 = (2*np.pi/N*signal_power - 2*np.pi*noise_PSD)/integral
    return g2

def speech_PSD(a,g2,size):
    """ Use the provided LPC coefficients to build an estimate
    of the corresponding speech PSD (power spectral density).
    Return its evaluation at (size) regularly spaced normalized
    pulsations (between 0 and pi)

    Inputs :
        a : Coefficients of the all-pole filter
        g2 : Square of the gain of the all-pole filter
        size : Requested size of the output array

    """
    omega = np.linspace(0,np.pi,size)
    def scalar_PSD(omega):
        k = np.arange(a.shape[0])
        return g2/(np.abs(1-a@np.exp(-1j*k*omega)))**2
    vec_PSD = np.vectorize(scalar_PSD)
    return vec_PSD(omega)

def wiener_filtering(signal_dft, speech_PSD, noise_PSD):
    """ Compute transfer function from speech_PSD and noise_PSD
    and filtered the signal in the frequency domain

    Inputs :
        signal_dft : signal to filter in frequency domain (numpy array)
        speech_PSD : estimate of the speech power spectral density (float or numpy array)
        noise_PSD : estimate of the noise power spectral density (float or numpy array)

    Outputs :
        filtered : filtered signal in frequency domain (numpy array)

    """
    transfer = speech_PSD/(speech_PSD + noise_PSD)
    filtered = transfer*signal_dft
    return filtered, transfer

def denoise_frame(x,p,noise_PSD,iterations):
    """Denoise a frame in the STFT domain by applying iterative Wiener filtering
    and approximating the PSD of the signal through the use of the all-pole model

    Inputs :
        x : signal frame to denoise (numpy array)
        p : order of the all-pole model of the vocal tract (int)
        noise_PSD : estimate of the noise power spectral density (float)
        iterations : number of iterations to perform (int)

    Outputs :
        s_i : estimate of the denoised speech frame (numpy array)
    """
    dft = np.fft.rfft(x) #
    # First signal iterant is the original (noisy) signal
    s_i = x
    for it in range(iterations):
        a,_ = lpc_analysis(s_i,p) # Compute the coefficients of the all-pole filter from the current iterant
        g2 = squared_gain(a,noise_PSD,x)
        filtered_dft,transfer = wiener_filtering(dft,speech_PSD(a,g2,len(dft)),noise_PSD)
        s_i = np.fft.irfft(filtered_dft)

    return s_i, transfer

def lowpass_filter(s,sr,fmax):
    S = np.fft.rfft(s)
    freq = np.linspace(0,sr,len(s)//2+1)
    H = (freq <= fmax).astype(float)
    return np.fft.irfft(H*S)

def frame_nrj(x):
    """ Compute the energy of frame x
    Inputs:
        x : frame to analyse
    Outputs :
        fullband_nrj : total energy in the frame
        subband_nrj : energy contained in the subband of relevant voice frequencies
    """
    NFFT = 128
    X = np.fft.fftshift(np.fft.fft(x,n=NFFT))
    X = X[int(NFFT/2):]
    fullband_nrj = np.sum(np.abs(X)**2/NFFT)
    subband_nrj = np.sum(np.abs(X[1:4])**2/NFFT)
    return fullband_nrj, subband_nrj

def denoise(y,frame_size,p,iterations):
    # Get the list of frames for the STFT filtering
    list_frames,y_padded,w_a,w_s = sp.data.frame_split(y,frame_size)
    n_frames = len(list_frames)

    wiener_filter = np.zeros(int(frame_size//2)+1)
    s = np.zeros(y_padded.shape)

    noise_PSD = np.var(w_a*y_padded[:frame_size])

    for frame in range(n_frames):
        idx = list_frames[frame]
        if(np.sum((w_a*y_padded[idx])**2) > frame_size*noise_PSD):
            denoised,wiener_filter = sp.process.denoise_frame(w_a*y_padded[idx],p,noise_PSD,iterations)
            s[idx] = s[idx] + w_s*denoised
        else:
            s[idx] = s[idx] + w_s*(np.fft.irfft(wiener_filter*np.fft.rfft(w_a*y_padded[idx])))

    s = s/np.std(s)
    return s[:len(y)]
