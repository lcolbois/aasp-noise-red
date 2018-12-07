import numpy as np
def autocorrelation(s,k):
    """Compute the autocorrelation of signal s at lag k, used in both the LHS
    and the RHS of the linear system that must be solved when computing
    the filter parameters"""
    N = len(s)
    if(k < 0):
        raise ValueError("k should be positive")
    elif(k>=N):
        raise ValueError("k should be smaller than the signal length")
    else:
        return s[k:N-1]@s[0:N-1-k]


def lpc_analysis(s,p):
    """Compute the parameters a_k of the p-order all-pole filter by solving the linear system obtained by MAP with
    observed signal s. Following case 2 from the article"""
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

def speech_PSD(a,g,omega):
    """ Use the provided LPC coefficients to build an estimate
        of the corresponding speech PSD (power spectral density).
        Return its evaluation at the normalized frequencies contained
        in omega
    """
    def scalar_PSD(omega):
        k = np.arange(a.shape[0])
        return g**2/(np.abs(1-a@np.exp(-1j*k*omega)))**2
    vec_PSD = np.vectorize(scalar_PSD)
    return vec_PSD(omega)

def wiener_filtering(signal_dft, speech_PSD, noise_PSD):
    """ Compute transfer function from speech_PSD and noise_PSD
    and filterd the signal in the frequency domain
    """
    transfer = speech_PSD/(speech_PSD + noise_PSD)
    return transfer*signal_dft

def denoise_frame(x,p,sr,iterations):
    """Denoise a frame in the STFT domain by applying iterative Wiener filtering
    and approximating the PSD of the signal through the use of the all-pole model
    """
    noise_PSD = 9e-4
    #For now : estimator of PSD of the noise. Should be implemented with more care to improve results
    #For gaussian noise : P_n = sigma_n^2

    dft = np.fft.rfft(x) #
    omega = np.linspace(0,np.pi,len(dft))

    # First signal iterant is the original (noisy) signal
    x_i = x
    for it in range(iterations):
        a,g = lpc_analysis(x_i,p) # Compute the coefficients of the all-pole filter from the current iterant
        filtered_dft = wiener_filtering(dft,speech_PSD(a,g,omega),noise_PSD)
        x_i = np.fft.irfft(filtered_dft)
    return x_i
