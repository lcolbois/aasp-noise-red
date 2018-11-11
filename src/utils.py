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


def compute_parameters(s,p):
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

def process_frame(s_frame,p):
    """Analyses frame s to compute the coefficients of a p-th order all pole filter, and return the synthesized frame
    obtained by filtering a random gaussian signal with this filter"""
    frame_size = s_frame.shape[0]
    a,g = compute_parameters(s_frame,p)

    y_padded_frame = np.zeros(p+frame_size) #Pad y with p zeros at the beginning
    w = np.random.randn(frame_size) # Excitation : centered gaussian (valid for unvoiced speech)
    for n in range(frame_size):
        y_padded_frame[n+p] = a@np.flip(y_padded_frame[n:n+p],0)+g*w[n] # Synthetize signal using the filter + the excitation
    return y_padded_frame[p:frame_size+p]
