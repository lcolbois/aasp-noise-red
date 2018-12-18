import numpy as np
from scipy.io import wavfile
import scipy.signal.windows as windows
import scipy.signal as signal
from scipy.fftpack import fft, ifft
import math

# Inspired from https://github.com/LCAV/easy-dsp/blob/master/algorithms/vad.py
class VAD(object):

    """
    Class to apply Voice Activity Detection (VAD) algorithms.
    Methods
    --------
    decision(X)
        Detect voiced segment by estimating noise floor level.
    decision_energy(frame, thresh)
        Detect voiced segment based on energy of the signal.
    """

    def __init__(self, N, fs, init_frame=[], init_level=False, tol=0, N_fft=128, tau_up=10, tau_down=40e-3, T_up=3, T_down=1.2):

        """
        Constructor for VAD (Voice Activity Detector) class.
        Parameters
        -----------
        N : int
            Length of frame.
        fs : float or int
            Sampling frequency
        tau_up : float
            Time in seconds.
        tau_down : float
            Time in seconds.
        T_up : float
            Time in seconds.
        T_down : float
            Time in seconds.
        """

        self.tol = tol
        self.N = N
        self.N_fft = N_fft
        self.T = self.N/float(fs)
        self.tau_up = tau_up
        self.tau_down= tau_down
        self.T_up = T_up
        self.T_down = T_down
        self.fs = fs
        self.eta_up = 1.2
        self.eta_down = 40e-3
        self.eta_min = 2

        self.W = windows.boxcar(math.ceil(N_fft/2))

        if init_level == True :
            X = fft(init_frame, N_fft)
            X = X[math.ceil(N_fft/2):]
            L = np.sum((self.W*abs(X))**2)/len(X)
            self.L_min = (self.T/self.tau_down)*L
        else :
            self.L_min = 10

        self.lambda_init = 1
        self.V = False
        #self.L_min = init_levels(self, init_frame)
        #self.dft = DFT(N)

    def decision_noise_level(self, x):

        """
        Detect voiced segment by estimating noise floor level.
        Parameters
        -----------
        X : numpy array
            RFFT of one signal with length self.N/2+1
        """
        X = fft(x, self.N_fft)
        X = X[math.ceil(self.N_fft/2):]

        L = np.sqrt(np.sum(self.W*abs(X))**2/len(X))

        # estimate noise floor
        if L > self.L_min:
            L_min = (1-self.T/self.tau_up)*self.L_min + self.T/self.tau_up*L
        else:
            L_min = (1-self.T/self.tau_down)*self.L_min + self.T/self.tau_down*L
        # voice activity decision
        if L/L_min < self.T_down:
            V = False
        elif L/L_min > self.T_up:
            V = True
        else:
            V = self.V

        self.L_min = L_min
        self.V = V

        return V

    def decision_energy(self, frame, thresh):

        """
        Detect voiced segment based on energy of the signal.
        Parameters
        -----------
        frame : numpy array
            One signal of length self.N
        thresh : float
            Threshold for detecting voiced segment.
        """
        tol = 1e-14
        E = np.log10(np.linalg.norm(frame)+self.tol)
        if E <= thresh:
            return False
        else:
            return True

    def decision_statistical(self, x, thresh) :
        X = fft(x, self.N_fft)
        X = X[math.ceil(self.N_fft/2):]
        tol=1e-14

        K = X.shape[0]
        gamma = abs(X)
        xhi = gamma - 1

        LAMBDA = np.sum(gamma - np.log10(gamma+tol) - 1)/K

        """if LAMBDA > self.eta_min:
            self.eta_min = (1-self.T/self.eta_up)*self.eta_min + self.T/self.eta_up*LAMBDA
        else:
            self.eta_min = (1-self.T/self.eta_down)*self.eta_min + self.T/self.eta_down*LAMBDA"""

        if LAMBDA > self.eta_min:
            self.eta_min = (1-1/self.eta_up)*self.eta_min + 1/self.eta_up*LAMBDA
        else:
            self.eta_min = (1-1/self.eta_down)*self.eta_min + 1/self.eta_down*LAMBDA

        # voice activity decision
        if LAMBDA < self.eta_down:
            V = False
        elif LAMBDA > self.eta_up:
            V = True
        else:
            V = self.V

        return V
