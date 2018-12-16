# aasp-noise-red

***We develop a small module of speech enhancement based on Wiener filtering. The Wiener filter is build from estimates of the speech and noise power spectral densities (PSD). The speech PSD is estimated by LPC analysis, which allows to compute the parameters of an all-pole model of the vocal tract, while a voice detection algorithm (VAD) allows to identify speechless frames on which to estimate the noise PSD. We observe that our algorithm does increases the SNR and is convincing from an hearing perspective. It does however tend to reduce the speech intelligibility, which is tested through a pretrained neural network for spoken work classification.***


## Prerequisites
- numpy 1.13.3
- pyroomacoustics 0.1.20
- tensorflow 1.12.0

### Report in Notebook format : see src/Report.ipynb


### Demo
Go to src/ and launch "python demo.py"  
This produces 4 files :  
- clean_speech.wav : original version of a clean speech signal
- noisy_speech.wav : noisy version of the speech signal
- denoising.wav : denoised version of the noisy speech that does not make use of Voice Activity Detection
- VAD+denoising.wav : denoised version of the noisy using both Voice Activity Detection and Wiener filtering
  
The algorithm parameters can easily be tweaked in the code for demo.py

### Evaluate  
Go to src/ and launch "python evaluate_single_sample.py" or "python evaluate_several_samples.py". This demonstrates the process of feeding noisy and denoised speech samples to a pretrained neural network for spoken word classification, as a measure of speech intelligibility.

### "Speech" package  
Go to src/speech/  
The package is composed of   
- data.py -> Wavefile reading and writing, spliting in frames
- process.py -> Main module that contains all the necessary computations and denoising algorithms
- vad.py -> Voice Activity Detection module, mainly a VAD class with a method able to give a decision given a signal frame
- evaluate.py -> MSE computation between clean and recovered speech signal, computation of the a posteriori SNR
