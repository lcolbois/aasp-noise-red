'''
Example of how to use single_noise_channel removal algorithm. In this example we also use what we have seen in other examples:
We're gonna synthetize a signal, Then we're gonna do processing on it using the algorithm and finally we are going to label
the newly obtained file and compare them to the file without any processing.
'''


import numpy as np
from scipy.io import wavfile
import utils

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pyroomacoustics as pra
import matplotlib.pyplot as plt

# import tf and functions for labelling
import tensorflow as tf

# import denoising library
import speech as sp


from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

# here we recreate some function from tensorflow to be able to extract the results obtained via labelling and to plot them if we want to.

# load the graph we're gonna use for labelling
def  load_graph(f):
    with tf.gfile.FastGFile(f,'rb') as graph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph.read())
        tf.import_graph_def(graph_def, name='')

# load the labels we're gonna use with the graph
def load_labels(f):
    return [line.rstrip() for line in tf.gfile.GFile(f)]

# run the graph and label our file. We add the fact that this function returns the prediction such that we can work with it afterwards.
def run_graph(wav_data, labels, index, how_many_labels=3):
    with tf.Session() as session:
        softmax_tensor = session.graph.get_tensor_by_name("labels_softmax:0")
        predictions, = session.run(softmax_tensor,{"wav_data:0": wav_data})

    top_k = predictions.argsort()[-how_many_labels:][::-1]
    for node_id in top_k:
        human_string = labels[node_id]
        score = predictions[node_id]
        print('%s (score = %.5f)' % (human_string, score))
    return predictions[index]

# main function used for labelling. We add a retrun to this function to recover the results.
# this function labels wavfiles so you always need to create a wavfile of your sound to label it.

def label_wav(wav,labels,graph,word):

    if not wav or not tf.gfile.Exists(wav):
        tf.logging.fatal('Audio file does not exist %s',wav)
    if not labels or not tf.gfile.Exists(labels):
        tf.logging.fatal('Labels file does not exist %s', labels)
    if not graph or not tf.gfile.Exists(graph):
        tf.logging.fatal('Graph file does not exist %s', graph)

    labels_list = load_labels(labels)
    load_graph(graph)

    with open(wav,'rb') as wav_file:
        wav_data = wav_file.read()
    index = labels_list.index(word)
    return run_graph(wav_data,labels_list,index)

if __name__ == '__main__':

    '''
    User parameters for synthetizing the signal
    '''

    # the SNR values in dB we use to create the different samples
    snr_vals = np.arange(100,0,-5)
    # desired basis word(s) (can also be a list)
    desired_word = 'go'
    #choose your label file
    labels_file = "conv_labels.txt"
    #choose your graph file
    graph_file = "my_frozen_graph.pb"
    # destination directory to write your new samples
    dest_dir = 'output_final_single_noise_removal'
    if not os.path.exists(dest_dir):
    	os.makedirs(dest_dir)

    '''
    Parameters of the algortihm
    '''
    # the number of LPC coefficents to consider
    lpc_order = 10
    # the number of iterations to compute the Wiener filter
    iterations = 10
    # the lenght of our FFT
    fft_len = 400
    # number of bins we're gonna use for our FFT
    n_fft_bins = fft_len//2 + 1
    # parameter of the noise Update
    alpha = 0.0

    '''
    Selecting words from the dataset as in the example of the GoogleSpeechCommand
    '''
    # create the dataset object
    dataset = pra.datasets.GoogleSpeechCommands(download=True,subset=10)

    # separate the noise and the speech samples
    noise_samps = dataset.filter(speech=0)
    speech_samps = dataset.filter(speech=1)
    # filter the speech samples to take only the desired word(s)
    speech_samps = speech_samps.filter(word=desired_word)

    # pick one sample of each (from the noise samples and the speech samples filtered)
    speech = speech_samps[1]
    noise = noise_samps[4]

    # print the information of our chosen speech and noise file
    print("speech file info :")
    print(speech.meta)
    print("noise file info:")
    print(noise.meta)
    print()

    '''
    Create new samples using Pyroomacoustics as in example how_to_synthesize_a_signal.py
    '''

    # creating a noisy_signal array for each snr value
    speech_signal, sr = sp.data.load(speech.meta.as_dict()['file_loc'])
    noisy_signal = np.zeros((len(snr_vals),speech_signal.shape[0]))

    seed = 0
    for i,snr in enumerate(snr_vals):
        noisy_signal[i] = sp.data.add_noise_from_file(speech_signal,sr,noise.meta.as_dict()['file_loc'],snr)

    processed_signal = np.zeros(noisy_signal.shape)
    processed_signal_VAD = np.zeros(noisy_signal.shape)

    # we run the algorithm for each of our possible signal
    for i in range(len(snr_vals)):
        processed_signal[i] = sp.process.denoise(noisy_signal[i],fft_len,lpc_order,iterations)
        processed_signal_VAD[i] = sp.process.denoise_with_vad(noisy_signal[i],sr,fft_len,lpc_order,iterations,alpha)

    '''
    Write to WAV + labelling of our processed noisy signals
    '''
    # labelling our different single noise channel removed signals and comparing their classification with the one for the original noisy signals
    score_processing = np.zeros(len(snr_vals))
    score_processing_VAD = np.zeros(len(snr_vals))
    score_original = np.zeros(len(snr_vals))

    for i, snr in enumerate(snr_vals):
        print("SNR : %f dB" % snr)
        dest = os.path.join(dest_dir,"denoised_snr_db_%d.wav" %(snr))
        signal = pra.normalize(processed_signal[i], bits=16).astype(np.int16)
        wavfile.write(dest,16000,signal)
        score_processing[i] = label_wav(dest, labels_file, graph_file, speech.meta.as_dict()['word'])

        dest = os.path.join(dest_dir,"denoised_with_VAD_snr_db_%d.wav" %(snr))
        signal = pra.normalize(processed_signal_VAD[i], bits=16).astype(np.int16)
        wavfile.write(dest,16000,signal)
        score_processing_VAD[i] = label_wav(dest, labels_file, graph_file, speech.meta.as_dict()['word'])

        dest = os.path.join(dest_dir,"noisy_snr_db_%d.wav" %(snr))
        signal = pra.normalize(noisy_signal[i], bits=16).astype(np.int16)
        wavfile.write(dest,16000,signal)
        score_original[i] = label_wav(dest, labels_file, graph_file, speech.meta.as_dict()['word'])
        print()


    # plotting the result
    plt.plot(snr_vals,score_original, label="noisy")
    plt.plot(snr_vals,score_processing, label="denoised")
    plt.plot(snr_vals,score_processing_VAD,label="VAD + denoised")
    plt.legend()
    plt.title('SNR against percentage of confidence')
    plt.xlabel('SNR in dB')
    plt.ylabel('score')
    plt.grid()
    plt.show()
