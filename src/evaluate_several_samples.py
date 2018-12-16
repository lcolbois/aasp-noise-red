'''
example showing the improvement of single_noise_channel_removal on the full GoogleSpeechCommand
Dataset.
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
    User parameter for synthetizing the signal
    '''
    # the SNR values in dB we use to create the different samples
    snr_vals = np.arange(50,0,-5)
    # desired basis words. Here we have all the possible words in our model
    desired_word = ['no']
    # subest desired per word
    sub = 10
    #choose your label file
    labels_file = "conv_labels.txt"
    #choose your graph file
    graph_file = "my_frozen_graph.pb"
    # destination directory to write your new samples
    dest_dir = 'output_final_single_noise_removal_full'
    if not os.path.exists(dest_dir):
    	os.makedirs(dest_dir)

    '''
    Parameters of the algortihm
    '''
    # the number of LPC coefficents to consider
    lpc_order = 10
    # the number of iterations to compute the Wiener filter
    iterations = 5
    # the lenght of our FFT
    fft_len = 400
    # number of bins we're gonna use for our FFT
    n_fft_bins = fft_len//2 + 1
    # parameter update of the sigma
    alpha = 0.05

    '''
    Creating the Dataset object with all the value for the desired words
    '''
    # create the dataset object
    dataset = pra.datasets.GoogleSpeechCommands(download=True,subset=sub)
    # separate the noise and the speech samples
    noise_samps = dataset.filter(speech=0)
    speech_samps = dataset.filter(speech=1)
    print(len(speech_samps))
    # filter the speech samples to take only the desired word(s)
    speech_samps = speech_samps.filter(word=desired_word)
    # we choose our noise sample
    noise = noise_samps[0]
    # file location of the noise
    noise_file_location = noise.meta.as_dict()['file_loc']
    # creation of the map containing all the speech file locations
    speech_file_location = {}
    for s in speech_samps:
    	speech_file_location[s] = s.meta.as_dict()['file_loc']

    '''
    Create new samples using Pyroomacoustics as in example how_to_synthesize_a_signal.py
    '''
    # creation of the noisy signal for each SNR values
    noisy_signal = {}
    for s in speech_samps:
        speech_signal, sr = sp.data.load(speech_file_location[s])
        noisy_signal[s] = np.zeros((len(snr_vals),speech_signal.shape[0]))
        for i, snr in enumerate(snr_vals):
            noisy_signal[s][i] = sp.data.add_noise_from_file(speech_signal,sr,noise_file_location,snr)

    '''
    Processing of our noisy signals
    '''
    print("Let's go !")
    # we run the algorithm for all of our signal for all possible SNR values.
    # we create the map that will contain our newly computed signals
    processed_signal = {}
    processed_signal_VAD = {}
    for s in speech_samps:
        processed_signal[s] = np.zeros(noisy_signal[s].shape)
        processed_signal_VAD[s] = np.zeros(noisy_signal[s].shape)

    for s in speech_samps:
    	print('processing ...')
    	for i, snr in enumerate(snr_vals):
            processed_signal[s][i] = sp.process.denoise(noisy_signal[s][i],fft_len,lpc_order,iterations)
            processed_signal_VAD[s][i],_,_ = sp.process.denoise_with_vad(noisy_signal[s][i],sr,fft_len,lpc_order,iterations,alpha)

    '''
    Write to WAV and labelling of the samples.
    '''
    # creation of the map that are going to contain the score of the labelling function for our samples.
    score_map_processing = {}
    score_map_processing_VAD = {}
    score_map_original = {}
    # we are setting the values in our maps to 0
    for w in desired_word:
        score_map_original[w] = np.zeros([sub,len(snr_vals)])
        score_map_processing[w] = np.zeros([sub,len(snr_vals)])
        score_map_processing_VAD[w] = np.zeros([sub,len(snr_vals)])

    # now we are gonna compute the labelling
    idx = 0
    for s in speech_samps:
    	for i,snr in enumerate(snr_vals):
            word = s.meta.as_dict()['word']
            # destination of the processed signal
            dest_pro = os.path.join(dest_dir,"processed_signal%d%s_snr_db_%d" %(idx,word,snr))
            # destination of the processed with VAD signal
            dest_pro_vad = os.path.join(dest_dir,"processed_signal_VAD%d%s_snr_db_%d" %(idx,word,snr))
            # destination of the original siganl
            dest_ori = os.path.join(dest_dir,"original_signal%d%s_snr_db_%d" %(idx,word,snr))
            # noisy processed signal
            noisy_pro = pra.normalize(processed_signal[s][i], bits=16).astype(np.int16)
            wavfile.write(dest_pro,16000,noisy_pro)
            # noisy VAD+processed signal
            noisy_pro_vad = pra.normalize(processed_signal_VAD[s][i],bits=16).astype(np.int16)
            wavfile.write(dest_pro_vad,16000,noisy_pro_vad)
            # noisy original signal
            noisy_ori = pra.normalize(noisy_signal[s][i], bits=16).astype(np.int16)
            wavfile.write(dest_ori,16000,noisy_ori)
            # update the score maps
            print("score for original signal: ")
            score_map_original[word][idx][i] = label_wav(dest_ori, labels_file, graph_file, word)
            print()
            print("score for denoised signal: ")
            score_map_processing[word][idx][i] = label_wav(dest_pro, labels_file, graph_file, word)
            print()
            print("score for VAD + denoised signal: ")
            score_map_processing_VAD[word][idx][i] = label_wav(dest_pro_vad, labels_file, graph_file, word)
            print()
            idx +=1
            if(idx == sub):
            	idx = 0

    # creation of score average map
    score_map_original_avg = {}
    score_map_processing_avg = {}
    score_map_processing_VAD_avg = {}
    for w in desired_word:
        score_map_original_avg[w] = np.average(score_map_original[w],axis=0)
        score_map_processing_avg[w] = np.average(score_map_processing[w], axis=0)
        score_map_processing_VAD_avg[w] = np.average(score_map_processing_VAD[w],axis=0)

    # plotting of the result
    for w in desired_word:
        plt.subplot(2,1,1)
        plt.plot(snr_vals, score_map_original_avg[w], label='original_signal_for_%s' %w)
        plt.plot(snr_vals, score_map_processing_avg[w], label='denoised_signal_for_%s' %w)
        plt.plot(snr_vals, score_map_processing_VAD_avg[w], label='VAD+denoised_signal_for_%s' %w)
        plt.xlabel('SNR values [dB]')
        plt.ylabel('%% confidence')
        plt.legend()
    plt.grid()
    plt.show()
