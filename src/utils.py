import os, tarfile, bz2, requests, gzip 
from scipy.io import wavfile
import numpy as np
import pyroomacoustics as pra

def download_uncompress_tar_bz2(url, path='.'):

    # open the stream
    r = requests.get(url, stream=True)

    tmp_file = 'temp_file.tar'  

    # Download and uncompress at the same time.
    chunk_size = 4 * 1024 * 1024  # wait for chunks of 4MB
    with open(tmp_file, 'wb') as file:
        decompress = bz2.BZ2Decompressor()
        for chunk in r.iter_content(chunk_size=chunk_size):
            file.write(decompress.decompress(chunk))

    # finally untar the file to final destination
    tf = tarfile.open(tmp_file)
    tf.extractall(path)

    # remove the temporary file
    os.unlink(tmp_file)


def download_uncompress_tar_gz(url, path='.', chunk_size=None):

    tmp_file = 'tmp.tar.gz'
    if chunk_size is None:
        chunk_size = 4 * 1024 * 1024

    # stream the data
    r = requests.get(url, stream=True)
    with open(tmp_file, 'wb') as f:
        content_length = int(r.headers['content-length'])
        count = 0
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            count += 1
            print("%d bytes out of %d downloaded" % 
                (count*chunk_size, content_length))
    r.close()

    # uncompress
    tar_file = 'tmp.tar'
    with open(tar_file, "wb") as f_u:
        with gzip.open(tmp_file, "rb") as f_c:
            f_u.write(f_c.read())

    # finally untar the file to final destination
    tf = tarfile.open(tar_file)

    if not os.path.exists(path):
        os.makedirs(path)
    tf.extractall(path)

    # remove the temporary file
    os.unlink(tmp_file)
    os.unlink(tar_file)



def modify_input_wav(wav,noise,room_dim,max_order,snr_vals,mic_pos):

    '''
    for mono
    '''

    fs_s, audio_anechoic = wavfile.read(wav)
    fs_n, noise_anechoic = wavfile.read(noise)
    
    #Create a room for the signal
    room_signal= pra.ShoeBox(
        room_dim,
        absorption = 0.2,
        fs = fs_s,
        max_order = max_order)

    #rCeate a room for the noise
    room_noise = pra.ShoeBox(
        room_dim,
        absorption=0.2,
        fs=fs_n,
        max_order = max_order)

    #source of the signal and of the noise in their respectiv boxes
    room_signal.add_source([2,3.1,2],signal=audio_anechoic)
    room_noise.add_source([4,2,1.5], signal=noise_anechoic)

    #we add a microphone at the same position in both of the boxes
    room_signal.add_microphone_array(
        pra.MicrophoneArray(
            mic_pos.T, 
            room_signal.fs)
        )
    room_noise.add_microphone_array(
        pra.MicrophoneArray(
            mic_pos.T, 
            room_noise.fs)
        )

    #simulate both rooms
    room_signal.simulate()
    room_noise.simulate()

    #take the mic_array.signals from each room
    audio_reverb = room_signal.mic_array.signals
    noise_reverb = room_noise.mic_array.signals

    #verify the size of the two arrays such that we can continue working on the signal
    if(len(noise_reverb[0]) < len(audio_reverb[0])):
        raise ValueError('the length of the noise signal is inferior to the one of the audio signal !!')

    #normalize the noise_reverb
    noise_reverb = noise_reverb[0,:len(audio_reverb[0])]
    noise_normalized = noise_reverb/np.linalg.norm(noise_reverb)

    #initialize the noiy_signal
    noisy_signal = {}

    #for each snr values create a noisy_signal for the labelling function
    for snr in snr_vals:
        noise_std = np.linalg.norm(audio_reverb[0])/(10**(snr/20.))
        final_noise = noise_normalized*noise_std
        noisy_signal[snr] = audio_reverb[0] + final_noise
    return noisy_signal

def modify_input_wav_multiple_mics(wav,noise,room_dim,max_order,snr_vals,mic_array,pos_source,pos_noise):

    fs_s, audio_anechoic = wavfile.read(wav)
    fs_n, noise_anechoic = wavfile.read(noise)

    #Create a room for the signal
    room_signal= pra.ShoeBox(
        room_dim,
        absorption = 0.2,
        fs = fs_s,
        max_order = max_order)

    #Create a room for the noise
    room_noise = pra.ShoeBox(
        room_dim,
        absorption=0.2,
        fs=fs_n,
        max_order = max_order)

    #source of the signal and of the noise in their respectiv boxes
    room_signal.add_source(pos_source,signal=audio_anechoic)
    room_noise.add_source(pos_noise, signal=noise_anechoic)

    #we had the microphones array in both room
    room_signal.add_microphone_array(pra.MicrophoneArray(mic_array.T,room_signal.fs))
    room_noise.add_microphone_array(pra.MicrophoneArray(mic_array.T,room_noise.fs))

    #simulate both rooms
    room_signal.simulate()
    room_noise.simulate()

    #take the mic_array.signals from each room
    audio_reverb = room_signal.mic_array.signals
    noise_reverb = room_noise.mic_array.signals

    shape = np.shape(audio_reverb)

    noise_normalized = np.zeros(shape)

    #for each microphones
    if(len(noise_reverb[0]) < len(audio_reverb[0])):
        raise ValueError('the length of the noise signal is inferior to the one of the audio signal !!')
    noise_reverb = noise_reverb[:,:len(audio_reverb[0])]

    norm_fact = np.linalg.norm(noise_reverb[0])
    noise_normalized = noise_reverb / norm_fact

    #initilialize the array of noisy_signal
    noisy_signal = np.zeros([len(snr_vals),shape[0],shape[1]])

    for i,snr in enumerate(snr_vals):
        noise_std = np.linalg.norm(audio_reverb[0])/(10**(snr/20.))
        for m in range(shape[0]):
            
            final_noise = noise_normalized[m]*noise_std
            noisy_signal[i][m] = pra.normalize(audio_reverb[m] + final_noise)

    return noisy_signal

def modify_input_wav_beamforming(wav,noise,room_dim,max_order,snr_vals,mic_array,pos_source,pos_noise,N):

    fs_s, audio_anechoic = wavfile.read(wav)
    fs_n, noise_anechoic = wavfile.read(noise)

    #Create a room for the signal
    room_signal= pra.ShoeBox(
        room_dim,
        absorption = 0.2,
        fs = fs_s,
        max_order = max_order)

    #Create a room for the noise
    room_noise = pra.ShoeBox(
        room_dim,
        absorption=0.2,
        fs=fs_n,
        max_order = max_order)

    #source of the signal and of the noise in their respectiv boxes
    room_signal.add_source(pos_source,signal=audio_anechoic)
    room_noise.add_source(pos_noise, signal=noise_anechoic)

    #add the microphone array
    mics_signal = pra.Beamformer(mic_array, room_signal.fs,N)
    mics_noisy = pra.Beamformer(mic_array, room_noise.fs,N)
    room_signal.add_microphone_array(mics_signal)
    room_noise.add_microphone_array(mics_noisy)

    #simulate both rooms
    room_signal.simulate()
    room_noise.simulate()

    #take the mic_array.signals from each room
    audio_reverb = room_signal.mic_array.signals
    noise_reverb = room_noise.mic_array.signals

    #design beamforming filters
    mics_signal.rake_delay_and_sum_weights(room_signal.sources[0][:1])
    mics_noisy.rake_delay_and_sum_weights(room_signal.sources[0][:1])

    output_signal = mics_signal.process()
    output_noise = mics_noisy.process()

    #we're going to normalize the noise
    size = np.shape(audio_reverb)
    noise_normalized = np.zeros(size)
    
    #for each microphones
    if(len(noise_reverb[0]) < len(audio_reverb[0])):
        raise ValueError('the length of the noise signal is inferior to the one of the audio signal !!')
    output_noise = output_noise[:len(output_signal)]

    norm_fact = np.linalg.norm(noise_reverb[-1])
    noise_normalized = output_noise / norm_fact

    #initilialize the array of noisy_signal
    noisy_signal = np.zeros([len(snr_vals),np.shape(output_signal)[0]])

    for i,snr in enumerate(snr_vals):
        noise_std = np.linalg.norm(audio_reverb[-1])/(10**(snr/20.))
        final_noise = noise_normalized*noise_std
        noisy_signal[i] = pra.normalize(pra.highpass(output_signal + final_noise,fs_s))

    return noisy_signal



