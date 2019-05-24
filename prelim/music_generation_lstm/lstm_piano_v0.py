"""
# Changelog:

## lstm_single_instrument_v0
- initial adaptation from Pokemon

## lstm_single_instrument_v1
- changed lstm nodes to 1024
- comment is to revert to 512

## lstm_piano_v0
- from lstm_single_instrument_v0
- make use of one-hot encoding for taking note of note durations
- make use of stylenet's implementation

"""

import glob
import numpy as np
import pandas as pd
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM, LSTM, Bidirectional
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, History, Callback
import matplotlib.pyplot as plt
import time
import os
import sys

from collections import Counter

# From stylenet testings "check piano representation"
import pretty_midi
import matplotlib.pyplot as plt
import os
import lib.file_util
from lib.midi_util import midi_to_array_one_hot, stylify
from mido import MidiFile
from random import shuffle

#from lib.midi_util import 


# The MIDI pitches we use.
PITCHES = range(21,109,1)
OFFSET = 109-21

# The reverse of what is in encoding
PITCHES_MAP = { i : p for i, p in enumerate(PITCHES) }
print(len(PITCHES))

"""
######################################################

Setting Input and Output Folder

######################################################
"""

#INPUT_FOLDER = './input/magenta_2013'
#OUTPUT_FOLDER = './output/magenta_2013'
#INPUT_FOLDER = './input/Bach_Fuga sopra il Magnificat in D minor'
#OUTPUT_FOLDER = './output/Bach_Fuga sopra il Magnificat in D minor'
# INPUT_FOLDER = 'input/Bach_wv1041a'
#INPUT_FOLDER = 'input/Golden_Sun_2'
#OUTPUT_FOLDER = 'output/Golden_Sun_2_v0'
# OUTPUT_FOLDER = 'output/Bach_wv1041a'

INPUT_FOLDER = './data/preprocessed/single_piano/midi_bts_boy_with_luv'
PREPROCESSED_FOLDER =  './output/single_piano/midi_bts_boy_with_luv/preprocessed'
OUTPUT_FOLDER = './output/single_piano/midi_bts_boy_with_luv'


# INPUT_FOLDER = './input/midi_bts_single_boy_with_luv'
# OUTPUT_FOLDER = './output/midi_bts_single_boy_with_luv_input_100'

if not os.path.exists(INPUT_FOLDER):
    print("Input folder: {} does not exist".format(INPUT_FOLDER))
    sys.exit()

if not os.path.exists(OUTPUT_FOLDER):
    print("Creating folder: {}".format(OUTPUT_FOLDER))
    os.makedirs(OUTPUT_FOLDER)

"""
######################################################

Custom Training Callbacks

Custom callback for generating data as the model trains (so we can view the data immediately)

######################################################
"""
class ModelCheckpoint_GenerateData(Callback):
    """Generate data with model every given time period
    """

    def __init__(self, filepath, notes, network_input, period=1, verbose=True):
        super(ModelCheckpoint_GenerateData, self).__init__()
        self.filepath = filepath
        self.period = period
        self.epochs_since_last_save = 0
        self.verbose = verbose
        # Data needed for data generation
        self.notes = notes
        self.network_input = network_input

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch+1, **logs)
            # Do something with the model
            length_notes =  np.unique(self.notes, axis=0).shape[0]
            prediction_output, random_seed = generate_notes(self.model, self.notes, self.network_input, length_notes)
            create_midi(prediction_output, filepath, self.notes)
            if self.verbose > 0:
                 print('\nEpoch %05d: saving data to %s' % (epoch + 1, filepath))

"""
######################################################

Data Preprocessing

######################################################
"""


def get_notes_and_velocities():
    """ Get all the notes and chords from the midi files """
    notes = np.array([])
    velocities = np.array([])

    for file in glob.glob(INPUT_FOLDER + "/*.mid"):
        #midi = converter.parse(file)

        print("Parsing %s" % file)
        #mid_path = os.path
        try:
            midi_data = pretty_midi.PrettyMIDI(file)
            mid = MidiFile(file)
        except (KeyError, IOError, IndexError, EOFError, ValueError):
            print("NAUGHTY")
            break
            
        midi_array, velocity_array = midi_to_array_one_hot(mid, 4)
        
        print("[MIDI_ARRAY]")
        print(midi_array)
        print(midi_array.shape)
        test = midi_array.tolist()
        print(test[0])
        print(len(test[0]))
        #notes.append(notes, midi_array[:])
        #notes = np.concatenate((notes, midi_array), axis=0)
        #print(notes[0:5])
        print(notes.shape)

        print("[VELOCITY_ARRAY]")
        print(velocity_array)
        print(velocity_array.shape)
        # Just test one file
        #break

    return (midi_array, velocity_array)

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100
    #print(len(notes))
    #print(notes.shape)
    #print(notes[0])

    n = np.unique(notes, axis=0)
    #print(n[0])
    print(n.shape)
    pitchnames = n.tolist()
    #print(len(n))
    #print(len(pitchnames))


    # get all pitch names
    #pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    #Counter(str(e) for e in li)

    note_to_int = dict((str(note), number) for number, note in enumerate(pitchnames))
    #print(note_to_int)
    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[str(char)] for char in sequence_in])
        network_output.append(note_to_int[str(sequence_out)])

    #print(network_input[0])
    #print(network_output[0])
    #sys.exit()
    # reshape the input into a format compatible with LSTM layers
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # normalize input between 0 and 1
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

"""
######################################################

LSTM Network Structure

######################################################
"""

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(CuDNNLSTM(512,input_shape=(network_input.shape[1], network_input.shape[2]),return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Bidirectional(CuDNNLSTM(512, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(CuDNNLSTM(512)))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

"""
######################################################

Training

######################################################
"""

def train_network(n_epochs=50):
    """ Train a Neural Network to generate music """
    # Get notes from midi files
    notes, velocities = get_notes_and_velocities()
    print(notes.shape)
    # Get the number of unique instances
    #print(notes.tolist())
    #temp_n = Counter(str(e) for e in notes)
    #temp_v = Counter(str(e) for e in velocities)
    n = np.unique(notes, axis=0)
    print(n.shape)
    v = np.unique(velocities, axis=0)
    print(v.shape)

    n_vocab = n.shape[0]
    notes = notes.tolist()
    #print(notes.tolist()[0])
    #sys.exit()

    # Convert notes into numerical input
    network_input, network_output = prepare_sequences(notes, n_vocab)

    # Set up the model
    model = create_network(network_input, n_vocab)
    history = History()


    mc = ModelCheckpoint('./' + OUTPUT_FOLDER + '/LSTMmodel_{epoch:08d}.h5', 
                                     save_weights_only=True, period=5)
    mc_gd = ModelCheckpoint_GenerateData('./' + OUTPUT_FOLDER + '/out_{epoch:08d}', \
        period=5,verbose=True, notes=notes, network_input=network_input)

    model.summary()
    model.fit(network_input, network_output, epochs=n_epochs, batch_size=64, callbacks=[history,mc, mc_gd])
    model.save('./' + OUTPUT_FOLDER + '/LSTMmodel.h5')

    # Use the model to generate a midi
    prediction_output, random_seed = generate_notes(model, notes, network_input, len(set(notes)))
    #create_midi(prediction_output, 'pokemon_midi')
    file_name = './' + OUTPUT_FOLDER + '/out_' + str(n_epochs)
    create_midi(prediction_output, file_name, notes)

    # Plot the model losses
    pd.DataFrame(history.history).plot()
    plt.savefig('./' + OUTPUT_FOLDER + '/LSTM_Loss_per_Epoch.png')
    plt.close()



"""
######################################################

Generation of prediction to MIDI

######################################################
"""

def generate_notes(model, notes, network_input, n_vocab, random_seed=None):
    """
        Generate notes from the neural network based on a sequence of notes 

        prediction_output will be a series of embeddings of format [0, 0, 0, 1, 1 ...] etc

        Need to convert these time slices back


    """
    # pick a random sequence from the input as a starting point for the prediction
    #pitchnames = sorted(set(item for item in notes))
    #PITCHES_MAP

    np.random.seed(random_seed)


    n = np.unique(notes, axis=0)
    #print(n[0])
    print(n.shape)
    pitchnames = n.tolist()

    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 2048 notes
    for note_index in range(2048):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern,index)
        pattern = pattern[1:len(pattern)]

    return (prediction_output, random_seed)

def create_midi(prediction_output, filename, notes):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []
    
    piano_c_chord = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name='Piano')
    # create note and chord objects based on the values generated by the model
    max_len = len(prediction_output)
    print("max_len")
    print(max_len)

    on_list = []
    init = 0
    offset = 0
    for i, pattern in enumerate(prediction_output):

        # note on [1,1]
        # note sustained [0, 1]
        # note off [0, 0]
        arr_pattern = np.array(pattern)
        #print(arr_pattern)
        indices = np.where(pattern == 1)[0]
        #print(indices)
        indices = indices.tolist()

        # check all ones

        for _, pos in enumerate(indices):

            # Check for new notes
            # Check if note is on
            if pos % 0:
                on_list.append([pos, i])
            else:
                # odd number position
                # check for sustain/consider as new note if it appears
                if init == 1:
                    for j, on in enumerate(on_list):
                        if on[0] == (pos - 1):
                            on_list[j, 0] = on_list[j, 1] + 1 # add another tick to the sustain
                            break
            init = 1

        # Check if anything from the on_list is now off
        for _, on in enumerate(on_list):
            if (on[0] not in indices):
                # check if not sustain
                if ((on[0] + 1) not in indices):
                    # Remove the note and add to midi file
                    # constant velocity for now
                    start_tick = on[1]
                    end_tick = i
                    #total_ticks = end_tick - start_tick
                    start_time = start_tick * 0.5
                    end_time = end_tick * 0.5

                    note = pretty_midi.Note(velocity=90, pitch=PITCHES_MAP[int(on[0]/2)], start=start_time, end=end_time)
                    # add it to our piano 
                    piano.notes.append(note)


                    # remove the note
                    on_list.remove(on)


    # Add the piano instrument to the PrettyMIDI object
    piano_c_chord.instruments.append(piano)
    
    # Write out the MIDI data
    midi_name = filename + '.mid'
    piano_c_chord.write(midi_name)

    # midi_stream = stream.Stream(output_notes)
    # midi_stream.write('midi', fp='{}.mid'.format(filename))

def generate_given_model_path(file_path, random_seed=None):
    """ Generate a song given the file path to the model 
        and an optional random seed """

    notes = get_notes_and_velocities()

    # Get the number of pitch names
    n_vocab = len(set(notes))

    # Convert notes into numerical input
    network_input, network_output = prepare_sequences(notes, n_vocab)


    model = create_network(network_input, n_vocab)
    model.load_weights(file_path)


    # Use the model to generate a midi
    prediction_output, rs = generate_notes(model, notes, network_input, len(set(notes)), random_seed)
    #create_midi(prediction_output, 'pokemon_midi')
    file_name = './' + OUTPUT_FOLDER + '/out_rand_' + str(rs)
    create_midi(prediction_output, file_name, notes)

    return output_path



"""
######################################################

Main

######################################################
"""

if __name__ == "__main__":
    start_time = time.time()
    train_network(100)
    end_time = time.time()
    print("Total time elapsed: {}".format(end_time - start_time))

