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
import lib.file_util_v2
from lib.midi_util_v2 import midi_to_array_one_hot, stylify
from mido import MidiFile
from random import shuffle

import pickle

#from lib.midi_util import 


# The MIDI pitches we use.
# PITCHES = range(21,109,1)
# OFFSET = 109-21

# # The reverse of what is in encoding
# PITCHES_MAP = { i : p for i, p in enumerate(PITCHES) }
# print(len(PITCHES))




# A state is composed of the order of pitches contained in PITCHES_MAP
# state = 3*length of pitches_map
# where each pitch_state in a state is composed of [active or not, event, velocity]
PITCHES_TO_INDEX = {}
INDEX_TO_PITCHES = {}

# STATES TO INDICES
STATES_TO_INDEX = {}
INDEX_TO_STATES = {}



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
OUTPUT_FOLDER = './output/lstm_single_instrument_v2/midi_bts_boy_with_luv'

PREPROCESSED_FOLDER =  OUTPUT_FOLDER + '/preprocessed'


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
            print('length_notes: ')
            #print(length_notes)

            prediction_output, random_seed = generate_notes(self.model, self.notes, self.network_input, length_notes)
            #print('prediction_output')
            #print(prediction_output)

            create_midi(prediction_output, filepath, self.notes)
            if self.verbose > 0:
                 print('\nEpoch %05d: saving data to %s' % (epoch + 1, filepath))

"""
######################################################

Data Preprocessing

######################################################
"""


def get_notes_and_velocities():
    """
        Get all the notes from the midi files

        Creates the PITCHES_TO_INDEX and INDEX_TO_PITCHES mapping for the pitch dictionaries

        returns the list of 


    """
    global PITCHES_TO_INDEX
    global INDEX_TO_PITCHES


    notes = np.array([])
    velocities = np.array([])


    # CREATE PITCHES_MAP, a mapping of unique values
    # a dictionary always has unique values

    # first count the number of pitches for all files
    # for instrument in midi_data.instruments:
    #     print(instrument)
    #     for i, note in enumerate(instrument.notes):
    #         print(note)
    #         note_name = pretty_midi.note_number_to_name(note.pitch)
    #         print(note_name)
    #         print(pretty_midi.note_name_to_number(note_name))
    #         if (i > 5):
    #             break

    list_notes = []
    list_mid_data = []
    for file in glob.glob(INPUT_FOLDER + "/*.mid"):
        midi_data = pretty_midi.PrettyMIDI(file)
        for instrument in midi_data.instruments:
            print(instrument)
            for i, note in enumerate(instrument.notes):
                #print(note)
                #note_name = pretty_midi.note_number_to_name(note.pitch)
                list_notes.append(note.pitch)
                #list_notes.append(note_name)
                #print(pretty_midi.note_name_to_number(note_name))
    #print(len(list_notes))
    set_notes = set(item for item in list_notes)
    num_unique_notes = len(set_notes)
    #print(num_unique_notes)

    # Create the mappings
    for i, item in enumerate(set_notes):
        PITCHES_TO_INDEX[item] = i
        INDEX_TO_PITCHES[i] = item

    #print(len(PITCHES_TO_INDEX))
    #print(PITCHES_TO_INDEX)
    #print(len(INDEX_TO_PITCHES))
    #print(INDEX_TO_PITCHES)
    #import pickle

    file_path = OUTPUT_FOLDER + '/PITCHES_TO_INDEX.pkl'
    with open(file_path, "wb") as f:
        pickle.dump(PITCHES_TO_INDEX, f)

    file_path = OUTPUT_FOLDER + '/INDEX_TO_PITCHES.pkl'
    with open(file_path, "wb") as f:
        pickle.dump(INDEX_TO_PITCHES, f)

    list_midi_data = None
    # Get the quantized midi to array of each file
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

        midi_array = midi_to_array_one_hot(mid, 4, PITCHES_TO_INDEX)
        print("[MIDI_ARRAY]")
        #print(midi_array)
        #print(midi_array.shape)
        #test = midi_array.tolist()
        #print(test[0])
        #print(len(test[0]))
        #notes.append(notes, midi_array[:])
        #notes = np.concatenate((notes, midi_array), axis=0)
        #print(notes[0:5])
        #print(notes.shape)

        # Add in each song as part of a big sequence.
        # These are in effect the list of NON-UNIQUE sequnetial states
        if list_midi_data == None:
            list_midi_data = midi_array
        else:
            list_midi_data = np.vstack((list_midi_data, midi_array))
        #list_midi_data = np.concatenate((list_midi_data, midi_array), axis=0)
        #break

    return list_midi_data

def prepare_sequences(notes, n_vocab):
    """ 
        Prepare the sequences used by the Neural Network.
        Converts the sequences of states into embeddings based on list of unique states
        Returns
            network_input
            network_output


    """
    sequence_length = 100
    #print(len(notes))
    #print(notes.shape)
    #print(notes[0])

    #n = np.unique(notes, axis=0)
    #print(n[0])
    #print(n.shape)
    #pitchnames = n.tolist()
    #print(len(n))
    #print(len(pitchnames))
    #print(pitchnames[0:5])
    print(notes.shape)
    temp_notes = notes.tolist()
    #print(temp_notes[0:5])
    # get all pitch names
    #pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    #Counter(str(e) for e in li)

    #note_to_int = dict((str(note), number) for number, note in enumerate(pitchnames))
    #print("len(note_to_int)")
    #print(len(note_to_int))
    #print(note_to_int[0])
    #print(note_to_int)
    network_input = []
    network_output = []

    #print(STATES_TO_INDEX)
    # create input sequences and the corresponding outputs
    for i in range(0, len(temp_notes) - sequence_length, 1):
        sequence_in = temp_notes[i:i + sequence_length]
        sequence_out = temp_notes[i + sequence_length]
        network_input.append([STATES_TO_INDEX[str(state)] for state in sequence_in])
        #print(network_input[i])
        #print(len(network_input))
        network_output.append(STATES_TO_INDEX[str(sequence_out)])
        #print(network_output[i])
        #print(len(network_output))

    #print(network_input[0])
    #print(network_output[0])

    # reshape the input into a format compatible with LSTM layers
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    #print(network_input.shape)
    # normalize input between 0 and 1
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)
    #print(network_output.shape)
    #print(network_output[0:5])
    #print(len(np.unique(network_output, axis=0)))


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

    global STATES_TO_INDEX
    global INDEX_TO_STATES

    # Get list of all states from midi files
    notes = get_notes_and_velocities()

    # print(notes.shape)
    #print(notes[0])

    # Get the number of unique instances
    # So we can create an embedding!
    n = np.unique(notes, axis=0)
    # print(n.shape)

    # The number of unique states/"notes"
    n_vocab = n.shape[0]

    for i, item in enumerate(n):
        STATES_TO_INDEX[str(item.tolist())] = i
        INDEX_TO_STATES[i] = item.tolist()

    # print(len(STATES_TO_INDEX))
    # print(len(INDEX_TO_STATES))
    # print(INDEX_TO_STATES[0]) # match a
    # print(INDEX_TO_STATES[1])
    # print(str(INDEX_TO_STATES[0])) # match a
    # print(STATES_TO_INDEX[str(INDEX_TO_STATES[0])])
    # print(STATES_TO_INDEX[str(INDEX_TO_STATES[1])])


    file_path = OUTPUT_FOLDER + '/STATES_TO_INDEX.pkl'
    with open(file_path, "wb") as f:
        pickle.dump(STATES_TO_INDEX, f)

    file_path = OUTPUT_FOLDER + '/INDEX_TO_STATES.pkl'
    with open(file_path, "wb") as f:
        pickle.dump(INDEX_TO_STATES, f)


    # Convert the states into numerical input!
    # Base this on their encodings given unique states
    network_input, network_output = prepare_sequences(notes, n_vocab)

    #print(network_input[0])
    # Set up the model
    model = create_network(network_input, n_vocab)
    history = History()

    

    mc = ModelCheckpoint('./' + OUTPUT_FOLDER + '/LSTMmodel_{epoch:08d}.h5', 
                                     save_weights_only=True, period=10)
    mc_gd = ModelCheckpoint_GenerateData('./' + OUTPUT_FOLDER + '/out_{epoch:08d}', \
        period=10,verbose=True, notes=notes, network_input=network_input)
    try:

        model.summary()
        model.fit(network_input, network_output, epochs=n_epochs, batch_size=64, callbacks=[history,mc, mc_gd], validation_split=0.4)
        model.save('./' + OUTPUT_FOLDER + '/LSTMmodel.h5')

        # Use the model to generate a midi
        prediction_output, random_seed = generate_notes(model, notes, network_input, len(n_vocab))
        #create_midi(prediction_output, 'pokemon_midi')
        file_name = './' + OUTPUT_FOLDER + '/out_' + str(n_epochs)
        create_midi(prediction_output, file_name, notes)

    except KeyboardInterrupt:
        print("Keyboard interrupt")

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

        prediction_output will be a series of states of format [123, 0, 290, ...] and need to be
        converted back to state form with INDEX_TO_STATES


    """
    # pick a random sequence from the input as a starting point for the prediction
    #pitchnames = sorted(set(item for item in notes))
    #PITCHES_MAP

    np.random.seed(random_seed)


    # n = np.unique(notes, axis=0)
    # #print(n[0])
    # print(n.shape)
    # pitchnames = n.tolist()

    start = np.random.randint(0, len(network_input)-1)

    # int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    print("Starting pattern: ")
    # print(pattern)
    # print(start)
    prediction_output = []

    # generate 2048 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        #print('prediction_input shape:')
        #print(prediction_input.shape)
        #print(prediction_input)
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        #print("model's prediction")
        #print(prediction)
        index = np.argmax(prediction)
        #print("max index: ")
        #print(index)
        result = index

        prediction_output.append(result)

        pattern = np.append(pattern,index)
        pattern = pattern[1:len(pattern)]


    return (prediction_output, random_seed)

def create_midi(prediction_output, filename, notes):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []
    print("here in create_midi")
    piano_c_chord = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name='Piano')
    # create note and chord objects based on the values generated by the model
    max_len = len(prediction_output)
    print("max_len")
    print(max_len)
    print(prediction_output)
    on_list = []
    on_list_pos_only = []
    init = 0
    offset = 0

    #print(INDEX_TO_STATES)


    # this signifies a LIST OF STATES
    # where each number in the list represents a list
    for i, state_num in enumerate(prediction_output):
        #print(pattern)
        #print(len(pattern))
        #print(state_num)
        pattern = INDEX_TO_STATES[state_num]

        # i = tick, can be converted into time
        # note on [1,1]
        # note sustained [0, 1]
        # note off [0, 0]
        arr_pattern = np.array(pattern)
        #print('pattern')
        #print(arr_pattern)
        #print(arr_pattern.shape)
        indices = np.where(arr_pattern != 0)[0]
        #print(indices)
        indices = indices.tolist()

        #print(indices)
        # check all ones

        for _, pos in enumerate(indices):
            # look at the value of the indices

            # Check for new notes
            # Check if note is on
            if (pos % 3) == 0:
                # in a  position of interval of 3
                if not pos in on_list_pos_only:
                    # this is not a sustain
                    # append [index of state, tick, velocity based on state]
                    on_list.append([pos, i, INDEX_TO_STATES[pos + 2]])
                    on_list_pos_only.append(pos)
            else:
                # odd number position, possibly a velocity or note state
                # check for sustain/consider as new note if it appears
                if init == 1:
                    for j, on in enumerate(on_list):
                        if on[0] == (pos - 1):
                            # for sustain
                            on_list[j][1] = on_list[j][1] + 1 # add another tick to the sustain
                            break
                        elif on[0] == (pos -2):
                            # for velocity
                            on_list[j][2] = 0

            # to always active init after first loop
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
                    print("Appending note: {}".format(INDEX_TO_PITCHES[int(on[0]/3)]))
                    note = pretty_midi.Note(velocity=on[2], pitch=INDEX_TO_PITCHES[int(on[0]/3)], start=start_time, end=end_time)
                    # add it to our piano 
                    piano.notes.append(note)


                    # remove the note
                    on_list_pos_only.remove(on[0])
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
    global INDEX_TO_STATES
    global STATES_TO_INDEX

    global PITCHES_TO_INDEX
    global INDEX_TO_PITCHES

    file_path = OUTPUT_FOLDER + '/INDEX_TO_STATES.pkl'
    INDEX_TO_STATES = pickle.load( open( file_path, "rb" ) )

    file_path = OUTPUT_FOLDER + '/STATES_TO_INDEX.pkl'
    STATES_TO_INDEX = pickle.load( open( file_path, "rb" ) )

    file_path = OUTPUT_FOLDER + '/PITCHES_TO_INDEX.pkl'
    PITCHES_TO_INDEX = pickle.load( open( file_path, "rb" ) )

    file_path = OUTPUT_FOLDER + '/INDEX_TO_PITCHES.pkl'
    INDEX_TO_PITCHES = pickle.load( open( file_path, "rb" ) )

    notes = get_notes_and_velocities()


    n = np.unique(notes, axis=0)
    print(n.shape)

    n_vocab = n.shape[0]

    # Convert notes into numerical input
    network_input, network_output = prepare_sequences(notes, n_vocab)


    model = create_network(network_input, n_vocab)
    model.load_weights(file_path)


    # Use the model to generate a midi
    prediction_output, rs = generate_notes(model, notes, network_input, n_vocab, random_seed)
    #create_midi(prediction_output, 'pokemon_midi')
    file_name = './' + OUTPUT_FOLDER + '/out_rand_' + str(rs)
    create_midi(prediction_output, file_name, notes)
    print(file_name)

    return file_name



def continue_training_network(start_epoch, end_epoch):
    global INDEX_TO_STATES
    global STATES_TO_INDEX

    global PITCHES_TO_INDEX
    global INDEX_TO_PITCHES

    file_path = OUTPUT_FOLDER + '/INDEX_TO_STATES.pkl'
    INDEX_TO_STATES = pickle.load( open( file_path, "rb" ) )

    file_path = OUTPUT_FOLDER + '/STATES_TO_INDEX.pkl'
    STATES_TO_INDEX = pickle.load( open( file_path, "rb" ) )

    file_path = OUTPUT_FOLDER + '/PITCHES_TO_INDEX.pkl'
    PITCHES_TO_INDEX = pickle.load( open( file_path, "rb" ) )

    file_path = OUTPUT_FOLDER + '/INDEX_TO_PITCHES.pkl'
    INDEX_TO_PITCHES = pickle.load( open( file_path, "rb" ) )

    # Get list of all states from midi files
    notes = get_notes_and_velocities()

    print(notes.shape)
    print(notes[0])

    # Get the number of unique instances
    # So we can create an embedding!
    n = np.unique(notes, axis=0)
    print(n.shape)

    # The number of unique states/"notes"
    n_vocab = n.shape[0]


    print(len(STATES_TO_INDEX))
    print(len(INDEX_TO_STATES))
    print(INDEX_TO_STATES[0]) # match a
    print(INDEX_TO_STATES[1])
    print(str(INDEX_TO_STATES[0])) # match a
    print(STATES_TO_INDEX[str(INDEX_TO_STATES[0])])
    print(STATES_TO_INDEX[str(INDEX_TO_STATES[1])])


    # Convert the states into numerical input!
    # Base this on their encodings given unique states
    network_input, network_output = prepare_sequences(notes, n_vocab)

    #print(network_input[0])
    # Set up the model
    model = create_network(network_input, n_vocab)
    history = History()


    # Load from previous file!
    file_path = OUTPUT_FOLDER + "/LSTMmodel_{:08d}.h5".format(start_epoch)
    model.load_weights(file_path)



    mc = ModelCheckpoint('./' + OUTPUT_FOLDER + '/LSTMmodel_start_' + str(start_epoch) + "_{epoch:08d}.h5", 
                                     save_weights_only=True, period=10)
    mc_gd = ModelCheckpoint_GenerateData('./' + OUTPUT_FOLDER + '/out_start_' + str(start_epoch) + '_{epoch:08d}', \
        period=10,verbose=True, notes=notes, network_input=network_input)

    n_epochs = end_epoch - start_epoch
    try:

        model.summary()
        model.fit(network_input, network_output, epochs=n_epochs, batch_size=64, callbacks=[history,mc, mc_gd], validation_split=0.4)
        model.save('./' + OUTPUT_FOLDER + '/LSTMmodel.h5')

        # Use the model to generate a midi
        prediction_output, random_seed = generate_notes(model, notes, network_input, len(n_vocab))
        #create_midi(prediction_output, 'pokemon_midi')
        file_name = './' + OUTPUT_FOLDER + '/out_' + str(n_epochs)
        create_midi(prediction_output, file_name, notes)

    except KeyboardInterrupt:
        print("Keyboard interrupt")

    # Plot the model losses
    pd.DataFrame(history.history).plot()
    plt.savefig('./' + OUTPUT_FOLDER + '/LSTM_Loss_per_Epoch.png')
    plt.close()

"""
######################################################

Main

######################################################
"""

# if __name__ == "__main__":
#     start_time = time.time()
#     try:
#         train_network(500)
#     except:
#         print("Error in main")

#     end_time = time.time()
#     print("Total time elapsed: {}".format(end_time - start_time))


if __name__ == "__main__":
    start_time = time.time()
    try:
        #train_network(500)
        start_epoch = 140
        end_epoch = 500
        continue_training_network(start_epoch, end_epoch)
    except Exception as e:
        print(e)
        print("Error in main")

    end_time = time.time()
    print("Total time elapsed: {}".format(end_time - start_time))

# if __name__ == "__main__":
#     model_path = "/home/agatha/Documents/EE298z/miniproj-2/prelim/music_generation_lstm/output/lstm_single_instrument_v2/midi_bts_boy_with_luv/LSTMmodel_00000100.h5"
# start_time = time.time()

# generate_given_model_path(model_path, random_seed=1000)

# end_time = time.time()
# print("Total time elapsed: {}".format(end_time - start_time))


