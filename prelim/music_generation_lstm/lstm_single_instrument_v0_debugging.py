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

# INPUT_FOLDER = './input/midi_bts_single_boy_with_luv'
# OUTPUT_FOLDER = './output/midi_bts_single_boy_with_luv_input_100'
INPUT_FOLDER = './data/preprocessed/single_piano/midi_bts_boy_with_luv'
OUTPUT_FOLDER = './output/single_piano/midi_bts_boy_with_luv_lstm_single_instrument_v0_debugging'

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
            prediction_output, random_seed = generate_notes(self.model, self.notes, self.network_input, len(set(self.notes)))
            create_midi(prediction_output, filepath)
            if self.verbose > 0:
                 print('\nEpoch %05d: saving data to %s' % (epoch + 1, filepath))

"""
######################################################

Data Preprocessing

######################################################
"""


def get_notes():
    """ Get all the notes and chords from the midi files """
    notes = []

    for file in glob.glob('./' + INPUT_FOLDER + "/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    print("len(note_to_int)")
    print(len(note_to_int))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    # reshape the input into a format compatible with LSTM layers
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    print(network_input[0])
    print(network_input.shape)
    sys.exit()

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
    notes = get_notes()

    # Get the number of pitch names
    n_vocab = len(set(notes))

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
    create_midi(prediction_output, file_name)

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
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    pitchnames = sorted(set(item for item in notes))

    np.random.seed(random_seed)

    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern,index)
        pattern = pattern[1:len(pattern)]

    return (prediction_output, random_seed)

def create_midi(prediction_output, filename):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='{}.mid'.format(filename))

def generate_given_model_path(file_path, random_seed=None):
    """ Generate a song given the file path to the model 
        and an optional random seed """

    notes = get_notes()

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
    create_midi(prediction_output, file_name)

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

