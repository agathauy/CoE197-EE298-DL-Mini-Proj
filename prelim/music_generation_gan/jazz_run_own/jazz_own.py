#import dependencies

from __future__ import print_function, division

import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob

from music21 import converter, instrument, note, chord, stream
from keras.layers import Input, Dense, Reshape, Dropout, CuDNNLSTM, Bidirectional
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import np_utils
import time

def get_notes():
    #using music21, we get all notes and chords from the MIDI dataset
    notes = []
    # i = 0
    for file in glob.glob("jazz_data/*.mid"):
        midi = converter.parse(file)
        # i = i + 1
        # if i > 5: break
        print("Parsing %s" % file)

        notes_to_parse = None

        # try: # file has instrument parts
        #     s2 = instrument.partitionByInstrument(midi)
        #     notes_to_parse = s2.parts[0].recurse() 
        # except: # file has notes in a flat structure
        #     notes_to_parse = midi.flat.notes
        
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

def prepare_sequences(notes, n_vocab):
    #sequence length determines how many notes/chords will the network need  in order
    #to predict the next note in the sequence
    #orig: sequence_length = 100
    sequence_length = 500

    #Get all pitch names
    pitchnames = sorted(set(item for item in notes))

    #map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # Normalize input between -1 and 1
    network_input = (network_input - float(n_vocab)/2) / (float(n_vocab)/2)
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


