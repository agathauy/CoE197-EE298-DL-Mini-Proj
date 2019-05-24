# Adapted from StyleNet

import pretty_midi
import matplotlib.pyplot as plt
import os
from lib.midi_util import quantize
from mido import MidiFile
from random import shuffle

#mid_path = '/Users/Iman/Desktop/jazz'
#out_path = '/Users/Iman/Desktop/jazz_out'
mid_path = './twinkle_data/'
out_path = './'


if not os.path.exists(out_path):
    os.makedirs(out_path)

total = len(os.listdir(mid_path))


for i , filename in enumerate(os.listdir(mid_path)):
    print(filename)
    if filename.split('.')[-1] == 'mid' or filename.split('.')[-1] == 'MID' or filename.split('.')[-1] == 'midi':
        print("%d / %d" % (i,total))
        print(filename)
        try:
            midi_data = pretty_midi.PrettyMIDI(os.path.join(mid_path, filename))
            mid = MidiFile(os.path.join(mid_path, filename))
        except (KeyError, IOError, IndexError, EOFError, ValueError):
            print("NAUGHTY")
            continue

        time_sig_msgs = [ msg for msg in mid.tracks[0] if msg.type == 'time_signature' ]

        if len(time_sig_msgs) == 1:
            time_sig = time_sig_msgs[0]
            if not (time_sig.numerator == 4 and time_sig.denominator == 4):
                print('\tTime signature not 4/4. Skipping ...')
                continue
        else:
            print('\tNo time signature. Skipping ...')
            continue


        mid_q = quantize(mid, 4)

        if not mid_q:
            print('Invalid MIDI. Skipping...')
            continue


        piano = [instrument for instrument in midi_data.instruments if instrument.program < 8 ]
        piano = [instrument for instrument in piano if not instrument.is_drum ]

        if len(piano) > 0 and len(piano) < 3:
            for x in piano:
                x.program = 0

            midi_data.instruments = piano
            midi_data.write(os.path.join(out_path, filename))
        else:
            print('\tNO piano.')
        
        print("Success!")
