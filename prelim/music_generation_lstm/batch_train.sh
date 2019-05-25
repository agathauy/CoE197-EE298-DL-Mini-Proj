#!/bin/bash

# Stardew valley overture
python lstm_single_instrument_v6.py -i ./data/raw/stardew_valley_overture -o ./output/lstm_single_instrument_v6/stardew_valley_overture -e 150

# Super Mario bros
python lstm_single_instrument_v6.py -i ./data/raw/super_mario_bros_main -o ./output/lstm_single_instrument_v6/super_mario_bros_main -e 150


# Plants vs Zombies main menu
python lstm_single_instrument_v6.py -i ./data/raw/plants_vs_zombies -o ./output/lstm_single_instrument_v6/plants_vs_zombies -e 150

# Jazz dataset
python lstm_single_instrument_v6.py -i ./data/raw/jazz_subset -o ./output/lstm_single_instrument_v6/jazz_subset -e 400


# BTS sounds
# prelim/music_generation_lstm/data/preprocessed/midi_type_0/midi_bts
python lstm_single_instrument_v6.py -i ./data/preprocessed/midi_type_0/midi_bts -o ./output/lstm_single_instrument_v6/midi_bts -e 400


# Multi super mario bros
python lstm_single_instrument_v6.py -i ./data/raw/Super_Mario_Bros -o ./output/lstm_single_instrument_v6/Super_Mario_Bros -e 400

