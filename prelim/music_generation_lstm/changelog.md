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

## lstm_single_instrument_v2
- make use of quantization and do not manually do one-hot encoding
- one hot encoding including 


## lstm_single_instrument_v3
- one hot encoding including velocity

CREATE MIDI TIME: 0.009964704513549805
TOTAL GENERATION TIME ELAPSED: 6.631846189498901

Epoch 00500: saving data to ././output/lstm_single_instrument_v3/midi_bts_boy_with_luv/out_00000500
object of type 'int' has no len()
error in train_network
Total time elapsed: 2071.138938188553

## lstm_single_intstrument_4
- simple LSTM model based on lstm_single_instrument_v0

## lstm_single_instrument_5
- modifed lstm 3 to have constant velocity
- still have running problems

CREATE MIDI TIME: 0.008070230484008789
TOTAL GENERATION TIME ELAPSED: 16.918411254882812

Epoch 00040: saving data to ././output/lstm_single_instrument_v5/midi_bts_boy_with_luv/out_00000040
Epoch 41/500
1088/1948 [===============>..............] - ETA: 3s - loss: 0.1222^CKeyboard interrupt
Total time elapsed: 445.67898631095886
(ee298z) agatha@a-pc:~/Documents/EE298z/miniproj-2/prelim/music_generation_lstm$ 

