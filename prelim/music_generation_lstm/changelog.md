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

## lstm_single_instrument_v3
- one hot encoding including velocity