#from lstm_single_instrument import *
#from lstm_piano_v0 import *
from lstm_single_instrument_v2 import *

if __name__ == '__main__':
    #model_path = "/home/agatha/Documents/EE298z/miniproj-2/prelim/music_generation_lstm/output/midi_bts_single_boy_with_luv_input_50/LSTMmodel_00000090.h5"
    # #model_path = "/home/agatha/Documents/EE298z/miniproj-2/prelim/music_generation_lstm/output/single_piano/midi_bts_boy_with_luv/LSTMmodel_00000100.h5"
    # model_path = "/home/agatha/Documents/EE298z/miniproj-2/prelim/music_generation_lstm/output/lstm_single_instrument_v2/midi_bts_boy_with_luv/LSTMmodel_00000100.h5"
    # generate_given_model_path(model_path, random_seed=1000)

    model_path = "/home/agatha/Documents/EE298z/miniproj-2/prelim/music_generation_lstm/output/lstm_single_instrument_v2/midi_bts_boy_with_luv/LSTMmodel_00000100.h5"
    start_time = time.time()

    generate_given_model_path(model_path, random_seed=1000)

    end_time = time.time()
    print("Total time elapsed: {}".format(end_time - start_time))
