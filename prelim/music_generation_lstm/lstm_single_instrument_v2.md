Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /home/agatha/miniconda3/envs/ee298z/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cu_dnnlstm_1 (CuDNNLSTM)     (None, 100, 512)          1054720   
_________________________________________________________________
dropout_1 (Dropout)          (None, 100, 512)          0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 100, 1024)         4202496   
_________________________________________________________________
dropout_2 (Dropout)          (None, 100, 1024)         0         
_________________________________________________________________
bidirectional_2 (Bidirection (None, 1024)              6299648   
_________________________________________________________________
dense_1 (Dense)              (None, 256)               262400    
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 374)               96118     
_________________________________________________________________
activation_1 (Activation)    (None, 374)               0         
=================================================================
Total params: 11,915,382
Trainable params: 11,915,382
Non-trainable params: 0
____________________________