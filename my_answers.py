import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    step_size = 1
    for i in range(0, len(series), step_size):
        
        # current seq window 
        seq_from         = i 
        seq_to_exclusive = seq_from + window_size
        
        # make sure that there is an y left 
        # in case of len(series) % window_size == 0
        
        if seq_to_exclusive < len(series):
            X.append(series[seq_from:seq_to_exclusive])
            y.append(series[seq_to_exclusive])
            
    # reshape each 
    X = np.asarray(X)
    y = np.asarray(y)
    y.shape = (len(y),1)
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    import string
    punctuation = ['!', ',', '.', ':', ';', '?']
    # punctuation, and ascii allowed (it's lowercase anyway due to preceding prep)
    allowed = ''.join(punctuation)+string.ascii_lowercase
    return ''.join(c for c in text if c in allowed)
    
    # This would work for non-ascii letters as well 
    # return ''.join(c for c in text if c.isalpha() or c in punctuation)

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs  = []
    outputs = []

    for i in range(0, len(text), step_size):
    
        # current seq window 
        seq_from         = i 
        seq_to_exclusive = seq_from + window_size
        
        # make sure that there is an output left 
        # in case of len(series) % window_size == 0
        
        if seq_to_exclusive < len(text):
            inputs.append(text[seq_from:seq_to_exclusive])
            outputs.append(text[seq_to_exclusive])
            
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation("softmax"))
    
    return model
