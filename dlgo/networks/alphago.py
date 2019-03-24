from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D

def alphago_model(input_shape, is_policy_net=False,
                num_filters=192, first_kernel_size=5, other_kernel_size=3):

    model = Sequential()
    model.add(
        Conv2D(num_filters, first_kernel_size, input_shape=input_shape, padding='same',
            data_format='channels_first', activation='relu'))

    for i in range(2, 12):
        model.add(
            Conv2D(num_filters, other_kernel_size, padding='same',
                data_format='channels_first', activation='relu'))

    if is_policy_net:
        model.add(
            Conv2D(filters=1, kernel_size=1, padding='same',
                data_format='channels_first', activation='softmax'))
        model.add(Flatten())
        return model
    else:
        model.add(
            Conv2D(num_filters, other_kernel_size, padding='same',
                data_format='channels_first', activation='relu'))
        model.add(
            Conv2D(filters=1, kernel_size=1, padding='same',
                data_format='channels_first', activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='tanh'))
        return model
