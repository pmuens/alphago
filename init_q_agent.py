import argparse
import h5py

from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.layers import ZeroPadding2D, concatenate

from dlgo import rl
from dlgo import encoders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--output-file', required=True)
    args = parser.parse_args()

    board_size = args.board_size
    output_file = args.output_file

    encoder = encoders.get_encoder_by_name('simple', board_size)

    board_input = Input(shape=encoder.shape(), name='board_input')
    action_input = Input(shape=(encoder.num_points(),), name='action_input')

    conv1a = ZeroPadding2D((2, 2))(board_input)
    conv1b = Conv2D(64, (5, 5), activation='relu')(conv1a)

    conv2a = ZeroPadding2D((1, 1))(conv1b)
    conv2b = Conv2D(64, (3, 3), activation='relu')(conv2a)

    flat = Flatten()(conv2b)
    processed_board = Dense(512)(flat)

    board_and_action = concatenate([action_input, processed_board])
    hidden_layer = Dense(256, activation='relu')(board_and_action)
    value_output = Dense(1, activation='tanh')(hidden_layer)

    model = Model(inputs=[board_input, action_input], outputs=value_output)

    new_agent = rl.QAgent(model, encoder)
    with h5py.File(output_file, 'w') as outf:
        new_agent.serialize(outf)

if __name__ == '__main__':
    main()
