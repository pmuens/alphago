from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Input
from keras.models import Model
from dlgo.goboard_fast import GameState, Player
from dlgo import scoring
from dlgo import zero

def simulate_game(
        board_size,
        black_agent, black_collector,
        white_agent, white_collector):
    print('Starting the game!')
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    black_collector.begin_episode()
    white_collector.begin_episode()
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    print(game_result)
    if game_result.winner == Player.black:
        black_collector.complete_episode(1)
        white_collector.complete_episode(-1)
    else:
        black_collector.complete_episode(-1)
        white_collector.complete_episode(1)

def main():
    board_size = 9
    encoder = zero.ZeroEncoder(board_size)

    board_input = Input(shape=encoder.shape(), name='board_input')
    pb = board_input
    for i in range(4):
        pb = Conv2D(64, (3, 3),
            padding='same', data_format='channels_first', activation='relu')(pb)
        pb = BatchNormalization(axis=1)(pb)
        pb = Activation('relu')(pb)

    policy_conv = Conv2D(2, (1, 1),
        data_format='channels_first', activation='relu')(pb)
    policy_batch = BatchNormalization(axis=1)(policy_conv)
    policy_flat = Flatten()(policy_batch)
    policy_output = Dense(encoder.num_moves(), activation='softmax')(policy_flat)

    value_conv = Conv2D(1, (1, 1),
        data_format='channels_first', activation='relu')(pb)
    value_batch = BatchNormalization(axis=1)(value_conv)
    value_flat = Flatten()(value_batch)
    value_hidden = Dense(256, activation='relu')(value_flat)
    value_output = Dense(1, activation='tanh')(value_hidden)

    model = Model(inputs=[board_input], outputs=[policy_output, value_output])
    black_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
    white_agent = zerp.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
    c1 = zero.ZeroExperienceCollector()
    c2 = zero.ZeroExperienceCollector()
    black_agent.set_collector(c1)
    white_agent.set_collector(c2)

    for i in range(5):
        simulate_game(board_size, black_agent, c1, white_agent, c2)

    exp = zero.combine_experience([c1, c2])
    black_agent.train(exp, 0.01, 2048)

if __name__ == '__main__':
    main()
