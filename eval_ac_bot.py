import argparse
import h5py
from collections import namedtuple

from dlgo import rl
from dlgo import scoring
from dlgo.goboard_fast import GameState, Player

BOARD_SIZE = 19

class GameRecord(namedtuple('GameRecord', 'moves winner')):
    pass

def simulate_game(black_player, white_player):
    moves = []
    game = GameState.new_game(BOARD_SIZE)
    agents = {
        Player.black: black_player,
        Player.white: white_player
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)
    game_result = scoring.compute_game_result(game)
    print(game_result)
    return GameRecord(
        moves=moves,
        winner=game_result.winner
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent1', required=True)
    parser.add_argument('--agent2', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)

    args = parser.parse_args()
    agent1 = rl.load_ac_agent(h5py.File(args.agent1))
    agent2 = rl.load_ac_agent(h5py.File(args.agent2))
    num_games = args.num_games

    wins = 0
    losses = 0
    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player)
        if game_record.winner == color1:
            wins += 1
        else:
            losses += 1
        color1 = color1.other
    print('Agent 1 record: %d/%d' % (wins, wins + losses))

if __name__ == '__main__':
    main()
