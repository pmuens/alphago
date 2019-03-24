# AlphaGo

This repository contains a reference implementation of the [AlphaGo](https://deepmind.com/research/alphago/) AI by [DeepMind](https://deepmind.com).

## How to play

### Go

#### Bot vs. Bot

Run `python bot_v_bot.py` to let 2 Bots play against each other.

#### Human vs. Bot

Run `python mcts_go.py` to play against a bot.

### Tic-Tac-Toe

#### Human vs. Bot

Run `python play_ttt.py` to play against an [unbeatable](https://en.wikipedia.org/wiki/Minimax) bot.

## Reinforcement Learning

1. Run `python init_ac_agent.py --board-size 9 --output-file ./agents/ac_v1.h5`

1. Run `python self_play_ac.py --board-size 9 --learning-agent ./agents/ac_v1.h5 --num-games 5000 --experience-out ./experiences/exp_0001.h5` to let a bot play against itself and store experiences gathered during self play.

1. Run `python train_ac.py --learning-agent ./agents/ac_v1.h5 --agent-out ./agents/ac_v2.h5 ./--lr 0.01 --bs 1024 experiences/exp_0001.h5` to use experience data for agent improvements via Deep Reinforcement Learning.

1. Run `python eval_ac_bot.py --agent1 ./agents/ac_v2.h5 --agent2 ./agents/ac_v1.h5 --num-games 100` to check whether the new bot is stronger.

If the new agent is stronger start with it at 2.

Otherwise go to 2. again to generate more training data. Use multiple experience data files in 3.

Rinse and repeat.

## Resources

- [Book - Deep Learning and the Game of Go](https://www.manning.com/books/deep-learning-and-the-game-of-go)
- [Paper - Mastering the game of Go with deep neural networks and tree search](http://web.iitd.ac.in/~sumeet/Silver16.pdf)
- [Paper - Mastering the Game of Go without Human Knowledge](https://deepmind.com/documents/119/agz_unformatted_nature.pdf)
- [Video - Mastering Games without Human Knowledge](https://www.youtube.com/watch?v=Wujy7OzvdJk)
