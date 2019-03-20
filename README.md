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

Run `python self_play.py --board-size=19 --learning-agent=./agents/your_bot.h5 --experience-out ./experiences/experience.h5` to let a bot play against itself and store experiences gathered during self play.

Run `python train_pg.py --learning-agent=./agents/deep_bot.h5 --agent-out=./agents/deep_bot_improved.h5 ./experiences/experience.h5` to use experience data for agent improvements via Deep Reinforcement Learning.

Run `python eval_pg_bot.py --agent1=./agents/deep_bot_improved.h5 --agent2=./agents/deep_bot.h5` to check whether the new bot is stronger.

## Resources

- [Book - Deep Learning and the Game of Go](https://www.manning.com/books/deep-learning-and-the-game-of-go)
- [Paper - Mastering the game of Go with deep neural networks and tree search](http://web.iitd.ac.in/~sumeet/Silver16.pdf)
- [Paper - Mastering the Game of Go without Human Knowledge](https://deepmind.com/documents/119/agz_unformatted_nature.pdf)
- [Video - Mastering Games without Human Knowledge](https://www.youtube.com/watch?v=Wujy7OzvdJk)
