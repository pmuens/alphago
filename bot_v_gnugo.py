import h5py

from dlgo.agent.predict import load_prediction_agent
from dlgo.agent.termination import PassWhenOpponentPasses
from dlgo.gtp.play_local import LocalGtpBot

bot = load_prediction_agent(h5py.File('./agents/deep_bot.h5', 'r'))
gtp_bot = LocalGtpBot(go_bot=bot, termination=PassWhenOpponentPasses(),
                    handicap=0, opponent='gnugo')
gtp_bot.run()
