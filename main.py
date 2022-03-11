# add the provided files to system path
import sys
sys.path.append('./src')
from src.Chess_env import Chess_Env

import numpy as np
import torch
import torch.nn as nn
import argparse


class ChessTrainer:
    def __init__(self, opt):
        self.opt = opt

        # initialize environment
        self.env = Chess_Env(opt.size_board)
        S, X, allowed_a = self.env.Initialise_game()
        N_a = np.shape(allowed_a)[0]
        N_in = np.shape(X)[0]
        N_h = opt.N_h

        # initialize Q network
        self.Q = nn.Sequential(
            nn.Linear(N_in, N_h),
            nn.ReLU(inplace=True),
            nn.Linear(N_h, N_a),
        )

    def q_learning(self):







        import pdb; pdb.set_trace()


    def sarsa(self):
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["q-learning", "sarsa"])
    parser.add_argument("--size_board", type=int, default=4,
                        help="the size of the chessboard")
    parser.add_argument("--N_h", type=int, default=200,
                        help="number of hidden nodes")
    parser.add_argument("--N_episodes", type=int, default=1000,
                        help="number of games to be played")
    opt = parser.parse_args()

    # initialize training session
    trainer = ChessTrainer(opt)

    # run the algorithm
    if opt.mode == "q-learning":
        trainer.q_learning()
    elif opt.mode == "sarsa":
        raise NotImplementedError
    else:
        raise NotImplementedError
