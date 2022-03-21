


import numpy as np
from .degree_freedom_queen import *
from .degree_freedom_king1 import *
from .degree_freedom_king2 import *
from .generate_game import *


class Chess_Env:
    
    def __init__(self,N_grid : int, **kwargs):

        self.flag_state_check = kwargs.get("ce_state_check", False)
        self.flag_state_extra = kwargs.get("ce_state_extra", False)
        self.flag_reward_check = kwargs.get("ce_reward_check", False)
        self.flag_reward_draw = kwargs.get("ce_reward_draw", False)

        self.reward_ongoing = 0
        self.reward_draw = 0
        self.reward_win = 1
        self.reward_check = kwargs.get("ce_reward_check_value", 0.5)

        if self.flag_reward_draw:
            self.reward_draw = -1


        self.N_grid=N_grid                     # SIZE OF THE BOARD
        
        self.Board=np.zeros([N_grid,N_grid])   # THE BOARD, THIS WILL BE FILLED BY 0 (NO PIECE), 1 (AGENT'S KING), 2 (AGENT'S QUEEN), 3 (OPPONENT'S KING)
        
        self.p_k1=np.zeros([2,1])              # POSITION OF THE AGENT'S KING AS COORDINATES
        self.p_k2=np.zeros([2,1])              # POSITION OF THE OPPOENT'S KING AS COORDINATES
        self.p_q1=np.zeros([2,1])              # POSITION OF THE AGENT'S QUEEN AS COORDINATES
        
        self.dfk1=np.zeros([N_grid,N_grid])    # ALL POSSIBLE ACTIONS FOR THE AGENT'S KING (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES)
        self.dfk2=np.zeros([N_grid,N_grid])    # ALL POSSIBLE ACTIONS FOR THE OPPONENT'S KING (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES)
        self.dfq1=np.zeros([N_grid,N_grid])    # ALL POSSIBLE ACTIONS FOR THE AGENT'S QUEEN (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES)
        
        self.dfk1_constrain=np.zeros([N_grid,N_grid])  # ALLOWED ACTIONS FOR THE AGENT'S KING CONSIDERING ALSO THE OTHER PIECES
        self.dfk2_constrain=np.zeros([N_grid,N_grid])  # ALLOWED ACTIONS FOT THE OPPONENT'S KING CONSIDERING ALSO THE OTHER PIECES
        self.dfq1_constrain=np.zeros([N_grid,N_grid])  # ALLOWED ACTIONS FOT THE AGENT'S QUEEN CONSIDERING ALSO THE OTHER PIECES
        
        self.ak1=np.zeros([8])                         # ALLOWED ACTIONS OF THE AGENT'S KING (CONSIDERING OTHER PIECES), ONE-HOT ENCODED
        self.possible_king_a=np.shape(self.ak1)[0]     # TOTAL NUMBER OF POSSIBLE ACTIONS FOR AGENT'S KING
        
        self.aq1=np.zeros([8*(self.N_grid-1)])         # ALLOWED ACTIONS OF THE AGENT'S QUEEN (CONSIDERING OTHER PIECES), ONE-HOT ENCODED
        self.possible_queen_a=np.shape(self.aq1)[0]     # TOTAL NUMBER OF POSSIBLE ACTIONS FOR AGENT'S QUEEN
        
        self.check=0                                   # 1 (0) IF ENEMY KING (NOT) IN CHECK
        
        # THIS MAP IS USEFUL FOR US TO UNDERSTAND THE DIRECTION OF MOVEMENT GIVEN THE ACTION MADE (SKIP...)
        self.map=np.array([[1, 0],
                            [-1, 0],
                            [0, 1],
                            [0, -1],
                            [1, 1],
                            [1, -1],
                            [-1, 1],
                            [-1, -1]])

        
        
    def Initialise_game(self):
        
        
        # START THE GAME BY SETTING PIECIES
        
        self.Board,self.p_k2,self.p_k1,self.p_q1=generate_game(self.N_grid)
       
        # Allowed actions for the agent's king
        self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(self.p_k1, self.p_k2, self.p_q1, self.Board)
        
        # Allowed actions for the agent's queen
        self.dfq1_constrain, self.a_q1, self.dfq1  = degree_freedom_queen(self.p_k1, self.p_k2, self.p_q1, self.Board)
        
        # Allowed actions for the enemy's king
        self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(self.dfk1, self.p_k2, self.dfq1, self.Board, self.p_k1)
        
        # ALLOWED ACTIONS FOR THE AGENT, ONE-HOT ENCODED
        allowed_a=np.concatenate([self.a_q1,self.a_k1],0)
        
        # FEATURES (INPUT TO NN) AT THIS POSITION
        X=self.Features()

        
        
        return self.Board, X, allowed_a
        
    
    def OneStep(self,a_agent):
        
        # SET REWARD TO ZERO IF GAME IS NOT ENDED
        R=self.reward_ongoing
        # SET Done TO ZERO (GAME NOT ENDED)
        Done=0
        
        
        # PERFORM THE AGENT'S ACTION ON THE CHESS BOARD
        
        if a_agent < self.possible_queen_a:    # THE AGENT MOVED ITS QUEEN 
           
           # UPDATE QUEEN'S POSITION
           direction = int(np.ceil((a_agent + 1) / (self.N_grid - 1))) - 1
           steps = a_agent - direction * (self.N_grid - 1) + 1

           self.Board[self.p_q1[0], self.p_q1[1]] = 0
           
           mov = self.map[direction, :] * steps
           self.Board[self.p_q1[0] + mov[0], self.p_q1[1] + mov[1]] = 2
           self.p_q1[0] = self.p_q1[0] + mov[0]
           self.p_q1[1] = self.p_q1[1] + mov[1]

        else:                                 # THE AGENT MOVED ITS KING                               
           
           # UPDATE KING'S POSITION
           direction = a_agent - self.possible_queen_a
           steps = 1

           self.Board[self.p_k1[0], self.p_k1[1]] = 0
           mov = self.map[direction, :] * steps
           self.Board[self.p_k1[0] + mov[0], self.p_k1[1] + mov[1]] = 1
           self.p_k1[0] = self.p_k1[0] + mov[0]
           self.p_k1[1] = self.p_k1[1] + mov[1]

        
        # COMPUTE THE ALLOWED ACTIONS AFTER AGENT'S ACTION
        # Allowed actions for the agent's king
        self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(self.p_k1, self.p_k2, self.p_q1, self.Board)
        
        # Allowed actions for the agent's queen
        self.dfq1_constrain, self.a_q1, self.dfq1  = degree_freedom_queen(self.p_k1, self.p_k2, self.p_q1, self.Board)
        
        # Allowed actions for the enemy's king
        self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(self.dfk1, self.p_k2, self.dfq1, self.Board, self.p_k1)

        
        # CHECK IF POSITION IS A CHECMATE, DRAW, OR THE GAME CONTINUES
        
        # CASE OF CHECKMATE
        if np.sum(self.dfk2_constrain) == 0 and self.dfq1[self.p_k2[0], self.p_k2[1]] == 1:
           
            # King 2 has no freedom and it is checked
            # Checkmate and collect reward
            Done = 1       # The epsiode ends
            R = self.reward_win          # Reward for checkmate
            allowed_a=[]   # Allowed_a set to nothing (end of the episode)
            X=[]           # Features set to nothing (end of the episode)
        
        # CASE OF DRAW
        elif np.sum(self.dfk2_constrain) == 0 and self.dfq1[self.p_k2[0], self.p_k2[1]] == 0:
           
            # King 2 has no freedom but it is not checked
            Done = 1        # The epsiode ends
            R = self.reward_draw       # Reward for draw
            allowed_a=[]    # Allowed_a set to nothing (end of the episode)
            X=[]            # Features set to nothing (end of the episode)
        
        # THE GAME CONTINUES
        else:

            if self.flag_reward_check:
                if self.check == 1:
                    R = self.reward_check
            
            # THE OPPONENT MOVES THE KING IN A RANDOM SAFE LOCATION
            allowed_enemy_a = np.where(self.a_k2 > 0)[0]
            a_help = int(np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
            a_enemy = allowed_enemy_a[a_help]

            direction = a_enemy
            steps = 1

            self.Board[self.p_k2[0], self.p_k2[1]] = 0
            mov = self.map[direction, :] * steps
            self.Board[self.p_k2[0] + mov[0], self.p_k2[1] + mov[1]] = 3

            self.p_k2[0] = self.p_k2[0] + mov[0]
            self.p_k2[1] = self.p_k2[1] + mov[1]
            
            
            
            # COMPUTE THE ALLOWED ACTIONS AFTER THE OPPONENT'S ACTION
            # Possible actions of the King
            self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(self.p_k1, self.p_k2, self.p_q1, self.Board)
            
            # Allowed actions for the agent's king
            self.dfq1_constrain, self.a_q1, self.dfq1  = degree_freedom_queen(self.p_k1, self.p_k2, self.p_q1, self.Board)
            
            # Allowed actions for the enemy's king
            self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(self.dfk1, self.p_k2, self.dfq1, self.Board, self.p_k1)

            assert self.check == 0, "cannot move to check"

            # ALLOWED ACTIONS FOR THE AGENT, ONE-HOT ENCODED
            allowed_a=np.concatenate([self.a_q1,self.a_k1],0)
            # FEATURES
            X=self.Features()
            
            
        
        return self.Board, X, allowed_a, R, Done
        
        
    # DEFINITION OF THE FEATURES (SEE ALSO ASSIGNMENT DESCRIPTION)
    def Features(self):
        
        
        s_k1 = np.array(self.Board == 1).astype(float).reshape(-1)   # FEATURES FOR KING POSITION
        s_q1 = np.array(self.Board == 2).astype(float).reshape(-1)   # FEATURES FOR QUEEN POSITION
        s_k2 = np.array(self.Board == 3).astype(float).reshape(-1)   # FEATURE FOR ENEMY'S KING POSITION
        
        check=np.zeros([2])    # CHECK? FEATURE
        check[self.check]=1

        if self.flag_state_check:
            check = np.array([self.check])
        
        K2dof=np.zeros([8])   # NUMBER OF ALLOWED ACTIONS FOR ENEMY'S KING, ONE-HOT ENCODED
        K2dof[np.sum(self.dfk2_constrain).astype(int)]=1

        if self.flag_state_extra:
            ef = self.extra_features()
            x = np.concatenate([s_k1, s_q1, s_k2, check, K2dof, ef], 0)
        else:
            # ALL FEATURES...
            x = np.concatenate([s_k1, s_q1, s_k2, check, K2dof],0)
        
        return x

    def extra_features(self):
        pos = np.where(self.Board==3)
        posx = pos[0][0]
        posy = pos[1][0]

        ie = self.N_grid-1

        # walls
        wall_l = posx == 0
        wall_r = posx == ie
        wall_t = posy == 0
        wall_b = posy == ie

        wall = wall_l or wall_r or wall_b or wall_t

        # corner
        c_1 = wall_l and wall_t
        c_2 = wall_r and wall_t
        c_3 = wall_l and wall_b
        c_4 = wall_r and wall_b

        corner = c_1 or c_2 or c_3 or c_4

        f_walls = [wall, wall_l, wall_r, wall_t, wall_b]
        f_corner = [corner, c_1, c_2, c_3, c_4]


        # support
        pos_ko = np.where(self.Board==1)
        pos_ko_x = pos_ko[0][0]
        pos_ko_y = pos_ko[1][0]

        pos_qo = np.where(self.Board==2)
        pos_qo_x = pos_qo[0][0]
        pos_qo_y = pos_qo[1][0]

        line_a = posx == pos_ko_x and ( abs(posy-pos_ko_y) == 2)
        line_b = posy == pos_ko_y and ( abs(posx-pos_ko_x) == 2)

        line_ours_x = pos_qo_x == pos_ko_x
        line_ours_y = pos_qo_y == pos_ko_y


        support_a = line_a and line_ours_y
        support_b = line_b and line_ours_x


        # middle
        q_middle = (0 < pos_qo_x) and (pos_qo_x < ie) and (0 < pos_qo_y) and (pos_qo_y < ie)

        f_support = [support_a, support_b, line_ours_x, line_ours_y, q_middle]

        extra_features = np.array(f_support).astype(float).reshape(-1)
        return extra_features
        
        


        
        
        
        
        
        
