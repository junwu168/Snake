import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        
        # Two tuples (self.s, self.a): previous states
        # s_prime: current state
        # act: to generate a_prime for s_prime
        # In training: update the Q and N tables
        # In both (train & test): give a_prime
        #########
        # Update: update for the previous states
        #   Init case:  no need to update
        #   Other cases: 
        #       N table: self.N[self.s][self.a] += 1
        #       Q table: self.Q[self.s][self.a] = 
        # reward:
        # max_a Q(s_(t+1), a)
        # Start from current state: s_prime
        # go throught Q[s_prime]
        # find the action giving you max Q
        # get the max Q
        ########
        # give a_prime
        # train and test
        # train: do the exploration: 
        # for a in actions:
        #     if N[s_prime][a] < Ne:
        #        q_val = 1
        #     else:
        #        q_val = Q
            # if q_val >= val:
            #     val = q_val
            #     a_prime = a
        # action with max q_val will be your q_prime
        # for test: only Q table 

        # self.s = s_prime
        # self.a = a_prime
        # self.points = points

        s_prime = self.generate_state(environment)

        if((self._train != False) and (self.s != None)):
            max_q = float("-inf")
            reward = self.getReward(points, dead)
            self.N[self.s][self.a] += 1

            for a in self.actions:
                if(self.Q[s_prime][a] >= max_q):
                    max_q = self.Q[s_prime][a]
            
            alpha = self.C / (self.C + self.N[self.s][self.a])
            self.Q[self.s][self.a] = self.Q[self.s][self.a] + alpha * (reward+self.gamma * max_q - self.Q[self.s][self.a])
        
        if (not dead):
            self.s = s_prime
            self.points = points
        
        else:
            self.reset()

            return 0
        
        max_act = float("-inf")

        a_prime = utils.UP

        
        for a in self.actions:
            val  = 0
            if(self._train and self.N[self.s][a] < self.Ne):
                val = 1
            else:
                val = self.Q[self.s][a]
            
            if val >= max_act:
                a_prime = a
                max_act = val

        self.a = a_prime
        return a_prime

    def getReward(self, points, dead):
        if(dead):
            return -1
        elif(points > self.points):
            return 1
        else:
            return -0.1



    def generate_state(self, environment):


        food_dir_x, food_dir_y = self.checkFoodDir(environment[0], environment[1], environment[3], environment[4])
        adjoining_wall_x, adjoining_wall_y = self.checkWallDir(environment[0], environment[1])
        adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = self.checkSnakeBody(environment[0], environment[1], environment[2])
        
        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)

    
    def checkFoodDir(self,snake_head_x, snake_head_y,food_x, food_y):
        
        if (food_x < snake_head_x):
            food_dir_x = 1
        elif(food_x > snake_head_x):
            food_dir_x = 2
        else:
            food_dir_x = 0

        if(food_y < snake_head_y):
            food_dir_y = 1
        elif(food_y > snake_head_y):
            food_dir_y = 2
        else:
            food_dir_y = 0

        return food_dir_x, food_dir_y

    def checkWallDir(self, snake_head_x, snake_head_y):
        
        if(snake_head_x + 1 == utils.DISPLAY_WIDTH-1):
            adjoining_wall_x = 2
        elif(snake_head_x == 1):
            adjoining_wall_x = 1
        else:
            adjoining_wall_x = 0
        
        if(snake_head_y + 1 == utils.DISPLAY_HEIGHT-1):
            adjoining_wall_y = 2
        elif(snake_head_y == 1):
            adjoining_wall_y = 1
        else:
            adjoining_wall_y = 0
        
        return adjoining_wall_x, adjoining_wall_y
    
    def checkSnakeBody(self, snake_head_x, snake_head_y, body):

        adjoining_body_right = int((snake_head_x + 1, snake_head_y) in body)
        adjoining_body_left = int((snake_head_x - 1, snake_head_y) in body)

        adjoining_body_top = int((snake_head_x, snake_head_y - 1) in body)
        adjoining_body_bottom = int((snake_head_x, snake_head_y + 1) in body)

        return adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right
    
