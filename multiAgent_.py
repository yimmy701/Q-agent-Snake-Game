import os
import pygame
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DQN import DQNAgent
from random import randint
import random
import statistics
import torch.optim as optim
import torch 
import datetime
import distutils.util
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'


def define_parameters():
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1/100
    params['learning_rate'] = 0.00013629
    params['first_layer_size'] = 200   
    params['second_layer_size'] = 20   
    params['third_layer_size'] = 50    
    params['episodes'] = 250          
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.h5'
    params['train'] = True
    params["test"] = False
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
    return params


class Game:
    ## initialize PyGAME 
    
    def __init__(self, game_width, game_height):
        pygame.display.set_caption('SnakeGame')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height + 60))
        self.bg = pygame.image.load("img/background23.png")
        self.crash = False
        self.player = QSnake(self)
        
        self.food = Food()
        
        self.score = 0


class QSnake(object):
    def __init__(self, game):
        x = 0.5 * game.game_width
        y = 0.5 * game.game_height
        self.x = x
        self.y = y 
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.image = pygame.image.load('img/snakeBody1.png')
        self.x_change = 20
        self.y_change = 0

    def update_position(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def do_move(self, move, x, y, game, food, agent):
        move_array = [self.x_change, self.y_change]

        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]
        elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        if self.x < 0 or self.x > game.game_width - 20  \
                or self.y < 0 \
                or self.y > game.game_height - 20 \
                or [self.x, self.y] in self.position:
            game.crash = True
        eat(self, food, game)

        self.update_position(self.x, self.y)

    def display_player(self, x, y, food, game):
        self.position[-1][0] = x
        self.position[-1][1] = y

        if game.crash == False:
            for i in range(food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))

            update_screen()
        else:
            pygame.time.wait(300)


class Food(object):
    def __init__(self):
        ## if use manual / random / appleDirecting player, set self.x_food = 220, self.y_food = 200
        # self.x_food = 220
        # self.y_food = 200
        self.x_food = 230
        self.y_food = 210
        self.image = pygame.image.load('img/food1.png')
        self.color = (223, 163, 49)
       
        
    def randomize_position(self):
        (self.x_food,self.y_food) = (random.randint(0, grid_width - 1)*gridsize, random.randint(0, grid_height - 1)*gridsize)


    def food_coord(self, game, player):
        x_rand = randint(0, 20 - 1)*gridsize
        self.x_food = x_rand - 10
        y_rand = randint(0, 20 - 1)*gridsize
        self.y_food = y_rand - 10
        if [self.x_food, self.y_food] not in player.position:
            return self.x_food, self.y_food
        else:
            self.food_coord(game, player)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen()
    
    def draw(self, surface):
        r = pygame.Rect((self.x_food, self.y_food), (gridsize, gridsize))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, (93, 216, 228), r, 1)

up = (0,-1)
down = (0,1)
left = (-1,0)
right = (1,0)
gridsize = 20
screen_width = 480
screen_height = 480
grid_width = screen_width/gridsize
grid_height = screen_height/gridsize

class ManualPlayer():
    def __init__(self):
        self.length = 1
        # a list, with first element placing snake initially at the center
        self.positions = [(screen_width/2, screen_height/2)] 
        self.direction = random.choice([up, down, left, right])
        self.color = (17, 24, 47)
        self.score = 0

    def get_head_position(self):
        return self.positions[0] #first element in the list

    def turn(self, point):
        if self.length > 1 and (point[0]*-1, point[1]*-1) == self.direction:
            return
        else:
            self.direction = point

    def move(self):
        cur = self.get_head_position()
        x,y = self.direction
        new = (((cur[0]+(x*gridsize))%screen_width), (cur[1]+(y*gridsize))%screen_height)
        if len(self.positions) > 2 and new in self.positions[2:]:
            self.reset()
        
        else:
            self.positions.insert(0,new)
            if len(self.positions) > self.length:
                self.positions.pop() #removes the first item

    def reset(self):
        self.length = 1
        self.positions = [((screen_width/2), (screen_height/2))]
        self.direction = random.choice([up, down, left, right])
        self.score = 0

    def draw(self,surface):
        for p in self.positions:
            r = pygame.Rect((p[0], p[1]), (gridsize,gridsize))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, (93,216, 228), r, 1)

    def takeAction(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.turn(up)
                elif event.key == pygame.K_DOWN:
                    self.turn(down)
                elif event.key == pygame.K_LEFT:
                    self.turn(left)
                elif event.key == pygame.K_RIGHT:
                    self.turn(right)
                    
                    
## Note: There is a bug that hasn't been fixed that causes RandomSnake and AppleDirectingSnake to ONLY move if curser was constantly moving on the game interface.
class RandomSnake(ManualPlayer):
    def takeAction(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            else:
                action = random.choice([up,down,left,right])
                self.turn(action) #pretty slow and dumb but it does move!
                
class AppleDirectingSnake(ManualPlayer):
    def __init__(self):
        self.length = 1
        # a list, with first element placing snake initially at the center
        self.positions = [(screen_width/2, screen_height/2)] 
        self.direction = random.choice([up, down, left, right])
        self.color = (17, 24, 47)
        self.score = 0
        
    def setDirection(self,apple):
        self.direction = (apple.x_food,apple.y_food) #directed by apple's position
    def takeAction(self,apple):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            else:
                #apple on the left side of snake
                if apple.x_food < self.get_head_position()[0]: 
                    self.turn(left)
                    if apple.x_food==self.get_head_position()[0]:
                        #apple above snake   
                        if apple.y_food < self.get_head_position()[1]: 
                            self.turn(up)
                        #apple on the same row as snake
                        if apple.y_food == self.get_head_position()[1]: 
                            self.turn(left)
                        #apple below snake
                        if apple.y_food > self.get_head_position()[1]: 
                            self.turn(down)
                        
                #apple on the right side of snake
                if apple.x_food > self.get_head_position()[0]: 
                    self.turn(right)
                    if apple.x_food==self.get_head_position()[0]:
                        #apple above snake   
                        if apple.y_food < self.get_head_position()[1]: 
                            self.turn(up)
                        #apple on the same row as snake
                        if apple.y_food == self.get_head_position()[1]: 
                            self.turn(right)
                        #apple below snake
                        if apple.y_food > self.get_head_position()[1]: 
                            self.turn(down)
                
                #apple on the same column as snake
                if apple.x_food == self.get_head_position()[0]: 
                    #apple above snake   
                    if apple.y_food < self.get_head_position()[1]: 
                        self.turn(up)
                    #apple below snake
                    if apple.y_food > self.get_head_position()[1]: 
                        self.turn(down)
                        
                        
def eat(player, food, game):
    if player.x == food.x_food and player.y == food.y_food:
        food.food_coord(game, player)
        player.eaten = True
        game.score += 1

def get_record(score, record):
    if score >= record:
        return score
    else:
        return record


def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE:   ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 500))
    game.gameDisplay.blit(text_score_number, (130, 500))
    game.gameDisplay.blit(text_highest, (190, 500))
    game.gameDisplay.blit(text_highest_number, (380, 500))
    game.gameDisplay.blit(game.bg, (10, 10))


def display(player, food, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    player.display_player(player.position[-1][0], player.position[-1][1], player.food, game)
    food.display_food(food.x_food, food.y_food, game)


def update_screen():
    pygame.display.update()


def initialize_game(player, game, food1, agent, batch_size):
    state_init1 = agent.get_state(game, player, food1)# [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food1, agent)
    
    state_init2 = agent.get_state(game, player, food1)
   
    reward1 = agent.set_reward(player, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory, batch_size)


def plot_seaborn(array_counter, array_score, train):
    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13,8))
    fit_reg = False if train== False else True        
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        #color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg = fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [np.mean(array_score)]*len(array_counter)
    ax.plot(array_counter,y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='# games', ylabel='score')
    plt.show()


def get_mean_stdev(array):
    return statistics.mean(array), statistics.stdev(array)    


def test(params):
    params['load_weights'] = True
    params['train'] = False
    params["test"] = False 
    score, mean, stdev = runQ(params)
    return score, mean, stdev

def drawGrid(surface):
    
    for y in range(0, int(screen_height/gridsize)):
        for x in range(0, int(screen_width/gridsize)):
            if (x+y)%2 == 0:
                r = pygame.Rect((x*gridsize, y*gridsize), (gridsize,gridsize))
                pygame.draw.rect(surface,(93,216,228), r)
            else:
                rr = pygame.Rect((x*gridsize, y*gridsize), (gridsize,gridsize))
                pygame.draw.rect(surface, (84,194,205), rr)

def runQ(params):
    
    pygame.init()
    agent = DQNAgent(params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    total_score = 0
    while counter_games < params['episodes']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Initialize classes
        game = Game(500, 500)
        player1 = game.player
        food1 = game.food

        # Perform first move
        initialize_game(player1, game, food1, agent, params['batch_size'])
        if params['display']:
            display(player1, food1, game, record)
        
        steps = 0       # steps since the last positive reward
        while (not game.crash) and (steps < 100):
            if not params['train']:
                agent.epsilon = 0.01
            else:
                # agent.epsilon is set to give randomness to actions
                agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

            # get old state
            state_old = agent.get_state(game, player1, food1)

            # perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < agent.epsilon:
                final_move = np.eye(3)[randint(0,2)]
            else:
                # predict action based on the old state
                with torch.no_grad():
                    state_old_tensor = torch.tensor(state_old.reshape((1, 11)), dtype=torch.float32).to(DEVICE)
                    prediction = agent(state_old_tensor)
                    final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]

            # perform new move and get new state
            player1.do_move(final_move, player1.x, player1.y, game, food1, agent)
            state_new = agent.get_state(game, player1, food1)

            # set reward for the new state
            reward = agent.set_reward(player1, game.crash)
            
            # if food is eaten, steps is set to 0
            if reward > 0:
                steps = 0
                
            if params['train']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new, game.crash)

            record = get_record(game.score, record)
            if params['display']:
                display(player1, food1, game, record)
                pygame.time.wait(params['speed'])
            steps+=1
        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])
        counter_games += 1
        total_score += game.score
        print(f'Game {counter_games}      Score: {game.score}')
        score_plot.append(game.score)
        counter_plot.append(counter_games)
    mean, stdev = get_mean_stdev(score_plot)
    if params['train']:
        model_weights = agent.state_dict()
        torch.save(model_weights, params["weights_path"])
    if params['plot_score']:
        plot_seaborn(counter_plot, score_plot, params['train'])
    return total_score, mean, stdev

def runSimple():
    pygame.init()

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((screen_width, screen_height), 0, 32)

    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    drawGrid(surface)

    snake = ManualPlayer()
    apple = Food()
    # snake = AppleDirectingSnake()
    # snake = RandomSnake()
    
  
    myfont = pygame.font.SysFont("monospace",16)

    while (True):
        clock.tick(10)

        # snake.takeAction(apple)
        snake.takeAction()
        
        drawGrid(surface)
        snake.move()
        
        if snake.get_head_position() == (apple.x_food, apple.y_food):
            snake.length += 1
            snake.score += 1
            apple.randomize_position()
               
        snake.draw(surface)
       
        apple.draw(surface)
        screen.blit(surface, (0,0))
        text = myfont.render("Score {0}".format(snake.score), 1, (0,0,0))
        screen.blit(text, (5,10))
        pygame.display.update()
        
        
#if run agents other than Q-learning Player, comment line 514-535 and uncomment line 536.
if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=False)
    parser.add_argument("--speed", nargs='?', type=int, default=50)
    parser.add_argument("--bayesianopt", nargs='?', type=distutils.util.strtobool, default=False)
    args = parser.parse_args()
    print("Args", args)
    params['display'] = args.display
    params['speed'] = args.speed
    
    if params['train']:
        print("Training...")
        params['load_weights'] = False   # when training, the network is not pre-trained
        runQ(params)
    if params['test']:
        print("Testing...")
        params['train'] = False
        params['load_weights'] = True
        runQ(params)
    # runSimple() 
