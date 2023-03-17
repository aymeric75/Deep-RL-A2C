import gym
from gym import spaces
import pygame
import random
import numpy as np


#tutta la griglia
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, moore_machine, render_mode = "human", train = True, size = 4):

        self._PICKAXE = "imgs/pickaxe.png"
        self._GEM = "imgs/gem.png"
        self._DOOR = "imgs/door.png"
        self._ROBOT = "imgs/robot.png"
        self._LAVA = "imgs/lava.jpg"
        self._train = train

        self.size = size #4x4 world
        self.window_size = 512 #size of the window
        
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.window = None
        self.clock = None

        self.automaton = moore_machine

        self.action_space = spaces.Discrete(4)
        # 0 = GO_DOWN
        # 1 = GO_RIGHT
        # 2 = GO_UP
        # 3 = GO_LEFT
        self.observation_space = spaces.MultiDiscrete([4,4,2,2,2,2])

        self._action_to_direction = {
            0: np.array([0, 1]), #DOWN
            1: np.array([1, 0]), #RIGHT
            2: np.array([0, -1]), #UP
            3: np.array([-1, 0]), #LEFT
        }

    def reset(self):
        '''
        TUTTO IL RESET 
        '''
        self._agent_location = np.array([0, 0])
        self._gem_location = np.array([0, 3])
        self._pickaxe_location = np.array([3, 2])
        self._exit_location = np.array([3, 0])
        self._lava_location = np.array([3, 3])
        self._state = np.array([0, 0, 0, 0])

        self._has_pickaxe = False
        self._has_gem = False
        self._went_into_lava = False
        self._task_completed = False

        self._gem_display = True
        self._pickaxe_display = True
        self._robot_display = False if self._train else True
        
        if self.render_mode == "human":
            self._render_frame()

        observation = self._agent_location
        observation = np.append(observation, self._state)
        info = self._get_info()
        reward = -3
        return observation

    def _update_conditions(self):
        if self._check_lava():
            return True
        elif (self._agent_location == self._exit_location).all() and self._has_gem and self._has_pickaxe:
            self._task_completed = True
            return True
        else:
            return False
        
    def _check_termination(self):
        if self._went_into_lava and self._task_completed:
            termination = False  
        elif self._went_into_lava:
            termination = False
        elif self._task_completed:
            termination = True 
        else:
            termination = False
        return termination
    
    def _check_pickaxe(self):
        if not self._has_pickaxe:
            if (self._agent_location == self._pickaxe_location).all():
                self._has_pickaxe = True
                # self._pickaxe_display = False

    def _check_gem(self):
        if not self._has_gem:
            if (self._agent_location == self._gem_location).all(): #and self._has_pickaxe:
                self._has_gem = True
                # self._gem_display = False

    def _check_lava(self):
        if (self._agent_location == self._lava_location).all():
            self._went_into_lava = True
            return True
        else:
            return False

    def step(self, action):
        '''
        TUTTI GLI STEP
        '''
        reward = -3

        #MOVEMENT
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location+direction, 0, self.size-1)
        observation = self._agent_location

        if self._check_lava():
            reward = -4

        if self._went_into_lava and self._task_completed:
            reward = -4
            if self.render_mode == "human":
                self._render_frame()    
        elif self._went_into_lava:
            reward = -4
            if self.render_mode == "human":
                self._render_frame()
        elif self._task_completed:
            reward = 0
            if self._check_lava():
                reward = -4
            if self.render_mode == "human":
                self._render_frame()
        else:
            self._check_pickaxe()
            self._check_gem()
            if self._has_pickaxe and self._has_gem:
                reward = -1
            elif self._has_pickaxe or self._has_gem:
                reward = -2

            if self.render_mode == "human":
                self._render_frame()

        _ = self._update_conditions()
        termination = self._check_termination()

        if self._task_completed and not self._went_into_lava:
            reward = 0

        if self._went_into_lava:
            self._state[3]= 1
        else:
            if self._has_pickaxe:
                self._state[0] = 1
            if self._has_gem:
                self._state[1] = 1
            if self._task_completed:
                self._state[2] = 1

        observation = np.append(observation, self._state)

        info = self._get_info()
        return observation, reward, termination, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _get_obs(self):
        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1,0,2)
            )
        img = img[:,:,::-1]
        obs = img
        return obs

    def _get_info(self):
        info = {
            "robot location": self._agent_location,
            "inventory": "empty"
        }
        if self._has_gem:
            info["inventory"] = "gem"
        elif self._has_pickaxe: 
            info["inventory"] = "pickaxe"
        else:
            info["inventory"] = "empty"
        return info

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size,self.window_size))
        canvas.fill((255,255,255))

        pix_square_size = (self.window_size/self.size)

        for x in range(self.size+1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
        
        if self.render_mode == "human":
            pickaxe = pygame.image.load(self._PICKAXE)
            gem = pygame.image.load(self._GEM)
            door = pygame.image.load(self._DOOR)
            robot = pygame.image.load(self._ROBOT)
            lava = pygame.image.load(self._LAVA)
            self.window.blit(canvas, canvas.get_rect())

            if self._robot_display:
                self.window.blit(robot, (pix_square_size*self._agent_location[0],pix_square_size*self._agent_location[1]))
            if self._pickaxe_display:
                self.window.blit(pickaxe, (pix_square_size*self._pickaxe_location[0], pix_square_size*self._pickaxe_location[1]))
            if self._gem_display:
                self.window.blit(gem, (pix_square_size*self._gem_location[0],32+pix_square_size*self._gem_location[1]))
            self.window.blit(door, (pix_square_size*self._exit_location[0], pix_square_size*self._exit_location[1]))
            self.window.blit(lava, (pix_square_size*self._lava_location[0] + 2, pix_square_size*self._lava_location[1] + 2))
                
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
