import gym

from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed


from environment3 import GridWorldEnv
import matplotlib.pyplot as plt


from MooreMachine import MooreMachine
import pygame
from pygame.locals import *

import numpy as np

transition_function = {0:{0:2, 1:0, 2:0, 3:1, 4:0}, 1:{0:1, 1:1, 2:1, 3:1, 4:1}, 2:{0:2, 1:3, 2:2, 3:1, 4:2}, 3:{0:3, 1:3, 2:4, 3:1, 4:3}, 4:{0:4, 1:4, 2:4, 3:1, 4:4}}
output_function = [3,4,2,1,0]

minecraft_machine = MooreMachine(transition_function, output_function)




class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.rewardss = []
        self.arr_mean_rewards = []
        self.episode_rewards = []
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        self.episode_rewards = []
        #self.training_env.reset()
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        #os.system("echo "+str(self.locals))
        #os.system("echo "+str(len(self.training_env)))
        

        self.episode_rewards.append(float(self.locals['rewards'][0]))



        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """


        self.arr_mean_rewards.append(np.mean(self.episode_rewards))


        plt.plot([i for i in range(len(self.arr_mean_rewards))], self.arr_mean_rewards)
        plt.xlabel("episode")
        plt.ylabel("mean of train acc rew")
        plt.savefig("BASELINEtrain_acc_rewards"+".png")
        plt.close()

        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass



import os
os.environ["SDL_VIDEODRIVER"] = "dummy"



env = GridWorldEnv(minecraft_machine, "human", False)

model = PPO('MlpPolicy', env, n_steps=512 , verbose=1)
model.learn(total_timesteps=25000, callback=CustomCallback())
model.save("a2c_minecraft")


exit()

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        
        newenv = GridWorldEnv(minecraft_machine, "human", False)

        env = newenv
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init



if __name__ == "__main__":

    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000, callback=CustomCallback())

    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()



















# del model # remove to demonstrate saving and loading

# model = A2C.load("a2c_minecraft")

# obs = env.reset()



# while True:

#     action, _states = model.predict(obs)

#     obs, rewards, dones, info = env.step(int(action))

#     # print(obs) # [0 1 0 0 0 0]
#     # print(rewards) # -3
#     # print(dones) # False
#     # print(info) # {'robot location': array([0, 1]), 'inventory': 'empty'}

#     #env.render()

#     os.system("echo rewards : "+str(rewards))
