"""
    Utility functions
"""
import gym
import controllers.fetchEnvs.slideEnv as slideEnv
import controllers.fetchEnvs.pushEnv as pushEnv
import controllers.fetchEnvs.pickAndPlaceEnv as pickPlaceEnv
import controllers.robosuite.robosuiteNutAssemblyEnv as robosuiteNutAssemblyEnv
import controllers.robosuite.nutAssemblyDenseEnv as nutAssemblyDenseEnv
import requests
from mpi4py import MPI

def connected_to_internet(url:str='http://www.google.com/', timeout:int=5):
    """
        Check if system is connected to the internet
        Used when running code on MIT Supercloud
    """
    try:
        _ = requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("No internet connection available.")
    return False

def print_dash(num_dash:int=50):
    """
        Print dash
    """
    print('_'*num_dash)

def make_env(env_name:str):
    """
        Make an environment and return
        This will check if the environment exists 
        within gym or our custom environment files
    """
    try:
        # check if environment exists in gym
        env = gym.make(env_name)
        print(f'Making Environment: {env_name}')
        return env
    except:
        pass
    try:
        # else import from custom slide environment file
        env = getattr(slideEnv, env_name)()
        print_dash()
        print(f'Making Environment: {env_name}')
        print_dash()
        return env
    except:
        pass
    try:
        env = getattr(pushEnv, env_name)()
        print_dash()
        print(f'Making Environment: {env_name}')
        print_dash()
        return env 
    except:
        pass
    try:
        env = getattr(pickPlaceEnv, env_name)()
        print_dash()
        print(f'Making Environment: {env_name}')
        print_dash()
        return env
    except:
        pass
    try:
        env = getattr(robosuiteNutAssemblyEnv, env_name)()
        print_dash()
        print(f'Making Environment: {env_name}')
        print_dash()
        return env
    except:
        pass
    try:
        env = getattr(nutAssemblyDenseEnv, env_name)()
        print_dash()
        print(f'Making Environment: {env_name}')
        print_dash()
        return env 
    except:
        # only add except in the last try
        print('_'*150)
        print(f'No Environment with the name {env_name} found. Please check the name of the environment or mujoco installation')
        print('_'*150)

def get_pretty_env_name(env_name:str):
    if 'FetchPickAndPlace' in env_name:
        new_env_name = 'pickPlace'
        exp_name = env_name
        exp_name = exp_name.replace('FetchPickAndPlace','')
        return f"{new_env_name}{exp_name}"
    if 'FetchSlide' in env_name:
        new_env_name = 'slide'
        exp_name = env_name
        exp_name = exp_name.replace('FetchSlide','')
        exp_name = exp_name.replace('Control','')
        return f"{new_env_name}{exp_name}"
    if 'FetchPush' in env_name:
        new_env_name = 'push'
        exp_name = env_name
        exp_name = exp_name.replace('FetchPush','')
        return f"{new_env_name}{exp_name}"
    if 'Nut' in env_name and 'Dense' not in env_name:
        return "NutAssembly"
    if 'Nut' in env_name and 'Dense' in env_name:
        return "NutAssemblyDense"
    

if __name__ == "__main__":
    env_names = ['FetchSlide-v1','FetchSlide','FetchPickAndPlacePerfect', 'FetchPushImperfect', 'NutAssembly']
    for name in env_names:
        env = make_env(name)
        # print(f"{name}: {env.reset()}")