import os
import argparse
import random

import gym
import numpy as np

import collections
import pickle
from tqdm import tqdm

import d4rl

def download_d4rl_mujoco_data(env_name, dataset_type, fraction=1.0):
	datasets = []

	data_dir = 'data/'

	print(data_dir)

	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	assert env_name in ['walker2d', 'halfcheetah', 'hopper']
	assert dataset_type in ['medium', 'medium-expert', 'medium-replay', 'random', 'expert']

	if fraction < 1:
		name = f'{env_name}-{dataset_type}-{str(fraction)}-v2'
	elif fraction == 1.0:
		name = f'{env_name}-{dataset_type}-v2'
	else:
		raise RuntimeError("Fraction ratio is larger than 1.0")
	pkl_file_path = os.path.join(data_dir, name)

	print("processing: ", name)

	env = gym.make(f'{env_name}-{dataset_type}-v2')
	dataset = env.get_dataset()

	N = dataset['rewards'].shape[0]
	data_ = collections.defaultdict(list)

	use_timeouts = False
	if 'timeouts' in dataset:
		use_timeouts = True

	episode_step = 0
	paths = []
	for i in range(N):
		done_bool = bool(dataset['terminals'][i])
		if use_timeouts:
			final_timestep = dataset['timeouts'][i]
		else:
			final_timestep = (episode_step == 1000-1)
		for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
			data_[k].append(dataset[k][i])
		if done_bool or final_timestep:
			episode_step = 0
			episode_data = {}
			for k in data_:
				episode_data[k] = np.array(data_[k])
			paths.append(episode_data)
			data_ = collections.defaultdict(list)
		episode_step += 1

	paths = random.sample(paths, int(len(paths) * fraction))
	returns = np.array([np.sum(p['rewards']) for p in paths])
	num_samples = np.sum([p['rewards'].shape[0] for p in paths])
	print(f'Number of samples collected: {num_samples}')
	print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')




	np.save(f'{pkl_file_path}.npy', paths)
	with open(f'{pkl_file_path}.pkl', 'wb') as f:
		pickle.dump(paths, f)





def download_d4rl_adroit_data(env_name):
	datasets = []

	data_dir = 'data/'

	print(data_dir)

	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	assert env_name in ['pen-human-v1', 'door-human-v1', "hammer-human-v1", 'pen-human-v0', 'door-human-v0', "hammer-human-v0"]

	pkl_file_path = os.path.join(data_dir, env_name)

	print("processing: ", env_name)

	env = gym.make(env_name)
	if ("door" in env_name) or ("hammer" in env_name):
		dataset = d4rl.qlearning_dataset(env)
	else:
		dataset = env.get_dataset()

	N = dataset['rewards'].shape[0]
	data_ = collections.defaultdict(list)

	episode_step = 0
	paths = []
			

	if ("door" in env_name) or ("hammer" in env_name):

		for i in range(N):
			done_bool = bool(dataset['terminals'][i])
			final_timestep = (episode_step == len(dataset['rewards'])-1)
			for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
				data_[k].append(dataset[k][i])
			if done_bool or final_timestep:
				episode_step = 0
				episode_data = {}
				for k in data_:
					episode_data[k] = np.array(data_[k])
				paths.append(episode_data)
				data_ = collections.defaultdict(list)
			episode_step += 1
	else:

		for i in range(N-1):
			done_bool = bool(dataset['timeouts'][i])
			final_timestep = (episode_step == len(dataset['rewards'])-1)
			for k in ['observations', 'actions', 'rewards', 'terminals', 'timeouts']:
				data_[k].append(dataset[k][i])
			if done_bool or final_timestep:
				episode_step = 0
				episode_data = {}
				data_['next_observations'] = data_['observations'][1:]
				data_["next_observations"].append(dataset["observations"][i])
				for k in data_:
					episode_data[k] = np.array(data_[k])
				paths.append(episode_data)
				data_ = collections.defaultdict(list)
			episode_step += 1


	returns = np.array([np.sum(p['rewards']) for p in paths])
	num_samples = np.sum([p['rewards'].shape[0] for p in paths])
	print(f'Number of samples collected: {num_samples}')
	print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

	np.save(f'{pkl_file_path}.npy', paths)
	with open(f'{pkl_file_path}.pkl', 'wb') as f:
		pickle.dump(paths, f)


def download_d4rl_franka_data(env_name):
	datasets = []

	data_dir = 'data/'

	print(data_dir)

	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	assert env_name in ['kitchen-complete-v0', 'kitchen-partial-v0', 'kitchen-mixed-v0']

	pkl_file_path = os.path.join(data_dir, env_name)

	print("processing: ", env_name)

	env = gym.make(env_name)
	dataset = d4rl.qlearning_dataset(env)

	N = dataset['rewards'].shape[0]
	data_ = collections.defaultdict(list)

	episode_step = 0
	paths = []
	for i in range(N):
		done_bool = bool(dataset['terminals'][i])
		final_timestep = (episode_step == len(dataset['rewards'])-1)
		for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
			data_[k].append(dataset[k][i])
		if done_bool or final_timestep:
			episode_step = 0
			episode_data = {}
			for k in data_:
				episode_data[k] = np.array(data_[k])
			paths.append(episode_data)
			data_ = collections.defaultdict(list)
		episode_step += 1

	returns = np.array([np.sum(p['rewards']) for p in paths])
	num_samples = np.sum([p['rewards'].shape[0] for p in paths])
	print(f'Number of samples collected: {num_samples}')
	print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

	np.save(f'{pkl_file_path}.npy', paths)
	with open(f'{pkl_file_path}.pkl', 'wb') as f:
		pickle.dump(paths, f)





def episode_split_on_state(dataset,
						   clips_to_eps : bool = True,
						   eps : float =1e-5):

	if clips_to_eps:
		lim = 1 - eps
		dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

	dones_float = np.zeros_like(dataset["rewards"])

	for i in range(len(dones_float) - 1):
		if (
			np.linalg.norm(
				dataset["observations"][i + 1] - dataset["next_observations"][i]
			)
			> 1e-6
		):
			dones_float[i] = 1
		else:
			dones_float[i] = 0
			
	dones_float[-1] = 1

	return dones_float


def download_d4rl_antmaze_data_qlearning(env_name: str = "antmaze-umaze-v2",
							   clips_to_eps : bool = True,
							   eps : float =1e-5):
	# if you call get_dataset(*antmaze_envs), it doesnot gives you next_observation key
	# so, we start from qlearning dataset and split it to episode
	# https://github.com/Farama-Foundation/D4RL/issues/204#issuecomment-1463795988
	datasets = []
	print("you are downlading antmaze dataset with qlearning version")

	data_dir = 'data/'

	print(data_dir)

	if not os.path.exists(data_dir):
			os.makedirs(data_dir)

	antmaze_envs = ["antmaze-umaze-v2",
					"antmaze-umaze-diverse-v2",
					"antmaze-medium-diverse-v2",
					"antmaze-medium-play-v2",
					"antmaze-large-diverse-v2",
					"antmaze-large-play-v2",]
	
	assert env_name in antmaze_envs
	# for env_name in antmaze_envs:

	name = env_name
	pkl_file_path = os.path.join(data_dir, name)

	print("processing: ", name)

	env = gym.make(name)
	dataset = d4rl.qlearning_dataset(env)
	dones_float = episode_split_on_state(dataset, clips_to_eps, eps)

	data_ = collections.defaultdict(list)


	paths = []
	for i in tqdm(range(len(dataset["observations"]))):
		for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
			data_[k].append(dataset[k][i])
		if dones_float[i] == 1.0 and i + 1 < len(dataset["observations"]) :
			episode_data = {}
			if np.sum(data_["terminals"]) > 0:
				end_index = data_["terminals"].index(True)
				for k in data_:
					episode_data[k] = np.array(data_[k][:end_index+1])
			else:
				for k in data_:
					episode_data[k] = np.array(data_[k])
			paths.append(episode_data)
			data_ = collections.defaultdict(list)

	returns = np.array([np.sum(p['rewards']) for p in paths])
	num_samples = np.sum([p['rewards'].shape[0] for p in paths])
	print(f'Number of samples collected: {num_samples}')
	print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

	np.save(f'{pkl_file_path}.npy', paths)
	with open(f'{pkl_file_path}.pkl', 'wb') as f:
		pickle.dump(paths, f)



def download_d4rl_antmaze_data(env_name: str = "antmaze-umaze-v2",
							   clips_to_eps : bool = True,
							   eps : float =1e-5,
							   split_on_goal : bool = True):
	# if you call get_dataset(*antmaze_envs), it doesnot gives you next_observation key
	# so, we start from qlearning dataset and split it to episode
	# https://github.com/Farama-Foundation/D4RL/issues/204#issuecomment-1463795988
	datasets = []
	print("you are downlading antmaze dataset with get_dataset version")

	data_dir = 'data/'

	print(data_dir)

	if not os.path.exists(data_dir):
			os.makedirs(data_dir)

	antmaze_envs = ["antmaze-umaze-v2",
					"antmaze-umaze-diverse-v2",
					"antmaze-medium-diverse-v2",
					"antmaze-medium-play-v2",
					"antmaze-large-diverse-v2",
					"antmaze-large-play-v2",]
	
	assert env_name in antmaze_envs
	# for env_name in antmaze_envs:

	name = env_name
	pkl_file_path = os.path.join(data_dir, name)

	print("processing: ", name)

	env = gym.make(name)
	keys = ['observations', 'next_observations', 'actions', 'rewards', 'timeouts']

	if split_on_goal:
		dataset = env.get_dataset()
		dones_float = np.zeros_like(dataset["rewards"])
		for i in range(len(dataset["infos/goal"])-1):
			if np.array_equal(dataset["infos/goal"][i], dataset["infos/goal"][i+1]):
				dones_float[i] = 0
			else:
				dones_float[i] = 1
		keys.append("infos/goal")

		dataset["observations"] = dataset["observations"][:-1,:]
		dataset["next_observations"] = dataset["observations"][1:,:]
		dataset["actions"] = dataset["actions"][:-1,:]
		dataset["rewards"] = dataset["rewards"][:-1]
		dataset["timeouts"] = dataset["timeouts"][:-1]
		dataset["infos/goal"] = dataset["infos/goal"][:-1,:]
	else:
		dataset = d4rl.qlearning_dataset(env)
		dones_float = episode_split_on_state(dataset, clips_to_eps, eps)

	data_ = collections.defaultdict(list)


	paths = []
	for i in tqdm(range(len(dataset["observations"])-1)):
		for k in keys:
			data_[k].append(dataset[k][i])
		if dones_float[i] == 1.0 and i + 1 < len(dataset["observations"]) :
			episode_data = {}
			for k in data_:
				episode_data[k] = np.array(data_[k])
			paths.append(episode_data)
			data_ = collections.defaultdict(list)

	returns = np.array([np.sum(p['rewards']) for p in paths])
	num_samples = np.sum([p['rewards'].shape[0] for p in paths])
	print(f'Number of samples collected: {num_samples}')
	print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

	np.save(f'{pkl_file_path}.npy', paths)
	with open(f'{pkl_file_path}.pkl', 'wb') as f:
		pickle.dump(paths, f)


def download_d4rl_maze2d_data(env_name: str = "maze2d-umaze-v1",
							  clips_to_eps : bool = True,
							  eps : float =1e-5,
							  use_timeouts : bool = True):
	# if you call get_dataset(*antmaze_envs), it doesnot gives you next_observation key
	# so, we start from qlearning dataset and split it to episode
	# https://github.com/Farama-Foundation/D4RL/issues/204#issuecomment-1463795988
	datasets = []

	data_dir = 'data/'

	print(data_dir)

	if not os.path.exists(data_dir):
			os.makedirs(data_dir)

	antmaze_envs = ["maze2d-umaze-v1",
					"maze2d-medium-v1",
					"maze2d-large-v1"]
	
	assert env_name in antmaze_envs
	# for env_name in antmaze_envs:

	name = env_name
	pkl_file_path = os.path.join(data_dir, name)

	print("processing: ", name)

	env = gym.make(name)
	dataset = env.get_dataset()

	dataset["next_observations"] = dataset["observations"][1:]

	if use_timeouts:
		dones_float = dataset["timeouts"].astype(float)
	else: 
		dones_float = np.zeros_like(dataset["rewards"])
		for i in range(len(dataset["observations"])-1):
			if (dataset["rewards"][i] == 1.) and (dataset["rewards"][i+1] == 0.):
				dones_float[i] = 1.
		
	data_ = collections.defaultdict(list)

	paths = []
	for i in tqdm(range(len(dataset["observations"])-1)):
		for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
			data_[k].append(dataset[k][i])
		if dones_float[i] == 1.0 and i + 1 < len(dataset["observations"]) :
			episode_data = {}
			for k in data_:
				episode_data[k] = np.array(data_[k])
			paths.append(episode_data)
			data_ = collections.defaultdict(list)

	returns = np.array([np.sum(p['rewards']) for p in paths])
	num_samples = np.sum([p['rewards'].shape[0] for p in paths])
	print(f'Number of samples collected: {num_samples}')
	print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

	np.save(f'{pkl_file_path}.npy', paths)
	with open(f'{pkl_file_path}.pkl', 'wb') as f:
		pickle.dump(paths, f)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", type=str, default="halfcheetah_medium")
	parser.add_argument("--fraction", type=float, default=1.0)
	parser.add_argument("--qlearning", action="store_true")

	args = parser.parse_args()
	if len(args.env_name.split("_")) == 2:
		env_name, dataset_type = args.env_name.split("_")
		download_d4rl_mujoco_data(env_name, dataset_type, fraction=args.fraction)
	elif "ant" in args.env_name:
		if args.qlearning:
			download_d4rl_antmaze_data_qlearning(args.env_name)
		else:
			download_d4rl_antmaze_data(args.env_name)
	elif "maze2d" in args.env_name:
		download_d4rl_maze2d_data(args.env_name)
	elif "kitchen" in args.env_name:
		download_d4rl_franka_data(args.env_name)
	else:
		download_d4rl_adroit_data(args.env_name)