from maddpg.maddpg_core import MaddpgTrainer
import maddpg_env
from maddpg_env.multiagent.environment import MultiAgentEnv
import torch
import numpy as np
import maddpg_env.make_env as make_env
import argparse
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import maddpg_env.multiagent.scenarios.simple_coverage as coverage

def get_trainers(env, num_adversaries, arglist=None):
    obs_dim_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    print(obs_dim_n)
    act_dim_n = [env.action_space[i].shape[0] for i in range(env.n)]
    trainers = [MaddpgTrainer(obs_dim_n=obs_dim_n, agent_index=i, env=env, act_dim_n=act_dim_n) for i in
                range(num_adversaries)]
    trainers += [MaddpgTrainer(obs_dim_n=obs_dim_n, agent_index=i, env=env, act_dim_n=act_dim_n) for i in
                 range(num_adversaries, env.n)]

    return trainers


def train(arglist=None):
    env = make_env.make_env(arglist.scenario)

    # env = make_env.make_env('simple_spread')
    trainers = get_trainers(env, num_adversaries=arglist.num_adversaries, arglist=arglist)

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    agent_info = [[[]]]  # placeholder for benchmarking info
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()
    print('Starting iterations...')
    done_count = 0
    while len(episode_rewards) < 100000:
        # get action
        if episode_step < 10:
            action_n = [agent.action(obs, noise=0.05) for agent, obs in zip(trainers, obs_n)]
        else:
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
        # pre_coverage = [coverage.Scenario.reward(agent, env.world) for agent in env.agents]
        # pre_rew =[sum(pre_coverage)]
        #pre_rew_n = pre_rew * env.n
        new_obs_n, rew_n, done_n, info_n = env.step(
            action_n)  # the obs_n,rew_n ... is for all the agent in the world
        # for i in range(env.n):
        #     rew_n[i] = rew_n[i] - pre_rew_n[i]

        episode_step += 1
        done = False

        for data in done_n:
            if data == True:
                done = True
            else:
                continue
        if (done):
            done_count += 1


        terminal = (episode_step >= arglist.max_episode_len)  # truncated
        # collect experience
        for i, agent in enumerate(trainers):
            agent.store(obs=obs_n[i], act=action_n[i], obs2=new_obs_n[i], rew=rew_n[i], done=done_n[i])
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew  # the sum of the reward for all the agent
            agent_rewards[i][-1] += rew  # the reward for each agent

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

        # increment global step counter
        train_step += 1

        # for benchmarking learned policies
        if train_step % 100 == 0:
            for agent in trainers:
                loss = agent.update(trainers)


        if (terminal or done) and (len(episode_rewards) % arglist.save_rate == 0):
            # U.save_state(arglist.save_dir, saver=saver)
            # print statement depends on whether or not there are adversaries
            if arglist.num_adversaries == 0:
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {},done_rate:{}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                    round(time.time() - t_start, 3), done_count / arglist.save_rate))
                done_count = 0
            else:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {} ".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                    [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
    #test_coverage(env, trainers)
    test(env,trainers)


def test_coverage(env, trainers):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    colors = ['Bules', 'Reds']
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()
    agent_pos = [[] for i in range(len(trainers))]
    episode_rewards = []
    while True:
        # get action
        action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)  # the obs_n,rew_n ... is for all the agent in the world
        for i, rew in enumerate(rew_n):
            episode_rewards.append(rew)  # the sum of the reward for all the agent
        for i in range(len(trainers)):
            agent_pos[i].append(obs_n[i][3:6])
        episode_step += 1
        obs_n = new_obs_n
        done = all(done_n)
        time.sleep(0.1)
        # env.render()
        if done or episode_step > 25:
            print('in episode the cumulate rewards is :{}'.format(sum(episode_rewards)))
            print('reward at each time step:{}'.format(episode_rewards))
            episode_rewards = []

            # plot the area of the landmark
            # theta = np.linspace(0,2*np.pi,100)
            for landmark in env.world.landmarks:
                pos = landmark.state.p_pos
                radius = landmark.size
                ax.scatter3D(pos[0], pos[1], pos[2], cmap="Reds")
                R, alpha = 0, np.linspace(0, 2 * np.pi, 100)
                x = pos[0] + radius * np.sin(alpha)
                y = pos[1] + radius * np.cos(alpha)
                z = np.zeros(x.shape)
                ax.plot(x, y, z)
                # radius = np.linspace(0,env.world.landmarks[i].size)
                # theta,radius = np.meshgrid(theta,radius)
                # x , y = env.world.landmarks[i].state.p_pos[0] + np.cos(theta)*radius,\
                #         env.world.landmarks[i].state.p_pos[1] + np.sin(theta)*radius
                # height = np.zeros(x.shape)
                # ax.plot_surface(x,y,height,cmap='rainbow')
            # print('agent position list:{}'.format(agent_pos))
            # plot the curve of the trajectory
            for i in range(len(trainers)):
                agent_pos[i] = np.vstack(agent_pos[i])
                ax.scatter3D(agent_pos[i][:, 0], agent_pos[i][:, 1], agent_pos[i][:, 2], cmap=colors[i])
                # print('agent:{} position:{}'.format(i,agent_pos[i]))
            # plot the coverage area of the Uav
            R, alpha = 0, np.linspace(0, 2 * np.pi, 100)
            for i in range(len(trainers)):
                finally_pos = agent_pos[i][::-1][0]
                R = finally_pos[2] * np.sin(env.agents[i].coverage_theta)
                x = finally_pos[0] + R * np.sin(alpha)
                y = finally_pos[1] + R * np.cos(alpha)
                z = np.zeros(x.shape)
                ax.plot(x, y, z)

            plt.show()
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            agent_pos = [[] for i in range(len(trainers))]
            obs_n = env.reset()
            print('step to complete the mission{}'.format(episode_step))
            episode_step = 0
            t_start = time.time()


def test(env, trainers):
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()
    while True:
        # get action
        action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)  # the obs_n,rew_n ... is for all the agent in the world
        episode_step += 1
        obs_n = new_obs_n
        done = all(done_n)
        time.sleep(0.1)
        env.render()
        if done or episode_step > 50:
            obs_n = env.reset()
            print('step to complete the mission{}'.format(episode_step))
            episode_step = 0
            t_start = time.time()

    get_trainers(env, arglist.num_adversaries, arglist)


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=5000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='simple_spread', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
