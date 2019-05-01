import numpy as np
import gym
import tensorflow as tf
import gpflow
from gpflow import autoflow
from gym.wrappers import Monitor

from pilco import PILCO

from controller import RbfController
from reward import ExponentialReward
import time
import matplotlib
import matplotlib.pyplot as plt

float_type = gpflow.settings.dtypes.float_type

np.random.seed(13)

## env -> rollout and expirement
class EnvWrapper():
    def __init__(self,target):
        self.env = gym.make('FetchReach-v1').env
        self.i=0
        #self.env = Monitor(self.env, 'video/' + 'vid', force=True)
        #self.env2 = gym.wrappers.Monitor(self.env, "./vid"+str(self.i), video_callable=lambda episode_id: True, force=True)
        self.env.goal = target
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.target=target
        self.distance_threshold=0.01

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        return ob['achieved_goal'], r, done, {}

    def reset(self):
        #self.env2.close()
        self.i+=1
        #self.env2 = gym.wrappers.Monitor(self.env, "./vid"+str(self.i), video_callable=lambda episode_id: True, force=True)
        self.env.reset()
        self.env.goal=self.target
        ob=self.env._get_obs()
        return ob['achieved_goal']

    def render(self):
        self.env.render()
        time.sleep(1/30)




def rollout(env,T, random=False):
    X = []
    Y = []
    x = env.reset()
    for t in range(T):
        if random:
            u = env.action_space.sample()
        else:
            u = pilco.compute_action(x[None, :])[0, :]
        new_x, _, done, _ = env.step(u)
        env.render()
        X.append(np.hstack((x, u)))
        Y.append(new_x - x)
        x=new_x
        if done:
            break
    return np.stack(X),np.stack(Y)

@autoflow((float_type,[None, None]), (float_type,[None, None]))
def predict_one_step_wrapper(mgpr, m, s):
    return mgpr.predict_on_noisy_inputs(m, s)


@autoflow((float_type,[None, None]), (float_type,[None, None]), (np.int32, []))
def predict_trajectory_wrapper(pilco, m, s, horizon):
    return pilco.predict(m, s, horizon)


@autoflow((float_type,[None, None]), (float_type,[None, None]))
def compute_action_wrapper(pilco, m, s):
    return pilco.controller.compute_action(m, s)


@autoflow((float_type, [None, None]), (float_type, [None, None]))
def reward_wrapper(reward, m, s):
    return reward.compute_reward(m, s)

def plot(pilco,X,Y,T):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 3))
    for i, m in enumerate(pilco.mgpr.models):
        y_pred_test, var_pred_test = m.predict_y(X)
        axes[i].plot(range(len(y_pred_test)), y_pred_test, Y[:, i])
        axes[i].fill_between(range(len(y_pred_test)),
                         y_pred_test[:, 0] - 2 * np.sqrt(var_pred_test[:, 0]),
                         y_pred_test[:, 0] + 2 * np.sqrt(var_pred_test[:, 0]), alpha=0.3)
    plt.show()

    m_p = np.zeros((T, state_dim))
    S_p = np.zeros((T, state_dim, state_dim))
    m_init = X[0:1, 0:state_dim]
    S_init = np.diag(np.ones(state_dim) * 0.1)
    for h in range(T):
        m_h, S_h, _ = predict_trajectory_wrapper(pilco, m_init, S_init, h)
        m_p[h, :], S_p[h, :, :] = m_h[:], S_h[:, :]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 3))
    for i in range(state_dim):
        axes[i].plot(range(T - 1), m_p[0:T - 1, i], X[1:T, i])  # can't use Y_new because it stores differences (Dx)
        axes[i].fill_between(range(T - 1),
                         m_p[0:T - 1, i] - 2 * np.sqrt(S_p[0:T - 1, i, i]),
                         m_p[0:T - 1, i] + 2 * np.sqrt(S_p[0:T - 1, i, i]), alpha=0.2)
    plt.show()

with tf.Session() as sess:
    target=np.array([1.2,0.38,0.38])
    env = EnvWrapper(target)
    T=50
    num_basis_functions = 20
    max_action = 4.0
    X,Y=rollout(env,T,random=True)
    state_dim = Y.shape[1]
    control_dim = X.shape[1] - Y.shape[1]
    controller = RbfController(state_dim,control_dim, num_basis_functions, max_action)
    reward = ExponentialReward(state_dim,t=target)
    pilco=PILCO(X,Y,controller=controller,reward=reward)
    plot(pilco,X,Y,T)
    n=8

    for i in range(n):
        env.reset()
        pilco.optimize_models()
        print("model optimization done!")
        pilco.optimize_policy()
        print("policy optimization done!")
        X_,Y_=rollout(env,T)
        plot(pilco,X_,Y_,T)
        X=np.vstack((X,X_[:T, :]))
        Y=np.vstack((Y,Y_[:T, :]))
        pilco.mgpr.set_XY(X,Y)
