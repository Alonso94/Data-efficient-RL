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

np.random.seed(0)

## env -> rollout and expirement
class EnvWrapper():
    def __init__(self):
        self.env = gym.make('CartPole-v0')

    def step(self, action):
        ob,_,done,_=self.env.step(action)
        return np.hstack((ob[:2],np.sin(ob[2]),np.cos(ob[2]),ob[-1])),done

    def reset(self):
        ob=self.env.reset()
        return np.hstack((ob[:2],np.sin(ob[2]),np.cos(ob[2]),ob[-1]))

    def render(self):
        self.env.render()
        time.sleep(1/30)



i=0
def rollout(env,T,random=False):
    start=time.time()
    X = []
    Y = []
    x=env.reset()
    env.render()
    for t in range(T):
        if random:
            u = env.env.action_space.sample()
        else:
            a = pilco.compute_action(x[None, :])[0, :]
            u=int(np.round(a[0]))
        new_x,done= env.step(u)
        env.render()
        X.append(np.hstack((x, u)))
        Y.append(new_x-x)
        x=new_x
        if done:
            break
    end=time.time()
    print ("time on real robot= %.1f s"%(end-start))
    return np.stack(X),np.stack(Y),end-start

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

def plot(pilco,X,Y,T,trial):
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 6))
    axes[0].set_title('One step prediction - Trial#%d' % trial)
    axes[2].set_xlabel('t')
    axes[1].set_ylabel('x')
    for i, m in enumerate(pilco.mgpr.models):
        y_pred_test, var_pred_test = m.predict_y(X)
        axes[i].plot(range(len(y_pred_test)), y_pred_test, Y[:, i])
        axes[i].fill_between(range(len(y_pred_test)),
                         y_pred_test[:, 0] - 2 * np.sqrt(var_pred_test[:, 0]),
                         y_pred_test[:, 0] + 2 * np.sqrt(var_pred_test[:, 0]), alpha=0.3)

    plt.savefig("onep%d.png" % trial)
    plt.show()

with tf.Session() as sess:
    p_start=time.time()
    env = EnvWrapper()
    T=50
    num_basis_functions = 10
    max_action = 1.0
    time_on_real_robot = 0
    X,Y,t=rollout(env,T,random=True)
    time_on_real_robot += t
    state_dim = Y.shape[1]
    control_dim = X.shape[1] - Y.shape[1]
    controller = RbfController(state_dim,control_dim, num_basis_functions, max_action)
    reward = ExponentialReward(state_dim,t=np.array([0.0,0.0,0.0,1.0,0.0]))
    pilco=PILCO(X,Y,controller=controller,reward=reward)
    plot(pilco,X,Y,T,0)
    n=7
    t_model=0
    t_policy=0
    for i in range(1,n):
        env.reset()
        t1 = time.time()
        pilco.optimize_models()
        t2 = time.time()
        t_model+=t2-t1
        print("model optimization done!")
        pilco.optimize_policy()
        t3 = time.time()
        t_policy+=t3-t2
        print("policy optimization done!")
        X_,Y_,t=rollout(env,T)
        time_on_real_robot += t
        plot(pilco,X_,Y_,T,i)
        X=np.vstack((X,X_[:T, :]))
        X=X[:2*T]
        Y=np.vstack((Y,Y_[:T, :]))
        Y=Y[:2*T]
        pilco.mgpr.set_XY(X,Y)
    print("t_robot= %.2f s" %time_on_real_robot)
    print("t_model= %.2f s" %t_model)
    print("t_policy= %.2f s" %t_policy)
    print("program running time = %d s" %(time.time()-p_start))