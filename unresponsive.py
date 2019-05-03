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
    def __init__(self,target):
        self.env = gym.make('FetchReach-v1').env
        self.env.goal = target
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.target=target
        self.distance_threshold=0.01
        self.q=[0.0,-1.0,0.0,2.0,0.0,0.5,0.0]

    def step(self, action):
        self.q+=action
        qpos = {
            'robot0:shoulder_pan_joint': self.q[0],
            'robot0:shoulder_lift_joint': self.q[1],
            'robot0:upperarm_roll_joint': self.q[2],
            'robot0:elbow_flex_joint': self.q[3]*0.0,
            'robot0:forearm_roll_joint':self.q[4],
            'robot0:wrist_flex_joint':self.q[5],
            'robot0:wrist_roll_joint':self.q[6]
        }
        for name, value in qpos.items():
            self.env.sim.data.set_joint_qpos(name, value)
        self.env.sim.forward()
        time.sleep(1 / 30)
        ob=self.env._get_obs()
        #print(ob)
        return np.concatenate((ob['achieved_goal'],self.q),axis=0)

    def reset(self):
        self.q = [0.0, -1.0, 0.0, 2.0, 0.0, 0.5, 0.0]
        qpos = {
            'robot0:shoulder_pan_joint':0.0 ,
            'robot0:shoulder_lift_joint': -1.0,
            'robot0:upperarm_roll_joint': 0.0,
            'robot0:elbow_flex_joint': 2.0,
            'robot0:forearm_roll_joint':0.0,
            'robot0:wrist_flex_joint': 0.5,
            'robot0:wrist_roll_joint': 0.0
        }
        for name, value in qpos.items():
            self.env.sim.data.set_joint_qpos(name, value)
        self.env.goal = self.target
        self.env.sim.forward()
        time.sleep(1 / 30)
        ob = self.env._get_obs()
        return np.concatenate((ob['achieved_goal'],self.q),axis=0)

    def render(self):
        self.env.render()
        time.sleep(1/30)



i=0
def rollout(env,T, random=False,trial=0):
    start=time.time()
    X = []
    Y = []
    x=env.reset()
    tt=[]
    env.render()
    rewards=[]
    for t in range(T):
        if random:
            u = np.random.rand(7)*0.1-0.05
        else:
            u = pilco.compute_action(x[None, :])[0, :]
        new_x = env.step(u)
        tt.append(t)
        distance=np.linalg.norm(new_x[:3]-env.target)
        rewards.append(distance)
        env.render()
        X.append(np.hstack((x, u)))
        Y.append(new_x-x)
        x=new_x
        if np.linalg.norm(new_x[:3]-env.target) <0.05:
            break
    plt.plot(tt, rewards)
    plt.title("distance to goal - Trial %d" %trial)
    plt.xlabel("t")
    plt.ylabel("d")
    plt.savefig("dist%d.png"%trial)
    plt.show()
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
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 6))
    axes[0].set_title('One step prediction - Trial#%d' % trial)
    axes[2].set_xlabel('t')
    axes[1].set_ylabel('x')
    for i, m in enumerate(pilco.mgpr.models):
        y_pred_test, var_pred_test = m.predict_y(X)
        axes[i].plot(range(len(y_pred_test)), y_pred_test, Y[:, i])
        axes[i].fill_between(range(len(y_pred_test)),
                         y_pred_test[:, 0] - 2 * np.sqrt(var_pred_test[:, 0]),
                         y_pred_test[:, 0] + 2 * np.sqrt(var_pred_test[:, 0]), alpha=0.3)
        if i==2: break

    plt.savefig("onep%d.png" % trial)
    plt.show()
    m_p = np.zeros((T, state_dim))
    S_p = np.zeros((T, state_dim, state_dim))
    m_init = X[0:1, 0:state_dim]
    S_init = np.diag(np.ones(state_dim) * 0.1)
    for h in range(T):
        m_h, S_h, _ = predict_trajectory_wrapper(pilco, m_init, S_init, h)
        m_p[h, :], S_p[h, :, :] = m_h[:], S_h[:, :]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 6))
    axes[0].set_title('Multi-step prediction - Trial#%d' % trial)
    axes[2].set_xlabel('t')
    axes[1].set_ylabel('x')
    for i in range(state_dim):
        axes[i].plot(range(T - 1), m_p[0:T - 1, i], X[1:T, i])  # can't use Y_new because it stores differences (Dx)
        axes[i].fill_between(range(T - 1),
                         m_p[0:T - 1, i] - 2 * np.sqrt(S_p[0:T - 1, i, i]),
                         m_p[0:T - 1, i] + 2 * np.sqrt(S_p[0:T - 1, i, i]), alpha=0.2)
        if i == 2: break

    plt.savefig("multistep%d.png" % trial)
    plt.show()

with tf.Session() as sess:
    p_start=time.time()
    target=np.array([1.2,0.38,0.38])
    env = EnvWrapper(target)
    T=50
    num_basis_functions = 50
    max_action = 0.1
    time_on_real_robot = 0
    X,Y,t=rollout(env,T,random=True,trial=0)
    time_on_real_robot += t
    state_dim = Y.shape[1]
    control_dim = X.shape[1] - Y.shape[1]
    controller = RbfController(state_dim,control_dim, num_basis_functions, max_action)
    reward = ExponentialReward(3,t=target)
    pilco=PILCO(X,Y,controller=controller,reward=reward)
    plot(pilco,X,Y,T,0)
    n=4
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
        X_,Y_,t=rollout(env,T,trial=i)
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