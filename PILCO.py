import numpy as np
import gym
import tensorflow as tf
import gpflow
from utils import MGPR,RBFController,ExponentionalReward

float_type = gpflow.settings.dtypes.float_type

### PILCO class
class PILCO(gpflow.models.Model):
    def __init__(self,X,Y):
        super(PILCO,self).__init__()

        #hyper parameters
        target=[]
        target_weights=[]
        num_basis_functions=50
        max_action=2.0

        # Algorithm parts
        self.horizon = 30
        inducing_points=None
        self.model=MGPR(X,Y)
        self.state_dim=Y.shape[1]
        self.control_dim=X.shape[1]-Y.shape[1]
        self.controller=RBFController(self.state_dim,self.control_dim,num_basis_functions,max_action)
        self.reward=ExponentionalReward(self.state_dim)
        self.optimizer=gpflow.train.ScipyOptimizer(method="L-BFGS-B")

        # initial state
        self.m_init=X[0:1,0:self.state_dim]
        self.s_init=np.diag(np.ones(self.state_dim)*0.1)

    def propagate(self,m_x,s_x):
        m_u,s_u,c_xu=self.controller.compute_action(m_x,s_x)

        m=tf.concat([m_x,m_u],axis=1)
        s1=tf.concat([s_x,s_x @ c_xu],axis=1)
        s2=tf.concat([tf.transpose(s_x @ c_xu),s_u],axis=1)
        s=tf.concat([s1,s2],axis=0)

        m_dx,s_dx,c_dx=self.model.predict_on_noisy(m,s)
        m=m_dx+m_x
        s=s_dx + s_x + s1@c_dx + c_dx.transpose() @ s1.transpose()

        m.set_shape([1,self.state_dim])
        s.set_shape([self.state_dim,self.state_dim])

        return m,s

    def predict(self,m,s,n):
        loop_vars=[tf.constant(0,tf.int32),m,s,tf.constant([[0]],float_type)]
        c=lambda j,m,s,reward : j<n
        body=lambda j,m,s,reward: (j+1,
                                   *self.propagate(m,s),
                                   tf.add(reward,self.reward.compute_reward(m,s)[0]))
        _,m,s,reward=tf.while_loop(c,body,loop_vars)

    @gpflow.name_scope('likelihood')
    def _build_likelihood(self):
        reward=self.predict(self.m_init,self.s_init,self.horizon)[2]
        return reward

    @gpflow.autoflow()
    def compute_reward(self):
        return self._build_likelihood()

    def optimize_models(self):
        # 200 iter
        self.model.optimize()
        ## print out things

    def optimize_police(self,maxiter=50):
        # 50 iter
        self.optimizerpt.minimize(self,maxiter)
        session=self.optimizer._model.enquire_session()
        best_parameters=self.read_values(session)
        best_reward=self.compute_reward()
        self.assign(best_parameters)



### env -> rollout and expirement
env = gym.make('Reacher-v2')

def rollout(T, random=False):
    X = []
    Y = []
    x = env.reset()
    for t in range(T):
        if random:
            u = env.action_space.sample()
        else:
            u = pilco.compute_action(x)[0]
        new_x, _, done, _ = env.step(u)
        env.render()
        X.append((x, u))
        Y.append(new_x - x)
        x=new_x
        if done:
            break
    return X,Y

sess = tf.Session()
X,Y=rollout(40,random=True)
pilco=PILCO(X,Y)
n=8
for i in range(n):
    env.reset()
    pilco.optimize_models()
    pilco.optimize_police()
    X_,Y_=rollout(40)
    X=np.vstack(X,X_)
    Y=np.vstack(Y,Y_)
    pilco.model.set_XY(X,Y)
