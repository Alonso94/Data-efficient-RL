import abc
import tensorflow as tf
import gpflow
from gpflow import Parameterized, Param, params_as_tensors, settings
import numpy as np

float_type = settings.dtypes.float_type


def squash(m, s, max=1.0):
    # Compute moments of the saturating function
    # e * (9 *sin(x(i))+sin(3 * x(i))) / 8
    k = tf.shape(m)[1]
    max_action = max * tf.ones((1, k), dtype=float_type)

    m_sf = max_action * tf.exp(-tf.diag_part(s) / 2) * tf.sin(m)

    lq = -(tf.diag_part(s)[:, None] + tf.diag_part(s)[None, :]) / 2
    s1 = (tf.exp(lq + s) - tf.exp(lq)) * tf.cos(tf.transpose(m) - m)
    s2 = (tf.exp(lq - s) - tf.exp(lq)) * tf.cos(tf.transpose(m) + m)
    s_sf=max_action * tf.transpose(max_action) * (s1-s2) /2

    c=max_action*tf.diag(tf.exp(-tf.diag_part(s)/2))*tf.cos(m)
    c_sf=tf.reshape(c,shape=(k,k))
    return m_sf,s_sf,c_sf

def randomize(model):
    m=1
    s=0.01
    model.kern.lengthscalse(m+s*np.random.normal(size=model.kern.lengthscalse.shape))
    model.kern.variance(m + s * np.random.normal(size=model.kern.variance.shape))
    if model.likelihood.variance.trainable:
        model.likelihood.variance.assign(m+s*np.random.normal)

class MGPR(gpflow.Parameterized):
    # Class for multi output gaussian process
    def __init__(self,X,Y):
        super(MGPR,self).__init__()

        self.num_out=Y.shape[1]
        self.num_dims=X.shape[1]
        self.num_datapoints=X.shape[0]

        self.create_models_and_optimizers(X,Y)

    def create_models_and_optimizers(self,X,Y):
        self.models=[]
        for i in range(self.num_out):
            kern=gpflow.kernels.RBF(input_dim=self.num_dims,ARD=True)
            kern.lengthscales.prior=gpflow.priors.Gamma(1,10)
            kern.variance.prior=gpflow.priors.Gamma(1.5,2)
            self.models.append(gpflow.models.GPR(X,Y[:,i:i+1],kern))
            self.models[i].clear()
            self.models[i].compile()
        self.optimizers=[]
        for model in self.models:
            optimizer=gpflow.train.ScipyOptimizer(method='L-BGFS-B')
            optimizer.minimize(model)
            self.optimizers.append(optimizer)

    def set_XY(self,X,Y):
        for i in range(self.num_out):
            self.models[i].X=X
            self.models[i].Y=Y[:,i:i+1]

    def optimize(self):
        for model,opt in zip(self.models,self.optimizers):
            session=opt._model.enquire_session(None)
            #randomize(model)
            opt._optimizer.minimize(session=session,
                                    feed_dict=opt._gen_feed_dict(opt,model,None),
                                    step_callback=None)
            likelihood=model.compute_log_likelihood()
            param=model.read_values(session)
            model.assign(param)

    def predict_on_noisy_inputs(self,m,s):

        # factorization cholesky
        self.k=tf.stack(model.kern(self.X) for model in self.models)
        batched_eye=tf.eye(tf.shape(self.X)[0],batch_shape=[self.num_out],dtype=float_type)
        L=tf.cholesky(self.k+self.noise[:,None,None]*batched_eye)
        ik=tf.cholesky_solve(L,batched_eye)
        Y_=tf.transpose(self.Y)[:,:,None]
        beta=tf.cholesky_solve(L,Y_)[:,:,0]

        # approximate GP regression at a noisy input via moment matching
        s=tf.tile(s[None,None,:,:],[self.num_out,self.num_out,1,1])
        inp=tf.tile((self.X-m)[None,:,:],[self.num_out,self.num_out,1,1])

        iL=tf.matrix_diag(1/self.lengthscales)
        iN=inp @ iL
        B=iL @ s[0,...]@iL+tf.eye(self.num_dims,dtype=float_type)

        t=tf.linalg.transpose(tf.matrix_diag(B,tf.linalg.transpose(iN),adjoint=True))
        lb=tf.exp(-tf.reduce_sum(iN*t,-1)/2)*beta
        tiL=t @ iL
        c=self.variance / tf.sqrt(tf.linalg.det(B))

        m_pred=(tf.reduce_sum(lb,-1)*c)[:,None]
        v_pred=tf.matmul(tiL,lb[:,:,None],adjoint_a=True)[...,0] * c[:,None]

        R=s@tf.matrix_diag(1/tf.square(self.lengthscales[None,:,:])+1/tf.square(self.lengthscales[:,None,:])+tf.eye(self.num_dims,dtype=float_type))
        x=inp[None,:,:,:]/tf.square(self.lengthscales[:,None,None,:])
        x2=-inp[:,None,:,:]/tf.square(self.lengthscales[None,:,None,:])
        Q=tf.matrix_solve(R,s)/2
        xs=tf.reduce_sum(x@Q*x,-1)
        x2s=tf.reduce_sum(x2@Q*x2,-1)
        maha=-2*tf.matmul(x@Q,x2,adjoint_b=True)+xs[:,:,:,None+x2s[:,:,None,:]]

        k=tf.log(self.variance)[:,None]-tf.reduce_sum(tf.square(iN),-1)/2
        L=tf.exp(k[:,None,:,None]+k[None,:,None,:]+maha)
        S=(tf.tile(beta[:,None,:,None],[1,self.num_out,1,1])@L@tf.tile(beta[None,:,:,None],[self.num_out,1,1,1]))[:,:,0,0]
        diagL=tf.transpose(tf.linalg.diag_part(tf.transpose(L)))
        S=S-tf.diag(tf.reduce_sum(tf.multiply(ik,diagL),[1,2]))
        S=S/tf.sqrt(tf.linalg.det(R))
        S=S+tf.diag(self.variance)
        s_pred=S-m_pred@tf.transpose(m_pred)

        return tf.transpose(m_pred),s_pred,tf.transpose(v_pred)


    @property
    def Y(self):
        return tf.concat([model.Y.parameter_tensor for model in self.models],axis=1)

    @property
    def X(self):
        return self.models[0].X.parameter_tensor

    @property
    def lengthscales(self):
        return tf.stack([model.kern.lengthscales.constrained_tensor for model in self.models])

    @property
    def variance(self):
        return tf.stack([model.kern.variance.constrained_tensor for model in self.models])

    @property
    def noise(self):
        return tf.stack([model.likelihood.variance.constrained_tensor for model in self.models])

class Reward(Parameterized):
    def __init__(self):
        Parameterized.__init__(self)

    @abc.abstractmethod
    def compute_reward(self, m, s):
        raise NotImplementedError

