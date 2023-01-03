import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax.scipy.optimize as sopt
import jax

import pandas as pd

data = pd.read_csv("./data.csv")
data.index = pd.to_datetime(data["date"])
data.drop(["date"], axis = 1, inplace=True)
lnoil = jnp.log(data["oil"].values)
lnng = jnp.log(data["ng"].values)

p0 = jnp.array([0.5, 0.5])

P = jnp.array([[p0[0], 1 - p0[0]], [1 - p0[1], p0[1]]])

def likelihood(theta, lnoil, lnng):

    obs = len(lnng)

    alpha1 = theta[0]
    alpha2 = theta[1]
    alpha3 = theta[2]
    alpha4 = theta[3]
    alpha5 = theta[4]
    alpha6 = theta[5]
    p11 = 1 / (1 + jnp.exp(-theta[6]))
    p22 = 1 / (1 + jnp.exp(-theta[7]))

    dist1 = (1/(alpha5 * jnp.sqrt(2 * jnp.pi))) * jnp.exp((-(lnng-alpha1-alpha3*lnoil)**2)/(2*alpha5**2))

    dist2 = (1/(alpha6 * jnp.sqrt(2 * jnp.pi))) * jnp.exp((-(lnng-alpha2-alpha4*lnoil)**2)/(2*alpha6**2))

    dist = jnp.transpose(jnp.array([dist1, dist2]))

    ov = jnp.array([1,1])

    P = jnp.array([[p11, 1-p11], [1- p22, p22]])
    
    xia = jnp.zeros([obs, 2])    
    xib = jnp.zeros([obs, 2])
    model_lik = jnp.zeros(obs)

    xia = xia.at[0].set(jnp.array([p11,p22]) * dist[1,] / jnp.matmul(jnp.array([p11,p22]), dist[1,]))

    for i in range(obs):
        xib = xib.at[i + 1].set(jnp.matmul(P, xia[i,]))
        xia = xia.at[i + 1].set((xib[i+1,] * dist[i+1,])/jnp.matmul(xib[i+1,], dist[i+1,]))
        model_lik = model_lik.at[i+1].set(jnp.matmul(xib[i+1,], dist[i+1,]))
	
	
    logl = jnp.sum(jnp.log(model_lik[2:obs]))
    return(-logl)
 

gpus = jax.devices("gpu")
likelihood_jit = jit(likelihood, device=gpus[0])

theta = jnp.array([-.05, .01, .2, .4, .1, .2, .5, .5])

## minimization-----

result_jit = sopt.minimize(likelihood_jit, x0=theta, args=(lnoil, lnng), method="BFGS")
print(result_jit)

## try jaxopt--------

import jaxopt

solver1 = jaxopt.GradientDescent(fun=likelihood_jit, maxiter=500)
res_gd = solver1.run(theta, lnoil, lnng)  ## similar retuls to above

## runs out of memory
# solver2 = jaxopt.LBFGS(fun=likelihood_jit, maxiter=500)
# res_lbfgs = solver2.run(theta, lnoil, lnng)  ## similar retuls to above
# solver3 = jaxopt.NonlinearCG(fun=likelihood_jit, maxiter=500)
# res_ncg = solver3.run(theta, lnoil, lnng)  ## similar retuls to above

### forward pass through the filter------------

obs = len(lnng)

alpha1_hat = result_jit.x[0]
alpha2_hat = result_jit.x[1]
alpha3_hat = result_jit.x[2]
alpha4_hat = result_jit.x[3]
alpha5_hat = result_jit.x[4]
alpha6_hat = result_jit.x[5]
p11_hat = 1 / (1 + jnp.exp(-result_jit.x[6]))
p22_hat = 1 / (1 + jnp.exp(-result_jit.x[7]))

dist1 = (1/(alpha5_hat * jnp.sqrt(2 * jnp.pi))) * jnp.exp((-(lnng-alpha1_hat - alpha3_hat*lnoil)**2)/(2*alpha5_hat**2))

dist2 = (1/(alpha6_hat * jnp.sqrt(2 * jnp.pi))) * jnp.exp((-(lnng-alpha2_hat-alpha4_hat*lnoil)**2)/(2*alpha6_hat**2))

dist = jnp.transpose(jnp.array([dist1, dist2]))

ov = jnp.array([1,1])

P = jnp.array([[p11_hat, 1-p11_hat], [1- p22_hat, p22_hat]])

xia = jnp.zeros([obs, 2])    
xib = jnp.zeros([obs, 2])
model_lik = jnp.zeros(obs)

xia = xia.at[0].set(jnp.array([p11_hat,p22_hat]) * dist[1,] / jnp.matmul(jnp.array([p11_hat,p22_hat]), dist[1,]))

for i in range(obs):
    xib = xib.at[i + 1].set(jnp.matmul(P, xia[i,]))
    xia = xia.at[i + 1].set((xib[i+1,] * dist[i+1,])/jnp.matmul(xib[i+1,], dist[i+1,]))
    model_lik = model_lik.at[i+1].set(jnp.matmul(xib[i+1,], dist[i+1,]))

import plotly.express as px

state_2 = pd.DataFrame(xia[:,2])
fig = px.line(state_2, title='NG/Oil')
fig.show()


## good results-------
