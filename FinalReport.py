#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:00:40 2020

@author: rafael
"""
import numpy as np
import pandas as pd
import scipy.stats as st

#Process data
data = pd.read_csv('londoncrime.csv')

years = np.sort(data.year.unique())
T = len(years)

years = np.array([2005, 2006, 2007, 2008, 2009, 2010, 2011])

boroughs = np.sort(data.borough.unique())
J = len(boroughs)


y = data.year.groupby([data.year, data.borough]).size().unstack().fillna(0).astype(int)


#Compute log-likelihood
def log_likelihood(y, theta, mui, sigma, tau_c, tau):
    T = len(mui)
    J = len(theta)
    
    ll = np.zeros((T-1,J))
    for i in range(1,T):
        for j in range(J):
            ll[i-1, j] = st.norm.logpdf(y[boroughs[j]][years[i]], loc = theta[j] + mui[i], scale = sigma)
    return ll



#Create conditional distributions
def sample_mu0(y, theta, mui, sigma, tau_c, tau):
    mu0 = st.norm.rvs(loc = mui[1], scale = tau)
    return mu0

def sample_mui(y, theta, mui, sigma, tau_c, tau, i):
    nom = (1/tau**2) * mui[i-1] + (1/tau**2) * mui[i+1]
    for j in range(J):
        nom = nom + (1/sigma**2) * (y[boroughs[j]][years[i]] - theta[j])
        
    denom =  2/tau**2 + J/sigma**2
    
    mui = st.norm.rvs(loc = nom/denom, scale = np.sqrt(1/denom))
    return mui

def sample_muT(y, theta, mui, sigma, tau_c, tau):
    nom = (1/tau**2) * mui[T-1]
    for j in range(J):
        nom = nom + (1/sigma**2) * (y[boroughs[j]][years[T]] - theta[j])
        
    denom =  1/tau**2 + J/sigma**2
    
    mui = st.norm.rvs(loc = nom/denom, scale = np.sqrt(1/denom))
    return mui  
        
def sample_thetaj(y, theta, mui, sigma, tau_c, tau, j):
    nom = 0
    for i in range(1, T + 1):
        nom = nom + (1/sigma**2) * (y[boroughs[j]][years[i]] - mui[i])
    denom = 1/tau**2 + T/sigma**2
    
    thetaj= st.norm.rvs(loc = nom/denom, scale = np.sqrt(1/denom))
    return thetaj

        
def sample_sigma(y, theta, mui, sigma, tau_c, tau):

    sigma2 = 0
    for i in range(1, T + 1):
        for j in range(J):
            sigma2 = sigma2 + (y[boroughs[j]][years[i]] - theta[j] - mui[i])**2
            
    sigma2 = st.invgamma.rvs((T*J)/2, scale = np.sqrt(sigma2)/2)
    return np.sqrt(sigma2)

def sample_tauc(y, theta, mui, sigma, tau_c, tau):
    tauc2 = np.sum(theta**2)#/(J-1)
    
    tauc2 = st.invgamma.rvs((J-1)/2, scale = np.sqrt(tauc2)/2)
    return np.sqrt(tauc2)


def sample_tau(y, theta, mui, sigma, tau_c, tau):
    
    tau2 = 0
    for i in range(1,T + 1):
        tau2 = tau2 + (mui[i]-mui[i-1])**2
    #tau2 = tau2/(T-1)
    
    tau2 = st.invgamma.rvs((T-1)/2, scale = np.sqrt(tau2)/2)
    return np.sqrt(tau2)
    

n = 250
m = 10

Mus = np.zeros((n, m, 7))
Thetas = np.zeros((n, m, 32))
Sigmas = np.zeros((n, m))
Taucs = np.zeros((n, m))
Taus  = np.zeros((n, m))
LLs = []

mui_old = np.array([5.375,  5.375  , 5.09375, 4.84375, 4.0625 , 3.875  , 2.9375 ])
theta_old = np.zeros(32)
sigma_old = 2.0
tauc_old = 2.0
tau_old = 5.0

for chain in range(m): 
    
    mui_old = np.array([5.375,  5.375  , 5.09375, 4.84375, 4.0625 , 3.875  , 2.9375 ])
    theta_old = np.zeros(32)
    sigma_old = 2.0
    tauc_old = 2.0
    tau_old = 5.
    
    for it in range(n):
        
        mui_new = np.zeros(T+1)
        mui_new[0] = sample_mu0(y, theta_old, mui_old, sigma_old, tauc_old, tau_old)
        for i in range(1,T):
            mui_new[i] = sample_mui(y, theta_old, mui_old, sigma_old, tauc_old, tau_old, i)
        mui_new[T] = sample_muT(y, theta_old, mui_old, sigma_old, tauc_old, tau_old)
        
        theta_new = np.zeros(J)
        #for j in range(J):
        #    theta_new[j] = sample_thetaj(y, theta_old, mui_old, sigma_old, tauc_old, tau_old, j)
            
        sigma_new = sample_sigma(y, theta_old, mui_old, sigma_old, tauc_old, tau_old)
        tauc_new =  sample_tauc(y, theta_old, mui_old, sigma_old, tauc_old, tau_old)
        tau_new =   sample_tau(y, theta_old, mui_old, sigma_old, tauc_old, tau_old)
        ll = log_likelihood(y, theta_new, mui_new, sigma_new, tauc_new, tau_new)
        LLs.append(ll)
    
    
        Mus[it, chain, :] = mui_new
        Thetas[it, chain, :] = theta_new
        Sigmas[it, chain] = sigma_new
        Taucs[it, chain] = tauc_new
        Taus[it, chain] = tau_new
        
        mui_old = mui_new
        theta_old = theta_new
        sigma_old = sigma_new
        tauc_old = tauc_new
        tau_old = tau_new
    

    for t in range(100, 200):
        plt.plot(np.array(Mus)[t,1:])



    M = np.zeros((J,J))
    for i in range(J):
        for j in range(J):
            M[i,j] = np.mean(np.array(Thetas)[:,i] > np.array(Thetas)[:,j])


mus = np.mean(np.array(Mus), axis = 0)
thetas = np.mean(np.array(Thetas), axis = 0)
sigma = np.mean(Sigmas)
taus = np.mean(Taucs)
tau = np.mean(Taus)


Ys = np.zeros((6, 32, 100))
def sample_dataset(mus, thetas, sigma, tauc, tau, nsamps = 100):
    for it in range(nsamps):
        k1 = np.random.choice(range(100, 250))
        k2 = np.random.choice(range(10))

        mus = Mus[k1,k2,:]
        thetas = Thetas[k1,k2,:]
        sigma = Sigmas[k1,k2]
        taus = Taucs[k1,k2]
        tau = Taus[k1,k2]
        
        for i in range(1, 7):
            for j in range(32):
                Ys[i-1,j,it] = max(st.norm.rvs(loc = mus[i] + thetas[j], scale = sigma), 0)

for i in range(len(years)-1):

    plt.hist(Ys[i].ravel(), bins = 10, density = True)
    plt.axvline(y.loc[years[i+1]].to_numpy().mean(), color='r')
    plt.axvline(y.loc[years[i+1]].to_numpy().max(), color='g')
    plt.axvline(y.loc[years[i+1]].to_numpy().min(), color='g')

    plt.axvline(np.quantile(y.loc[years[i+1]].to_numpy(), q=0.25), color='b')
    plt.axvline(np.quantile(y.loc[years[i+1]].to_numpy(), q=0.75), color='b')


    plt.xlabel('Number of crimes')
    plt.title('Simulated distribution and data statistics {}'.format(years[i+1]))
    plt.savefig('Year{}.png'.format(years[i+1]))
    plt.close()



mc = np.mean(np.array(Mus), axis = 0)
tc = np.mean(np.array(Thetas), axis = 0)
sc = np.mean(Sigmas)




for i in range(len(boroughs)):

    plt.hist(Ys[:,i,:].ravel(), bins = 10, density = True)
    plt.axvline(y[boroughs[i]].to_numpy().mean(), color='r')
    plt.axvline(y[boroughs[i]].to_numpy().max(), color='g')
    plt.axvline(y[boroughs[i]].to_numpy().min(), color='g')

    #plt.axvline(np.quantile(y[boroughs[i]].to_numpy(), q=0.25), color='b')
    #plt.axvline(np.quantile(y[boroughs[i]].to_numpy(), q=0.75), color='b')


    plt.xlabel('Number of crimes')
    plt.title('Simulated distribution and data statistics borough: {}'.format(boroughs[i]))
    plt.savefig('Borough: {}.png'.format(boroughs[i]))
    plt.close()



ybarT = data.groupby('year').size()[2006]/len(boroughs)



data.year.groupby([data.year, data.borough]).size().unstack().fillna(0).astype(int)



for i in range(len(years)):
    
    for k in range(5):
        plt.plot(np.array(Mus)[:,k,i])
        plt.xlabel('Itterations')
        plt.ylabel('Paramaeter Value')
        plt.title("Mu_{}".format(years[i]))
        #plt.ylim([3.5, 5.6])
    plt.savefig("Mu_{}.png".format(years[i]))
    plt.close()
    
    
    
    
    
  
for i in range(len(boroughs)):
    for k in range(5):
        plt.plot(np.array(Thetas)[:,k,i])
        plt.xlabel('Itterations')
        plt.ylabel('Paramaeter Value')
        plt.title("Theta_{}".format(boroughs[i]))
        #plt.ylim([3.5, 5.6])
    plt.savefig("Theta_{}.png".format(boroughs[i]))
    plt.close()  
    
    
    for k in range(5):
        plt.plot(np.array(Sigmas[:,k]))
        plt.xlabel('Itterations')
        plt.ylabel('Paramaeter Value')
        plt.title("sigma")
    #plt.ylim([3.5, 5.6])
    plt.savefig("sigma.png")
    plt.close()  
    
    for k in range(5):
        plt.plot(np.array(Taus[:,k]))
        plt.xlabel('Itterations')
        plt.ylabel('Paramaeter Value')
        plt.title("tau")
    #plt.ylim([3.5, 5.6])
    plt.savefig("tau.png")
    plt.close()  
    
    for k in range(5):
        plt.plot(np.array(Taucs[:,k]))
        plt.xlabel('Itterations')
        plt.ylabel('Paramaeter Value')
        plt.title("tauc")
    #plt.ylim([3.5, 5.6])
    plt.savefig("tauc.png")
    plt.close()  
    
    
    
# Create tables
def compute_MC_W_B(psi):
    '''
    psi: nxm array of parameters for m chains each of length n
    '''
    n, m = psi.shape
    
    psi_bar_j = np.mean(psi, axis = 0)
    sj = np.sum((psi - psi_bar_j[None,:])**2, axis = 0)/(n-1)
    W = np.mean(sj)
    
    psi_bar = np.mean(psi_bar_j)
    B = n/(m-1) * np.sum((psi_bar_j - psi_bar)**2)

    var_hat_plus = (n-1)/n * W + 1/n * B
    Rhat = np.sqrt(var_hat_plus/W)
    
    Vs = []
    for t in range(1, n):
        psi_t1 = psi[t:, :]
        psi_t2 = psi[:-t, :]
        
        psi_df = (psi_t1 - psi_t2)**2
        Vt = np.sum(psi_df)/(m*(n-t))
        
        Vs.append(Vt)
        
    rhos = 1 - np.array(Vs)/(2 * var_hat_plus)
    
    T = np.argmax((rhos[:-1] + rhos[1:] < 0))
    neff = m * n / (1 + 2 * np.sum(rhos[:T+1]))
        
    
    return var_hat_plus, Rhat, W, B, neff
    
data = {'Boroughs': boroughs}


q025 = np.zeros(32)
q250 = np.zeros(32)
q500 = np.zeros(32)
q750 = np.zeros(32)
q975 = np.zeros(32)

Rhats = np.zeros(32)
neffs  = np.zeros(32)


for i in range(len(boroughs)):
    q025[i] = np.quantile(Thetas[:,:,i], q = 0.025)
    q250[i] = np.quantile(Thetas[:,:,i], q = 0.250)
    q500[i] = np.quantile(Thetas[:,:,i], q = 0.500)
    q750[i] = np.quantile(Thetas[:,:,i], q = 0.750)
    q975[i] = np.quantile(Thetas[:,:,i], q = 0.975)

    var_hat_plus, Rhat, W, B, neff = compute_MC_W_B(Thetas[:,:,i])
    
    Rhats[i] = Rhat
    neffs[i] = neff

data['q025'] = q025
data['q250'] = q250
data['q500'] = q500
data['q750'] = q750
data['q975'] = q975
data['Rhat'] = Rhats
data['neff'] = neffs
df = pd.DataFrame (data, columns = ['Boroughs', 'q025', 'q250', 'q500', 'q750', 'q975', 'Rhat', 'neff'])
df.sort_values('q500', ascending = False).to_csv('boroughs.csv')


data = {'Years': years}


q025 = np.zeros(7)
q250 = np.zeros(7)
q500 = np.zeros(7)
q750 = np.zeros(7)
q975 = np.zeros(7)

Rhats = np.zeros(7)
neffs  = np.zeros(7)


for i in range(len(years)):
    q025[i] = np.quantile(Mus[:,:,i], q = 0.025)
    q250[i] = np.quantile(Mus[:,:,i], q = 0.250)
    q500[i] = np.quantile(Mus[:,:,i], q = 0.500)
    q750[i] = np.quantile(Mus[:,:,i], q = 0.750)
    q975[i] = np.quantile(Mus[:,:,i], q = 0.975)

    var_hat_plus, Rhat, W, B, neff = compute_MC_W_B(Mus[:,:,i])
    
    Rhats[i] = Rhat
    neffs[i] = neff

data['q025'] = q025
data['q250'] = q250
data['q500'] = q500
data['q750'] = q750
data['q975'] = q975
data['Rhat'] = Rhats
data['neff'] = neffs
df = pd.DataFrame (data, columns = ['Years', 'q025', 'q250', 'q500', 'q750', 'q975', 'Rhat', 'neff'])
df.to_csv('years.csv')



data = {'Params': ['sigma', 'tau', 'tauc']}


q025 = np.zeros(3)
q250 = np.zeros(3)
q500 = np.zeros(3)
q750 = np.zeros(3)
q975 = np.zeros(3)

Rhats = np.zeros(3)
neffs  = np.zeros(3)

for i in range(3):
    p = [Sigmas, Taus, Taucs][i]
    q025[i] = np.quantile(p[:,:], q = 0.025)
    q250[i] = np.quantile(p[:,:], q = 0.250)
    q500[i] = np.quantile(p[:,:], q = 0.500)
    q750[i] = np.quantile(p[:,:], q = 0.750)
    q975[i] = np.quantile(p[:,:], q = 0.975)
    
    var_hat_plus, Rhat, W, B, neff = compute_MC_W_B(p[:,:])
        
    Rhats[i] = Rhat
    neffs[i] = neff

data['q025'] = q025
data['q250'] = q250
data['q500'] = q500
data['q750'] = q750
data['q975'] = q975
data['Rhat'] = Rhats
data['neff'] = neffs
df = pd.DataFrame (data, columns = ['Params', 'q025', 'q250', 'q500', 'q750', 'q975', 'Rhat', 'neff'])
df.to_csv('params.csv')



plt.boxplot(Mus[:,0,1:], labels = [2006, 2007, 2008, 2009, 2010, 2011])
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Posterior latent distribution by year')
plt.savefig('crime_progression_by_year.png')


for j in range(10):
    for k in range(100, 150):
        plt.plot(Mus[k,j,:])













def WAIC(Mus, Thetas, Sigmas, Taucs, Taus):
    
    SL = np.zeros((1000, np.prod(y.shape)))
    
    for i_s in range(1000):
        m = np.random.choice(10)
        n = np.random.choice(250)
        
        theta = Thetas[n,m,:]
        mui   = Mus[n,m,:]         
        sigma = Sigmas[n,m]
        tauc  = Taucs[n,m]
        tau   = Taus[n,m]
        
        ll = log_likelihood(y, theta, mui, sigma, tauc, tau)
        SL[i_s,:] = ll.ravel()
        
    lppd = np.sum(np.log(np.mean(np.exp(SL), axis = 0)))
    
    pWAIC1 = 2 * (np.sum(np.log(np.mean(np.exp(SL), axis = 0))) - np.sum(np.mean(SL, axis = 0)))
    pWAIC2 = np.sum(np.var(SL, axis = 0))
        
        
    return -2 * (lppd - pWAIC2)
        
        
        


