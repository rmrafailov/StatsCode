#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:25:54 2020

@author: rafael
"""

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



class CHMM:
    def __init__(self, years, y):
        self.y = y
        self.years = years
        self.samples = self.sample_posterior(y, years)
        
    
    #Compute log-likelihood
    def log_likelihood(self, y, theta, mui, sigma, tau_c, tau):
        T = len(mui)
        J = len(theta)
        
        ll = np.zeros((T-1,J))
        for i in range(1,T):
            for j in range(J):
                ll[i-1, j] = st.norm.logpdf(y[boroughs[j]][years[i]], loc = theta[j] + mui[i], scale = sigma)
        return ll
    
    
    
    #Create conditional distributions
    def sample_mu0(self, y, theta, mui, sigma, tau_c, tau):
        mu0 = st.norm.rvs(loc = mui[1], scale = tau)
        return mu0
    
    def sample_mui(self, y, theta, mui, sigma, tau_c, tau, i):
        T = len(mui)
        J = len(theta)
        
        nom = (1/tau**2) * mui[i-1] + (1/tau**2) * mui[i+1]
        for j in range(J):
            nom = nom + (1/sigma**2) * (y[boroughs[j]][years[i]] - theta[j])
            
        denom =  2/tau**2 + J/sigma**2
        
        mui = st.norm.rvs(loc = nom/denom, scale = np.sqrt(1/denom))
        return mui
    
    def sample_muT(self, y, theta, mui, sigma, tau_c, tau):
        T = len(mui)
        J = len(theta)
        
        nom = (1/tau**2) * mui[T-1]
        for j in range(J):
            nom = nom + (1/sigma**2) * (y[boroughs[j]][years[T-1]] - theta[j])
            
        denom =  1/tau**2 + J/sigma**2
        
        mui = st.norm.rvs(loc = nom/denom, scale = np.sqrt(1/denom))
        return mui  
            
    def sample_thetaj(self, y, theta, mui, sigma, tau_c, tau, j):
        T = len(mui)
        J = len(theta)
        
        nom = 0
        for i in range(1, T):
            nom = nom + (1/sigma**2) * (y[boroughs[j]][years[i]] - mui[i])
        denom = 1/tau**2 + T/sigma**2
        
        thetaj= st.norm.rvs(loc = nom/denom, scale = np.sqrt(1/denom))
        return thetaj
    
            
    def sample_sigma(self, y, theta, mui, sigma, tau_c, tau):
        T = len(mui)
        J = len(theta)
        
        sigma2 = 0
        for i in range(1, T):
            for j in range(J):
                sigma2 = sigma2 + (y[boroughs[j]][years[i]] - theta[j] - mui[i])**2
                
        sigma2 = st.invgamma.rvs((T*J)/2, scale = sigma2/2)
        return np.sqrt(sigma2)
    
    def sample_tauc(self, y, theta, mui, sigma, tau_c, tau):
        T = len(mui)
        J = len(theta)
        
        tauc2 = np.sum(theta**2)#/(J-1)
        
        tauc2 = st.invgamma.rvs((J-1)/2, scale = tauc2/2)
        return np.sqrt(tauc2)
    
    
    def sample_tau(self, y, theta, mui, sigma, tau_c, tau):
        T = len(mui)
        J = len(theta)
        
        tau2 = 0
        for i in range(1,T):
            tau2 = tau2 + (mui[i]-mui[i-1])**2
        #tau2 = tau2/(T-1)
        
        tau2 = st.invgamma.rvs((T-1)/2, scale = tau2/2)
        return np.sqrt(tau2)
    
    def sample_posterior(self, y, years, n = 250, m = 10):
        
        T = len(years)
        J = 32

        print(n)
        print(m)
        
        Mus = np.zeros((n, m, T))
        Thetas = np.zeros((n, m, 32))
        Sigmas = np.zeros((n, m))
        Taucs = np.zeros((n, m))
        Taus  = np.zeros((n, m))
        LLs = []
        
        mui_old = np.zeros(T)
        mui_old[1:] = np.array([y.loc[year].to_numpy().mean() for year in years[1:]])
        mui_old[0] = mui_old[1]
        
        theta_old = np.zeros(32)
        sigma_old = 2.0
        tauc_old = 2.0
        tau_old = 5.0
        
        for chain in range(m): 
            
            mui_old = np.zeros(T)
            mui_old[1:] = np.array([y.loc[year].to_numpy().mean() for year in years[1:]])
            mui_old[0] = mui_old[1]
        
            theta_old = np.zeros(32)
            sigma_old = 2.0
            tauc_old = 2.0
            tau_old = 5.
            
            for it in range(n):
                
                mui_new = np.zeros(T)
                mui_new[0] = self.sample_mu0(y, theta_old, mui_old, sigma_old, tauc_old, tau_old)
                
                for i in range(1,T-1):
                    mui_new[i] = self.sample_mui(y, theta_old, mui_old, sigma_old, tauc_old, tau_old, i)
                mui_new[T-1] = self.sample_muT(y, theta_old, mui_old, sigma_old, tauc_old, tau_old)
                
                theta_new = np.zeros(J)
                for j in range(J):
                    theta_new[j] = self.sample_thetaj(y, theta_old, mui_old, sigma_old, tauc_old, tau_old, j)
                    
                sigma_new = self.sample_sigma(y, theta_old, mui_old, sigma_old, tauc_old, tau_old)
                tauc_new =  self.sample_tauc(y, theta_old, mui_old, sigma_old, tauc_old, tau_old)
                tau_new =   self.sample_tau(y, theta_old, mui_old, sigma_old, tauc_old, tau_old)
                ll = self.log_likelihood(y, theta_new, mui_new, sigma_new, tauc_new, tau_new)
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
        
        return Mus, Thetas, Sigmas, Taucs, Taus
    
    



def CV(params, years, y):
    Mus, Thetas, Sigmas, Taucs, Taus = params
    
    y = y.loc[years]
    
    def log_likelihood(y, years, theta, mui, sigma, tau_c, tau):
        T = len(mui)
        J = len(theta)
        
        ll = np.zeros((T,J))
        for i in range(T):
            for j in range(J):
                ll[i, j] = st.norm.logpdf(y[boroughs[j]][years[i]], loc = theta[j] + mui[i], scale = sigma)
        return ll
    
    SL = []
    for i_s in range(1000):
        m = np.random.choice(10)
        n = np.random.choice(250)
        
        theta = Thetas[n,m,:]
        mui   = Mus[n,m,-1]         
        sigma = Sigmas[n,m]
        tauc  = Taucs[n,m]
        tau   = Taus[n,m]
        
        ll = 0
        
        mu_ni = []
        for i in range(len(years)):
            mu_ni.append(st.norm.rvs(loc = mui, scale = np.sqrt(i+1) * tau))
    
        SL.append(log_likelihood(y.loc[years], years, theta, mu_ni, sigma, tauc, tau))
        
    SL = np.array(SL)
    sl = np.sum(np.log(np.mean(np.exp(SL), axis = 0)))
    
    return sl



def WAIC(y, years, Mus, Thetas, Sigmas, Taucs, Taus):
    

    def log_likelihood(y, years, theta, mui, sigma):
        T = len(years)
        J = len(theta)
        
        ll = np.zeros((T-1,J))
        for i in range(1,T):
            for j in range(J):
                ll[i-1, j] = st.norm.logpdf(y[boroughs[j]][years[i]], loc = theta[j] + mui[i], scale = sigma)
                
        return ll
    
    y = y.loc[years[1:]]
    SL = np.zeros((1000, np.prod(y.shape)))
    
    theta_bayes = np.zeros_like(Thetas[0,0,:])
    mui_bayes = np.zeros_like(Mus[0,0,:])
    sigma_bayes = 0
    
    for i_s in range(1000):
        m = np.random.choice(10)
        n = np.random.choice(250)
        
        theta = Thetas[n,m,:]
        mui   = Mus[n,m,:]         
        sigma = Sigmas[n,m]
        tauc  = Taucs[n,m]
        tau   = Taus[n,m]
        
        ll = log_likelihood(y, years, theta, mui, sigma)
        SL[i_s,:] = ll.ravel()
        
        theta_bayes = theta_bayes + theta
        mui_bayes = mui_bayes + mui
        sigma_bayes = sigma_bayes + sigma
        
    
    theta_bayes = theta_bayes/1000
    mui_bayes = mui_bayes/1000
    sigma_bayes = sigma_bayes/1000
    
    logl = np.sum(log_likelihood(y, years, theta_bayes, mui_bayes, sigma_bayes))
    pDIC1 = 2 * np.var(np.sum(SL, axis = -1))
    pDIC2 = 2 * (logl - np.mean(np.sum(SL, axis = -1)))
    DIC = -2 * (logl - pDIC2)                      
                          
        
    lppd = np.sum(np.log(np.mean(np.exp(SL), axis = 0)))
    
    pWAIC1 = 2 * (np.sum(np.log(np.mean(np.exp(SL), axis = 0))) - np.sum(np.mean(SL, axis = 0)))
    pWAIC2 = np.sum(np.var(SL, axis = 0))
    WAIC = -2 * (lppd - pWAIC2)    
        
    return -2 * logl, pDIC1, pDIC2, DIC, -2 * lppd, pWAIC1, pWAIC2, WAIC
        
model = CHMM(np.array([2005, 2006, 2007, 2008, 2009]), y)
llpd2 = CV(model.samples, years[-2:], y)

model = CHMM(np.array([2005, 2006, 2007, 2008, 2009, 2010, 2011]), y)
Mus, Thetas, Sigmas, Taucs, Taus = model.samples
logl, pDIC1, pDIC2, DIC, lppd, pWAIC1, pWAIC2, WAIC = WAIC(y, years, Mus, Thetas, Sigmas, Taucs, Taus)



model = CHMM(np.array([2005, 2006, 2007]), y)
llpd2007 = CV(model.samples, years[-4:], y)

model = CHMM(np.array([2005, 2006, 2007, 2008]), y)
llpd2008 = CV(model.samples, years[-3:], y)


model = CHMM(np.array([2005, 2006, 2007, 2008, 2009]), y)
llpd2009 = CV(model.samples, years[-2:], y)


model = CHMM(np.array([2005, 2006, 2007, 2008, 2009, 2010]), y)
llpd2010 = CV(model.samples, years[-1:], y)