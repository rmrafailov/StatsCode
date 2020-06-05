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
def log_likelihood(y, theta, mu, sigma):
    T = len(years)
    J = len(theta)
    
    ll = np.zeros((T-1,J))
    for i in range(1,T):
        for j in range(J):
            ll[i-1, j] = st.norm.logpdf(y[boroughs[j]][years[i]], loc = theta[j], scale = sigma)
    return ll

class HNM:
    def __init__(self, years, y):
        self.y = y
        self.years = years
        self.samples = self.sample_posterior(y, years)
        

    #Create conditional distributions
    def sample_mu(self, y, theta, mu, sigma, tau):
        mu = st.norm.rvs(loc = np.mean(theta), scale = tau/np.sqrt(J))
        return mu
            
    def sample_thetaj(self, y, years, theta, mu, sigma, tau, j):
        T = len(years)
        J = len(theta)
        
        nom = mu/tau**2
        for i in range(1,T):
            nom = nom + (1/sigma**2) * (y[boroughs[j]][years[i]])
        denom = 1/tau**2 + T/sigma**2
        
        thetaj= st.norm.rvs(loc = nom/denom, scale = np.sqrt(1/denom))
        return thetaj
    
            
    def sample_sigma(self, y, years, theta, mu, sigma, tau):
        T = len(years)
        J = len(theta)
        
        sigma2 = 0
        for i in range(1,T):
            for j in range(J):
                sigma2 = sigma2 + (y[boroughs[j]][years[i]] - theta[j])**2
        sigma2 = st.invgamma.rvs((T*J)/2, scale = sigma2/2)
        return np.sqrt(sigma2)
    
    def sample_tau(self, y, theta, mu, sigma, tau):
        tau2 = np.sum((theta - mu)**2)/(J-1)
        
        tau2 = st.invgamma.rvs((J-1)/2, scale = tau2/2)
        return np.sqrt(tau2)


    def sample_posterior(self, y, years, n = 250, m = 10):
        
        T = len(years)
        J = 32

        Mus = np.zeros((n, m))
        Thetas = np.zeros((n, m, 32))
        Sigmas = np.zeros((n, m))
        Taus  = np.zeros((n, m))
        LLs = []
        
        
        mu_old = 0
        theta_old = np.random.randn(32)
        sigma_old = 2.0
        tau_old = 5.0
        
        for chain in range(m):
            mu_old = 0
            theta_old = np.random.randn(32)
            sigma_old = 2.0
            tau_old = 5.0
        
            for it in range(n):
                
                mu_new = self.sample_mu(y, theta_old, mu_old, sigma_old, tau_old)
                theta_new = np.zeros(J)
                for j in range(J):
                    theta_new[j] = self.sample_thetaj(y, years, theta_old, mu_old, sigma_old, tau_old, j)
                    
                sigma_new = self.sample_sigma(y, years, theta_old, mu_old, sigma_old, tau_old)
                tau_new =   self.sample_tau(y, theta_old, mu_old, sigma_old, tau_old)
                ll = log_likelihood(y, theta_old, mu_old, sigma_old)
                LLs.append(ll)
            
            
                Mus[it, chain] = mu_new
                Thetas[it, chain,:] = theta_new
                Sigmas[it, chain] = sigma_new
                Taus[it, chain] = tau_new
                
                mu_old = mu_new
                theta_old = theta_new
                sigma_old = sigma_new
                tau_old = tau_new
            
        return Mus, Thetas, Sigmas, Taus
    
    


def CV(params, years, y):
    Mus, Thetas, Sigmas, Taus = params
    y = y.loc[years]
    
    def log_likelihood(y, years, theta, mui, sigma):
        T = len(years)
        J = len(theta)
        
        ll = np.zeros((T,J))
        for i in range(T):
            for j in range(J):
                ll[i, j] = st.norm.logpdf(y[boroughs[j]][years[i]], loc = theta[j], scale = sigma)
        return ll
    
    SL = []
    for i_s in range(1000):
        m = np.random.choice(10)
        n = np.random.choice(250)
        
        theta = Thetas[n,m,:]
        mui   = Mus[n,m,]         
        sigma = Sigmas[n,m]

        SL.append(log_likelihood(y.loc[years], years, theta, mui, sigma))
        
    SL = np.array(SL)
    sl = np.sum(np.log(np.mean(np.exp(SL), axis = 0)))
    
    return sl


def WAIC(y, years, Mus, Thetas, Sigmas, Taus):
    
    def log_likelihood(y, years, theta, mu, sigma):
        T = len(years)
        J = len(theta)
        
        ll = np.zeros((T-1,J))
        for i in range(1,T):
            for j in range(J):
                ll[i-1, j] = st.norm.logpdf(y[boroughs[j]][years[i]], loc = theta[j], scale = sigma)
        return ll

    
    y = y.loc[years[1:]]
    
    SL = np.zeros((1000, np.prod(y.shape)))
    
    theta_bayes = np.zeros_like(Thetas[0,0,:])
    mu_bayes = 0
    sigma_bayes = 0
    
    for i_s in range(1000):
        m = np.random.choice(10)
        n = np.random.randint(100,250)
        
        theta = Thetas[n,m,:]
        mu   = Mus[n,m]         
        sigma = Sigmas[n,m]
        
        ll = log_likelihood(y, years, theta, mu, sigma)
        SL[i_s,:] = ll.ravel()
        
        theta_bayes = theta_bayes + theta
        mu_bayes = mu_bayes + mu
        sigma_bayes = sigma_bayes + sigma
        
    theta_bayes = theta_bayes/1000
    mu_bayes = mu_bayes/1000
    sigma_bayes = sigma_bayes/1000
        
    logl = np.sum(log_likelihood(y, years, theta_bayes, mu_bayes, sigma_bayes))
    pDIC1 = 2 * np.var(np.sum(SL, axis = -1))
    pDIC2 = 2 * (logl - np.mean(np.sum(SL, axis = -1)))
    DIC = -2 * (logl - pDIC2)                      
                          
                          
    lppd = np.sum(np.log(np.mean(np.exp(SL), axis = 0)))
    
    pWAIC1 = 2 * (np.sum(np.log(np.mean(np.exp(SL), axis = 0))) - np.sum(np.mean(SL, axis = 0)))
    pWAIC2 = np.sum(np.var(SL, axis = 0))
    WAIC = -2 * (lppd - pWAIC2)
        
    return logl, pDIC1, pDIC2, DIC, lppd, pWAIC1, pWAIC2, WAIC   
        

model = HNM(np.array([2005, 2006, 2007, 2008, 2009]), y)
llpd2 = CV(model.samples, years[-2:], y)

model = HNM(np.array([2005, 2006, 2007, 2008, 2009, 2010, 2011]), y)
Mus, Thetas, Sigmas, Taus = model.samples
logl, pDIC1, pDIC2, DIC, lppd, pWAIC1, pWAIC2, WAIC  = WAIC(y, years, Mus, Thetas, Sigmas, Taus)


model = HNM(np.array([2005, 2006, 2007]), y)
llpd2007 = CV(model.samples, years[-4:], y)


model = HNM(np.array([2005, 2006, 2007, 2008]), y)
llpd2008 = CV(model.samples, years[-3:], y)


model = HNM(np.array([2005, 2006, 2007, 2008, 2009]), y)
llpd2009 = CV(model.samples, years[-2:], y)


model = HNM(np.array([2005, 2006, 2007, 2008, 2009, 2010]), y)
llpd2010 = CV(model.samples, years[-1:], y)



model = HNM(np.array([2005, 2006, 2007, 2008, 2009]), y)
llpd2 = CV(model.samples, years[-2:], y)