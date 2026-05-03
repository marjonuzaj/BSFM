import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm



class EXP:
    def __init__(self, data: pd.DataFrame, endogenous: str):
        self.df = data
        self.end = endogenous
        self.columns = self.df.columns
        self.exog = [value for value in self.columns if value != self.end]
        self.Y = data[self.end]
        self.X = data[self.exog]
        self.X = sm.add_constant(self.X)
        self.results = None
        self.inef_est = None
    
    def fit(self, nsim: int, drop:int) -> None:

        NSIM = nsim

        g0 = 10**(-2)    
        g1 = 10**(-2)                
            
        sigmav_post = 0.1
        lambda_post = 5
        
        
        Y = np.array(self.Y).reshape(-1,1)
        X = np.array(self.X)

        Nobs = Y.shape[0]
        
        u_post = np.random.exponential(scale = 1/lambda_post, size=(Nobs,1))

        beta_post_save = np.zeros((NSIM, X.shape[1]))
        sigmav_post_save = np.zeros((NSIM,1))
        lambda_post_save = np.zeros((NSIM,1))
        u_post_save = np.zeros((NSIM,Nobs))

        for i in range(NSIM):
            # draw beta
            beta_mean = np.linalg.inv(X.T @ X) @ (X.T @ (Y+ u_post))
            Sigma = (sigmav_post**2) * np.linalg.inv(X.T @ X)
            beta_post = np.random.multivariate_normal(beta_mean.flatten(), Sigma, 1).reshape(-1,1)

            # draw sigmav
            resid = Y - X @ beta_post + u_post
            shape_sigmav = Nobs / 2 + g0
            scale_sigmav = np.sum(resid**2)/2 + g1
            sigmav_post = np.sqrt(1/np.random.gamma(shape = shape_sigmav, scale = 1/scale_sigmav, size=1)).item()

            # draw lambda
            lambda_post = np.random.gamma(shape=(Nobs +g0), scale=1/(u_post.sum()+g1),size=1)[0]

            # draw u_post
            mean_u = -0.5 * (Y - X @ beta_post - lambda_post*(sigmav_post**2))
            unif = np.random.uniform(0,1,(Nobs,1))
            c = 1 + scipy.stats.norm.cdf(mean_u/sigmav_post) * (unif-1)
            u_post = scipy.stats.norm.ppf(c)*sigmav_post + mean_u
            
            
            beta_post_save[i,:] = beta_post.flatten()
            sigmav_post_save[i,0] = sigmav_post
            lambda_post_save[i,0] = lambda_post
            u_post_save[i,:] = u_post.flatten()


        # drop first {drop} draws
        beta_post_save = beta_post_save[drop:,:]
        sigmav_post_save = sigmav_post_save[drop:,0]
        lambda_post_save = lambda_post_save[drop:,0]
        u_post_save = u_post_save[drop:,:]

        self.results = {'beta_post': beta_post_save,
                        'sigmav_post': sigmav_post_save,
                        'lambda_post': lambda_post_save}
        
        self.inef_est = u_post_save.mean(axis=0)
            
            
    def summary(self) -> pd.DataFrame:
        
        index_names = ['const'] + self.exog + ['sigmav','lambda']
  
        results_df = pd.DataFrame(columns = ['Mean','Std'], index = index_names)

        beta_mean = self.results['beta_post'].mean(axis=0)
        beta_std = self.results['beta_post'].std(axis=0)
        sigmav_mean = self.results['sigmav_post'].mean(axis=0)
        sigmav_std = self.results['sigmav_post'].std(axis=0)
        lambda_mean = self.results['lambda_post'].mean(axis=0)
        lambda_std = self.results['lambda_post'].std(axis=0)

        results_df.loc[['const'] + self.exog, 'Mean'] = beta_mean
        results_df.loc[['const'] + self.exog, 'Std'] = beta_std
        results_df.loc['sigmav', 'Mean'] = sigmav_mean
        results_df.loc['sigmav', 'Std'] = sigmav_std
        results_df.loc['lambda', 'Mean'] = lambda_mean
        results_df.loc['lambda', 'Std'] = lambda_std


        return results_df
        