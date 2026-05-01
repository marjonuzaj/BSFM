import numpy as np 
import pandas as pd
import statsmodels.api as sm
import scipy 

class SFM:
    def __init__(self, data: pd.DataFrame, endogenous: str):
        self.df = data
        self.end = endogenous
        self.columns = self.df.columns
        self.exog = [value for value in self.columns if value != self.end]
        self.Y = data[self.end]
        self.X = data[self.exog]
        self.X = sm.add_constant(self.X)
    
    def fit_halfnormal(self, nsim: int, drop:int) -> None:
        self.model_type = 'halfnormal'

        NSIM = nsim

        g0 = 10**(-2)    
        g1 = 10**(-2)                
            
        sigmav_post = 0.1
        sigmau_post = 0.2
        
        
        Y = np.array(self.Y).reshape(-1,1)
        X = np.array(self.X)

        Nobs = Y.shape[0]
        
        u_post = np.abs(np.random.normal(0,sigmau_post,(Nobs,1)))

        beta_post_save = np.zeros((NSIM, X.shape[1]))
        sigmav_post_save = np.zeros((NSIM,1))
        sigmau_post_save = np.zeros((NSIM,1))
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

            # draw sigmau
            shape_sigmau = Nobs / 2 + g0
            scale_sigmau = np.sum(u_post**2)/2 + g1
            sigmau_post = np.sqrt(1/np.random.gamma(shape = shape_sigmau, scale = 1/scale_sigmau, size=1)).item()

            # draw u_post
            mean_u = (-(sigmau_post**2) * (Y - X @ beta_post)) / (sigmau_post**2 + sigmav_post**2)
            sigma_u = np.sqrt((sigmav_post**2) * (sigmau_post**2) / (sigmau_post**2 + sigmav_post**2))
            unif = np.random.uniform(0,1,(Nobs,1))
            c = 1 + scipy.stats.norm.cdf(mean_u/sigma_u) * (unif-1)
            u_post = scipy.stats.norm.ppf(c)*sigma_u + mean_u
            
            
            
            beta_post_save[i,:] = beta_post.flatten()
            sigmav_post_save[i,0] = sigmav_post
            sigmau_post_save[i,0] = sigmau_post
            u_post_save[i,:] = u_post.flatten()


        # drop first {drop} draws
        beta_post_save = beta_post_save[drop:,:]
        sigmav_post_save = sigmav_post_save[drop:,0]
        sigmau_post_save = sigmau_post_save[drop:,0]
        u_post_save = u_post_save[drop:,:]

        self.results = {'beta_post': beta_post_save,
                        'sigmav_post': sigmav_post_save,
                        'sigmau_post': sigmau_post_save}
        
        self.inef_est = u_post_save.mean(axis=0)
            
            
    def summary(self) -> pd.DataFrame:
        
        index_names = ['const'] + self.exog + ['sigmav']
        
        if self.model_type == 'halfnormal':
          extra_params = ['sigmau']
        elif self.model_type == 'exponential':
            extra_params = ['lambda']
        else:
            extra_params = ['mu','sigmau']
          
        index_names = index_names + extra_params
  
        results_df = pd.DataFrame(columns = ['Mean','Std'], index = index_names)

        beta_mean = self.results['beta_post'].mean(axis=0)
        beta_std = self.results['beta_post'].std(axis=0)
        sigmav_mean = self.results['sigmav_post'].mean(axis=0)
        sigmav_std = self.results['sigmav_post'].std(axis=0)

        results_df.loc[['const'] + self.exog, 'Mean'] = beta_mean
        results_df.loc[['const'] + self.exog, 'Std'] = beta_std
        results_df.loc['sigmav', 'Mean'] = sigmav_mean
        results_df.loc['sigmav', 'Std'] = sigmav_std

        for param in extra_params:
          key = f"{param}_post"
          if key in self.results:
            results_df.loc[param, 'Mean'] = self.results[key].mean()
            results_df.loc[param, 'Std'] = self.results[key].std()

        return results_df
        
        


if __name__ == "__main__":
    N = 200
    x1 = np.random.normal(0,1,(N,1))
    x2 = np.random.normal(0,1,(N,1))
    u = np.abs(np.random.normal(0,0.2,(N,1)))
    v = np.random.normal(0,0.2,(N,1))
    y = 1 + 0.5 *x1 + 0.4 * x2 + v -u
    
    df = pd.DataFrame()
    df['y'] = y.flatten()
    df['x1'] = x1.flatten()
    df['x2'] = x2.flatten()

    model = SFM(df,'y')
    
    model.fit_halfnormal(nsim = 5000, drop= 1000)
    model.summary()
