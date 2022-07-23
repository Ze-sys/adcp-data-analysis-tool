
import pandas as pd
class DepthAveragedProfilesStats(object):
    '''
    Class for calculating the statistics of the profiles, to be used by tidal_streamlit.py code.
    INPUT:
    df: dataframe with the profiles
    n1: first index of the profile
    n2: last index of the profile
    OUTPUT:
    u: u profile
    v: v profile
    w: w profile
    u_std: standard deviation of the u profile
    v_std: standard deviation of the v profile
    w_std: standard deviation of the w profile
    u_max: maximum value of the u profile
    v_max: maximum value of the v profile
    w_max: maximum value of the w profile
    u_min: minimum value of the u profile
    v_min: minimum value of the v profile
    w_min: minimum value of the w profile
    u_mean: mean value of the u profile
    v_mean: mean value of the v profile
    w_mean: mean value of the w profile
    u_median: median value of the u profile
    v_median: median value of the v profile
    u_sk: skewness of the u profile
    v_sk: skewness of the v profile
    w_sk: skewness of the w profile
    u_kurt: kurtosis of the u profile
    v_kurt: kurtosis of the v profile
    w_kurt: kurtosis of the w profile
    u_iqr: interquartile range of the u profile
    v_iqr: interquartile range of the v profile
    w_iqr: interquartile range of the w profile

    USAGE:
    >>> df = pd.read_csv('data.csv')
    >>> n1 = 0
    >>> n2 = 100
    >>> dap = DepthAveragedProfilesStats(df,n1,n2)
    >>> dap.u()
    >>> dap.v()
    >>> dap.w()
    >>> dap.u_std()
    >>> dap.v_std()
    >>> dap.w_std()
    etc.
    '''
    def __init__(self,df,n1,n2):
        self.df = df
        self.n1 = n1
        self.n2 = n2
    def u(self):
        u = self.df.iloc[self.n1:self.n2].u.mean()
        return u
    def v(self):
        v = self.df.iloc[self.n1:self.n2].v.mean()
        return v
    def w(self):
        w = self.df.iloc[self.n1:self.n2].w.mean()
        return w
    def u_std(self):
        u_std = self.df.iloc[self.n1:self.n2].u.std()
        return u_std
    def v_std(self):
        v_std = self.df.iloc[self.n1:self.n2].v.std()
        return v_std
    def w_std(self):
        w_std = self.df.iloc[self.n1:self.n2].w.std()
        return w_std
    def u_max(self):
        u_max = self.df.iloc[self.n1:self.n2].u.max()
        return u_max
    def v_max(self):
        v_max = self.df.iloc[self.n1:self.n2].v.max()
        return v_max
    def w_max(self):
        w_max = self.df.iloc[self.n1:self.n2].w.max()
        return w_max
    def u_min(self):
        u_min = self.df.iloc[self.n1:self.n2].u.min()
        return u_min
    def v_min(self):
        v_min = self.df.iloc[self.n1:self.n2].v.min()
        return v_min
    def w_min(self):
        w_min = self.df.iloc[self.n1:self.n2].w.min()
        return w_min
    def u_mean(self):
        u_mean = self.df.iloc[self.n1:self.n2].u.mean()
        return u_mean
    def v_mean(self):
        v_mean = self.df.iloc[self.n1:self.n2].v.mean()
        return v_mean
    def w_mean(self):
        w_mean = self.df.iloc[self.n1:self.n2].w.mean()
        return w_mean
    def u_median(self):
        u_median = self.df.iloc[self.n1:self.n2].u.median()
        return u_median
    def v_median(self):
        v_median = self.df.iloc[self.n1:self.n2].v.median()
        return v_median
    def w_median(self):
        w_median = self.df.iloc[self.n1:self.n2].w.median()
        return w_median
    def u_iqr(self):
        u_iqr = self.df.iloc[self.n1:self.n2].u.quantile(q=[0.25,0.75],)
        return u_iqr
    def v_iqr(self):
        v_iqr = self.df.iloc[self.n1:self.n2].v.quantile(q=[0.25,0.75],)
        return v_iqr
    def w_iqr(self):
        w_iqr = self.df.iloc[self.n1:self.n2].w.quantile(q=[0.25,0.75],)
        return w_iqr
    def u_skew(self):
        u_skew = self.df.iloc[self.n1:self.n2].u.skew()
        return u_skew
    def v_skew(self):
        v_skew = self.df.iloc[self.n1:self.n2].v.skew()
        return v_skew
    def w_skew(self):
        w_skew = self.df.iloc[self.n1:self.n2].w.skew()
        return w_skew
    def u_kurt(self):
        u_kurt = self.df.iloc[self.n1:self.n2].u.kurt()
        return u_kurt
    def v_kurt(self):
        v_kurt = self.df.iloc[self.n1:self.n2].v.kurt()
        return v_kurt
    def w_kurt(self):
        w_kurt = self.df.iloc[self.n1:self.n2].w.kurt()
        return w_kurt
    def all_stats(self):
        all_stats = pd.DataFrame({'u_max':self.u_max(),
                                   'v_max':self.v_max(),
                                   'w_max':self.w_max(),
                                   'u_min':self.u_min(),
                                   'v_min':self.v_min(),
                                   'w_min':self.w_min(),
                                   'u_mean':self.u_mean(),
                                   'v_mean':self.v_mean(),
                                   'w_mean':self.w_mean(),
                                   'u_median':self.u_median(),
                                   'v_median':self.v_median(),
                                   'w_median':self.w_median(),
                                   'u_skew':self.u_skew(),
                                   'v_skew':self.v_skew(),
                                   'w_skew':self.w_skew(),
                                   'u_kurt':self.u_kurt(),
                                   'v_kurt':self.v_kurt(),
                                    },
                                    index=[0])
        return all_stats

    def iqrs(self):
        iqrs = pd.DataFrame({'u_iqr':self.u_iqr(),
                            'v_iqr':self.v_iqr(),
                            'w_iqr':self.w_iqr()}, 
                            )
        return iqrs
                                            
