import pandas as pd


def monthly_amplitude(c):
    d = {}
    d['avgm'] = ((c.iloc[:,0] + c.iloc[:,1])/2).mean()
    return pd.Series(d, index = ['avgm'])
