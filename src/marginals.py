import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from typing import Tuple

class MarginalModel:
    
    def __init__(self):
        self.ecdf_x = None
        self.ecdf_y = None
        self.epsilon = 1e-6 

    def fit_transform(self, returns_x: pd.Series, returns_y: pd.Series) -> pd.DataFrame:
        self.ecdf_x = ECDF(returns_x)
        self.ecdf_y = ECDF(returns_y)
        
        return self.transform(returns_x, returns_y)

    def transform(self, returns_x: pd.Series, returns_y: pd.Series) -> pd.DataFrame:
        if self.ecdf_x is None or self.ecdf_y is None:
            raise ValueError("ECDFs not fitted. Call fit_transform first.")
            
        u = self.ecdf_x(returns_x)
        v = self.ecdf_y(returns_y)

        u = np.clip(u, self.epsilon, 1 - self.epsilon)
        v = np.clip(v, self.epsilon, 1 - self.epsilon)
        
        return pd.DataFrame({'U': u, 'V': v}, index=returns_x.index)