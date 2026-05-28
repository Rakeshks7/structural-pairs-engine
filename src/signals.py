import pandas as pd
import numpy as np
from copulae.core import BaseCopula
from typing import Tuple

class CopulaSignalGenerator:
    
    def __init__(self, fitted_copula: BaseCopula, upper_threshold: float = 0.95, lower_threshold: float = 0.05):
        self.copula = fitted_copula
        self.upper_thresh = upper_threshold
        self.lower_thresh = lower_threshold
        self.epsilon = 1e-4 

    def compute_conditional_probabilities(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        data = np.column_stack((u, v))

        v_plus = np.clip(v + self.epsilon, 0, 1)
        data_v_plus = np.column_stack((u, v_plus))
        p_u_given_v = (self.copula.cdf(data_v_plus) - self.copula.cdf(data)) / self.epsilon

        u_plus = np.clip(u + self.epsilon, 0, 1)
        data_u_plus = np.column_stack((u_plus, v))
        p_v_given_u = (self.copula.cdf(data_u_plus) - self.copula.cdf(data)) / self.epsilon
        
        return p_u_given_v, p_v_given_u

    def generate(self, pseudo_obs: pd.DataFrame) -> pd.DataFrame:
        u = pseudo_obs['U'].values
        v = pseudo_obs['V'].values
        
        p_u_given_v, p_v_given_u = self.compute_conditional_probabilities(u, v)
        
        signals = pd.DataFrame(index=pseudo_obs.index)
        signals['P(U|V)'] = p_u_given_v
        signals['P(V|U)'] = p_v_given_u
        
        signals['Position_X'] = 0
        signals['Position_Y'] = 0
        
        signals.loc[signals['P(U|V)'] > self.upper_thresh, 'Position_X'] = 1
        signals.loc[signals['P(U|V)'] > self.upper_thresh, 'Position_Y'] = -1

        signals.loc[signals['P(U|V)'] < self.lower_thresh, 'Position_X'] = -1
        signals.loc[signals['P(U|V)'] < self.lower_thresh, 'Position_Y'] = 1

        
        return signals