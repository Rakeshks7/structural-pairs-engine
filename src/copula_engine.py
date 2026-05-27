import numpy as np
import pandas as pd
import logging
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
from copulae import ClaytonCopula, GumbelCopula, FrankCopula

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CopulaStatArb:
    
    def __init__(self, asset_x_name: str, asset_y_name: str):
        self.asset_x = asset_x_name
        self.asset_y = asset_y_name
        self.ecdf_x = None
        self.ecdf_y = None
        self.best_copula = None
        self.copula_name = None
        
    def _compute_pseudo_observations(self, returns_x: pd.Series, returns_y: pd.Series) -> pd.DataFrame:
        self.ecdf_x = ECDF(returns_x)
        self.ecdf_y = ECDF(returns_y)
        
        u = self.ecdf_x(returns_x)
        v = self.ecdf_y(returns_y)

        u = np.clip(u, 1e-6, 1 - 1e-6)
        v = np.clip(v, 1e-6, 1 - 1e-6)
        
        return pd.DataFrame({'U': u, 'V': v})

    def fit(self, returns_x: pd.Series, returns_y: pd.Series):
        logging.info(f"Fitting marginals for {self.asset_x} and {self.asset_y}...")
        pseudo_obs = self._compute_pseudo_observations(returns_x, returns_y)

        copulas = {
            'Clayton': ClaytonCopula(dim=2),
            'Gumbel': GumbelCopula(dim=2),
            'Frank': FrankCopula(dim=2)
        }
        
        best_aic = np.inf
        
        for name, copula in copulas.items():
            try:
                copula.fit(pseudo_obs)
                log_lik = copula.log_lik(pseudo_obs)
                aic = 2 * 1 - 2 * log_lik
                
                logging.info(f"{name} Copula fitted. Log-Likelihood: {log_lik:.4f}, AIC: {aic:.4f}")
                
                if aic < best_aic:
                    best_aic = aic
                    self.best_copula = copula
                    self.copula_name = name
                    
            except Exception as e:
                logging.warning(f"Failed to fit {name} Copula: {e}")
                
        logging.info(f"Selected Best Model: {self.copula_name} Copula (AIC: {best_aic:.4f})")
        return self.best_copula

    def generate_signals(self, returns_x: pd.Series, returns_y: pd.Series, upper_thresh=0.95, lower_thresh=0.05) -> pd.DataFrame:
        if self.best_copula is None:
            raise ValueError("Copula not fitted. Call fit() first.")
            
        pseudo_obs = self._compute_pseudo_observations(returns_x, returns_y)
        data = np.column_stack((pseudo_obs['U'], pseudo_obs['V']))

        epsilon = 1e-4

        data_v_plus = np.column_stack((pseudo_obs['U'], np.clip(pseudo_obs['V'] + epsilon, 0, 1)))
        p_u_given_v = (self.best_copula.cdf(data_v_plus) - self.best_copula.cdf(data)) / epsilon

        data_u_plus = np.column_stack((np.clip(pseudo_obs['U'] + epsilon, 0, 1), pseudo_obs['V']))
        p_v_given_u = (self.best_copula.cdf(data_u_plus) - self.best_copula.cdf(data)) / epsilon
        
        signals = pd.DataFrame(index=returns_x.index)
        signals['P(U|V)'] = p_u_given_v
        signals['P(V|U)'] = p_v_given_u
        
        signals['Position_X'] = 0
        signals['Position_Y'] = 0

        signals.loc[signals['P(U|V)'] > upper_thresh, 'Position_X'] = -1
        signals.loc[signals['P(U|V)'] > upper_thresh, 'Position_Y'] = 1

        signals.loc[signals['P(U|V)'] < lower_thresh, 'Position_X'] = 1
        signals.loc[signals['P(U|V)'] < lower_thresh, 'Position_Y'] = -1
        
        return signals