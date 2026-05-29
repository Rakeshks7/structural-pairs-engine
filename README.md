# Copula Statistical Arbitrage Engine

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

An institutional-grade statistical arbitrage engine that shifts pairs trading from linear assumptions (Pearson correlation) to structural dependency modeling. 

Standard Z-score pairs trading often fails during market crashes because correlation goes to 1 in a panic. This engine utilizes **Copulas (Clayton, Gumbel, Frank)** to model *tail dependence* the probability that Asset A dislocates given Asset B's movement—ensuring the strategy remains robust during extreme market regime shifts.

## Core Features

*   **Marginal Distribution Modeling:** Uses Empirical Cumulative Distribution Functions (ECDF) to transform log returns into uniform margins, avoiding parametric misspecification of notoriously non-normal asset returns.
*   **Dynamic Copula Fitting:** Evaluates Archimedean copulas and selects the optimal structural fit using the Akaike Information Criterion (AIC).
*   **Conditional Probability Signals:** Generates buy/sell signals based on numerical differentiation of the Copula CDF ($P(U \le u | V = v)$), identifying true probabilistic mispricings.
*   **Vectorized Backtesting:** Includes a frictionless, vectorized backtesting environment that accounts for continuous compounding, capital allocation, and transaction costs.

## Mathematical Framework
This engine trades the conditional probability:
$$ P(U \le u | V = v) = \frac{\partial C(u,v)}{\partial v} $$
Where $C(u,v)$ is the fitted Copula function mapping the uniform marginals of the two assets. Extreme values (e.g., $<0.05$ or $>0.95$) indicate a breakdown in the historical structural relationship, triggering a mean-reverting entry.

## Disclaimer

For Educational and Research Purposes Only.
The code and strategies provided in this repository do not constitute financial advice, investment recommendations, or an offer to buy or sell any securities. Trading financial instruments involves significant risk of loss. The author assumes no responsibility for any trading losses incurred by using this software. Always conduct your own due diligence and consult with a licensed financial advisor before making trading decisions.