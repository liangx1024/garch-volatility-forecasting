# Volatility Forecasting with GARCH

A Python comparison of **Gaussian GARCH(1,1)**, **Student's t GARCH(1,1)**, **rolling variance**, and **EWMA** using SPY daily log returns.

**[View the notebook](notebooks/GARCH_Project.ipynb)** for the mathematics, code, results, and charts.

**[Read the executive summary (PDF)](notebooks/Executive_Summary.pdf)** for a brief overview.

## Method and results

- **Data:** Yahoo Finance adjusted SPY prices, 2010–2025.
- **Training:** 2010–2015; **validation:** 2016–2018; **final fit:** 2010–2018; **testing:** 2019–2025.
- **Evaluation:** one-day-ahead variance forecasts scored using QLIKE and MSE against observed squared shocks.
- **Finding:** Gaussian GARCH has the lowest losses in this sample, narrowly ahead of Student's t GARCH. Statistical significance has not been established.

Squared shocks are noisy volatility proxies. The Student's t comparison is exploratory because it was added after earlier test results were inspected.

## Files

- `notebooks/`: self-contained project report with saved outputs and executive summary PDF.
- `src/`: reusable forecasting, likelihood, and evaluation functions.
- `requirements.txt`: Python dependencies.

## Run

```bash
python -m pip install -r requirements.txt
```

Open `notebooks/GARCH_Project.ipynb`, select your Python environment, and run the cells in order. Internet access is required to download prices; saved results can be viewed without running the code.

## Reference

Bollerslev (1986), [Generalized autoregressive conditional heteroskedasticity](https://doi.org/10.1016/0304-4076(86)90063-1).
