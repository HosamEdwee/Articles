# Example of GARCH

Let’s work through a simple example to see how the GARCH model works in action. To keep things simple, we are going to use a made-up time series dataset for our example.

Let’s say we have the following time series data on the daily price fluctuations of a stock ($Y_t$):

| Day | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| Price | 0.1 | -0.2 | 0.3 | -0.1 | 0.2 |

Now, we want to understand and predict the variations in these fluctuations. For this, we will develop a GARCH(1,1) model, which means we will be using one lag of the past data values (AR term) and one lag of volatility (GARCH term).

Before we dive in, let’s introduce a few variables:
- $Y_t$: The price change at time $t$
- $\sigma^2_t$: The predicted variance (volatility squared) at time $t$
- $\epsilon_t$: The random error term at time $t$
- $\omega$, $\alpha$, $\beta$: GARCH model parameters

The GARCH(1,1) model can be expressed as:

$$
\sigma^2_t = \omega + \alpha(\epsilon^2_{t-1}) + \beta(\sigma^2_{t-1})
$$

Let’s break it down:
- $\omega$: A positive constant that represents the long-term average volatility.
- $\alpha(\epsilon^2_{t-1})$: The impact of the previous day’s squared error term on the current day’s variance, capturing the effect of past data values.
- $\beta(\sigma^2_{t-1})$: The impact of the previous day’s variance on the current day’s variance, capturing the effect of past volatility.

Now, let’s work through an estimation of the GARCH(1,1) model for our dataset. Please note that we would typically use software like R or Python to estimate these models, but we’ll keep it simple for illustrative purposes.

1. Calculate the daily returns’ squared error terms $(Y_t - \text{mean}(Y))^2$:

| Day | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| Error$^2$ | 0.01 | 0.04 | 0.09 | 0.01 | 0.04 |

2. Assume some values for $\omega$, $\alpha$, and $\beta$:

$\omega = 0.01$, $\alpha = 0.1$, $\beta = 0.8$ (these values are chosen arbitrarily for simplicity; actual estimation requires a more advanced technique)

3. Calculate the initial value of the variance for the first time point:

$\sigma^2_1 = \omega = 0.01$ (as we do not have data for the previous day)

4. Now, we can estimate the variance for the next time points using the GARCH(1,1) equation:

$\sigma^2_2 = \omega + \alpha(\epsilon^2_1) + \beta(\sigma^2_1) = 0.01 + 0.1(0.01) + 0.8(0.01) = 0.012$

$\sigma^2_3 = \omega + \alpha(\epsilon^2_2) + \beta(\sigma^2_2) = 0.01 + 0.1(0.04) + 0.8(0.012) \approx 0.0188$

And so on, until we estimate the variance for all time points.

5. Finally, we can calculate the predicted volatility:

Volatility = $\sqrt{\text{variance}}$

These volatility values give us an idea of how much the stock price might fluctuate on each day. Keep in mind, this is a simplified example, and in practice, you would use dedicated software packages to estimate GARCH models and provide more reliable estimations.

Nonetheless, this example should give you a basic understanding of how a GARCH model works and how it can help us predict variability in time series data.
