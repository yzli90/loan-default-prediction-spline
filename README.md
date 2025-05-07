
# Loan Default Prediction with Supervised Spline Feature Transformation

This project investigates the use of a supervised, nonlinear feature transformation approach based on cubic spline basis and logistic regression to improve loan default prediction. It is designed to handle nonlinear effects and enhance the information of features.

## Motivation

Accurately predicting whether a loan will be fully paid or charged off is an important task in credit risk modeling. This project utilized a transformation method by [1] where each numerical feature is nonlinearly mapped via cubic spline basis and fitted with a supervised logistic model to produce a probability-like transformed feature. The approach improve the information of a feature and consequently the prediction power of a model.

We compare the performance of models using these transformed features against models using original features across multiple years and classifiers (Logistic Regression and XGBoost).

## Supervised Spline Feature Transformation

Let $x \in \mathbb{R}$ be a numerical feature and $y \in \{0, 1\}$ indicate loan default where .

We apply the following transformation for each feature:

1. **Cubic Spline Basis Expansion**  
   We expand $x$ into a set of spline basis functions $(B_1(x), B_2(x), \ldots, B_K(x))$ with degree $d=3$ (cubic) and with knots placed at empirical quantiles $20\%,40\%,60\%,80\%$ $(K=5)$. Each spline basis function $B_k(x)$ is defined as a cubic polynomial within its corresponding subinterval.
   These basis functions are constructed to ensure $C^2$ continuity: that is, the resulting spline is continuous, and has continuous first and second derivatives across all knot points.

2. **Logistic Regression on Spline Basis of a Single Feature**  
   Fit a logistic model:
   \[
   P(y = 1 \mid x) = \sigma\left( \sum_{k=1}^{K} \beta_k B_k(x) \right)
   \]
   where \( \sigma(z) = \frac{1}{1 + e^{-z}} \)

3. **Transformed Feature**  
   The transformed feature \( \tilde{x} \) is defined as:
   \[
   \tilde{x} := \hat{P}(y = 1 \mid x)
   \]
   This captures both nonlinearity and supervised signal from \( x \).

## Project Pipeline

1. **Data Preparation**

   - LendingClub data (2007–2018) from Kaggle
   - Filter for "Fully Paid" and "Charged Off"
   - Split into `loan_YYYY.csv` and store in `Raw data/`

2. **Feature Engineering & Transformation**

   - Clean, encode, and impute features
   - For each numerical feature:
     - Create cubic spline basis
     - Fit logistic model to predict default
     - Use predicted probability as transformed feature

3. **Feature Selection**

   - 5-fold CV AUC ranking
   - Drop highly collinear features via VIF

4. **Model Training & Evaluation**

   - Compare raw vs transformed features
   - Models: Logistic Regression, XGBoost
   - Train on year \( t \), test on year \( t+1 \)
   - Metrics: AUC and Accuracy

5. **Visualization**

   - Plot transformation curves
   - Visualize in-sample and out-of-sample AUC trends

## Results Summary

- Transformed features generally outperform raw features in out-of-sample AUC.
- Improvement is more consistent with Logistic Regression.
- Transformed features show smoother and more monotonic relations with the outcome variable.

## Future Extensions

- Use AUC-maximizing loss functions (e.g., pairwise logistic loss):
  \[
  \sum_{(i,j): y_i = 1, y_j = 0} \log\left(1 + e^{-(f(x_i) - f(x_j))} \right)
  \]
- Extend to ensemble-based or neural network-based transformation pipelines.
- Incorporate uncertainty estimation or calibration metrics.

## How to Run

1. Download LendingClub data and split into yearly files.
2. Place CSVs into the `Raw data/` folder.
3. Run pipeline scripts:

```bash
python src/data_process.py
python src/features.py
python src/model.py
python src/visualization.py

