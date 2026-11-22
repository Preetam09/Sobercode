# 2. Regularized Logistic Regression

**Interfaces used:**
- Official liblinear Python interface (`pip install liblinear-official`, `liblinear.liblinearutil`).
- Official libsvm Python interface (`pip install libsvm-official`, `libsvm.svmutil`).

Code: `sonar/hw4_sonar.py`.

## 2.1 Identical attributes give identical weights

$$L(\mathbf w)=C\sum_{i=1}^n \ln\!\left(1+e^{-y_i\mathbf x_i^\top\mathbf w}\right)+\tfrac12\|\mathbf w\|_2^2 .$$

**$L$ has a unique minimizer.**
- Each term $\ln(1+e^{-y_i\mathbf x_i^\top\mathbf w})$ is convex in $\mathbf w$: it is the convex function $t\mapsto\ln(1+e^{-t})$ applied to a linear function of $\mathbf w$.
- $\tfrac12\|\mathbf w\|^2$ is strictly (in fact strongly) convex.
- So $L$ is strictly convex and continuous, and it is coercive since $L(\mathbf w)\ge\tfrac12\|\mathbf w\|^2$. Therefore it has exactly one minimizer $\mathbf w^*$.

**Proof (by symmetry).** Let $\tilde{\mathbf w}$ be $\mathbf w^*$ with the entries $k$ and $l$ swapped: $\tilde w_k=w^*_l$, $\tilde w_l=w^*_k$, and $\tilde w_j=w^*_j$ for $j\ne k,l$.
- Since $x_{i,k}=x_{i,l}$ for every $i$,
  $$\mathbf x_i^\top\tilde{\mathbf w}=\sum_{j\ne k,l}x_{i,j}w^*_j+x_{i,k}w^*_l+x_{i,l}w^*_k=\sum_{j\ne k,l}x_{i,j}w^*_j+x_{i,k}(w^*_k+w^*_l)=\mathbf x_i^\top\mathbf w^* .$$
- So every loss term is unchanged, and $\|\tilde{\mathbf w}\|_2=\|\mathbf w^*\|_2$ because only two coordinates were permuted.
- Hence $L(\tilde{\mathbf w})=L(\mathbf w^*)$, so $\tilde{\mathbf w}$ is also a minimizer.
- By uniqueness, $\tilde{\mathbf w}=\mathbf w^*$. Comparing coordinate $k$ gives $w^*_l=w^*_k$. $\blacksquare$

**Same result from the optimality condition.** Let $\sigma(t)=1/(1+e^{-t})$. Setting $\nabla L(\mathbf w^*)=0$ gives, for every coordinate $j$,
$$\frac{\partial L}{\partial w_j}=-C\sum_{i=1}^n y_i\,x_{i,j}\,\sigma\!\left(-y_i\mathbf x_i^\top\mathbf w^*\right)+w^*_j=0
\;\;\Longrightarrow\;\; w^*_j=C\sum_{i=1}^n y_i\,x_{i,j}\,\sigma\!\left(-y_i\mathbf x_i^\top\mathbf w^*\right).$$
The right-hand side depends on $j$ only through $x_{i,j}$. Since $x_{i,k}=x_{i,l}$ for all $i$, the expressions for $j=k$ and $j=l$ are identical, so $w^*_k=w^*_l$.

The regularizer is what makes this work. Without it, any split of $w_k+w_l$ between the two weights gives the same loss. The $\ell_2$ penalty favors the even split $w_k=w_l$, which has the smallest norm for a fixed sum.

## 2.2 Experiment on sonar (scaled)

**Setup:**
- 208 examples with 60 features. The provided index files (1-based) give 150 training and 58 test examples.
- Training set: 84 negative, 66 positive. Test set: 27 negative, 31 positive.
- Model: liblinear `-s 0` (L2-regularized logistic regression), with no bias term (liblinear's default, `-B -1`). This matches the $L(\mathbf w)$ above.

**5-fold cross validation:**
- The 150 training examples are randomly split (seed 0) into 5 folds of 30.
- For each $C$, the model is trained on 4 folds. Training error is measured on those 4 folds and validation error on the held-out fold. Both are averaged over the 5 folds.
- I wrote the CV loop myself because liblinear's `-v` option only reports validation accuracy.
- Ties go to the smaller $C$.
- The same folds are used for every model in Problem 3.

| C | avg. training error (%) | avg. validation error (%) |
|---|---|---|
| 0.1 | 17.50 | 30.00 |
| 1 | 9.33 | 30.00 |
| **10 (best)** | 4.17 | **29.33** |
| 100 | 0.17 | 29.33 |
| 1000 | 0.00 | 31.33 |

**Best $C = 10$.** C = 10 and C = 100 tie on validation error; the tie goes to the smaller $C$, which is the more regularized model. The final model is retrained with $C = 10$ on all 150 training examples:
- training error 5.33%
- **test error 27.59%** (16 of 58 misclassified)

**Observations:**
- As $C$ grows, regularization weakens and training error falls steadily, from 17.5% to 0%: with 60 features and 120 training points per fold, the data become linearly separable.
- Validation error stays at about 29–31% for every $C$, and is slightly worse at $C=1000$, where the model overfits.
- So a linear model's generalization on sonar depends little on $C$.

# 3. Support Vector Machine

All models use the same train/test split, the same 5-fold assignment, and $C\in\{0.1,1,10,100,1000\}$. Kernel parameters are libsvm defaults:
- polynomial: $(\gamma\,\mathbf x^\top\mathbf x'+0)^3$ with $\gamma=1/60$
- RBF: $\exp(-\gamma\|\mathbf x-\mathbf x'\|^2)$ with $\gamma=1/60$

## 3.1 Linear SVM (liblinear `-s 3`, L2-regularized L1-loss SVM, dual)

| C | avg. training error (%) | avg. validation error (%) |
|---|---|---|
| 0.1 | 14.83 | 30.67 |
| 1 | 7.00 | 30.67 |
| 10 | 2.50 | 31.33 |
| 100 | 2.33 | 31.33 |
| **1000 (best)** | 1.50 | **30.00** |

**Best $C = 1000$.** Retrained on all training data: training error 4.00%, **test error 31.03%** (18 of 58).

Training error does not quite reach 0 at large $C$ even though the folds are almost separable. For $C\ge100$, liblinear's dual solver stops at its default iteration limit before fully converging; it prints `WARNING: reaching max number of iterations`.

## 3.2 Kernel SVM (libsvm `-s 0`)

**Polynomial kernel (`-t 1`):**

| C | avg. training error (%) | avg. validation error (%) |
|---|---|---|
| 0.1 | 44.00 | 44.00 |
| 1 | 43.50 | 43.33 |
| 10 | 16.67 | 23.33 |
| **100 (best)** | 0.00 | **16.00** |
| 1000 | 0.00 | 18.00 |

**Best $C = 100$.** Retrained: training error 0.00%, **test error 12.07%** (7 of 58).

**RBF kernel (`-t 2`):**

| C | avg. training error (%) | avg. validation error (%) |
|---|---|---|
| 0.1 | 44.00 | 44.00 |
| 1 | 16.00 | 24.67 |
| 10 | 2.83 | 24.00 |
| **100 (best)** | 0.00 | **19.33** |
| 1000 | 0.00 | 19.33 |

**Best $C = 100$.** C = 100 and C = 1000 tie, and the tie goes to the smaller $C$. Retrained: training error 0.00%, **test error 13.79%** (8 of 58).

## 3.3 Comparison of test errors

| model | best C | training error (%) | **test error (%)** |
|---|---|---|---|
| Logistic regression (liblinear `-s 0`) | 10 | 5.33 | 27.59 |
| Linear SVM (liblinear `-s 3`) | 1000 | 4.00 | 31.03 |
| Kernel SVM, polynomial (libsvm `-t 1`) | 100 | 0.00 | **12.07** |
| Kernel SVM, RBF (libsvm `-t 2`) | 100 | 0.00 | 13.79 |

**Discussion:**

1. **Kernel SVMs are much better.** Both kernel SVMs roughly halve the test error of the two linear models: 12–14% versus 28–31%, and 16% / 19% versus 29–30% in cross validation. The two classes of sonar returns (rock vs. mine) are not well separated by a hyperplane in the original 60-dimensional space. The polynomial and RBF kernels allow a nonlinear decision boundary, and that generalizes much better.

2. **Logistic regression and the linear SVM are about equally good.** They differ by only 2 of 58 test examples, 27.6% versus 31.0% (each test example is worth 1.7%). Their cross-validation errors are within about 1 point of each other (29.3% vs 30.0%). This is expected: both are L2-regularized linear classifiers, and their losses (logistic vs. hinge) behave similarly on this data. The difference is within the noise of a 58-example test set.

3. **The value of C matters much more for the kernel SVMs.**
   - **Small C underfits.** With $C\le1$, both kernel SVMs have 44% training *and* validation error. That is exactly the share of positive examples in the training set (66 of 150), so the SVM predicts the majority class −1 everywhere.
   - **Why.** With the default $\gamma=1/60$, the polynomial kernel values $(\mathbf x^\top\mathbf x'/60)^3$ are small, so a large $C$ (weak regularization) is needed before the margin term lets the classifier fit anything.
   - **Large C works well.** At $C=100$ both kernel SVMs fit the training data perfectly (0% training error) and still give the lowest validation error. The kernel's built-in smoothness provides enough implicit regularization, and validation error rises only slightly at $C=1000$.
   - **The linear models are flat in C.** Their validation error barely changes with $C$; their limitation is the linear model itself, not the amount of regularization.

4. **About the test errors.** The kernel SVMs' test errors are below their CV errors, at 12% vs 16% and 14% vs 19%. With only 58 test examples, these numbers have a standard error of about ±4–5%. All four models are ranked the same way by cross validation and by test error.
