# Problems 2 & 3: sonar (scaled), 150 train / 58 test, 5-fold CV

## Logistic regression (liblinear -s 0)

| C | 5-fold avg. training error (%) | 5-fold avg. validation error (%) |
|---|---|---|
| 0.1 | 17.50 | 30.00 |
| 1 | 9.33 | 30.00 |
| 10 **(best)** | 4.17 | 29.33 |
| 100 | 0.17 | 29.33 |
| 1000 | 0.00 | 31.33 |

Best C = 10; retrained on all 150 training examples: training error = 5.33%, **test error = 27.59%** (16 of 58 test examples misclassified).

## Linear SVM (liblinear -s 3)

| C | 5-fold avg. training error (%) | 5-fold avg. validation error (%) |
|---|---|---|
| 0.1 | 14.83 | 30.67 |
| 1 | 7.00 | 30.67 |
| 10 | 2.50 | 31.33 |
| 100 | 2.33 | 31.33 |
| 1000 **(best)** | 1.50 | 30.00 |

Best C = 1000; retrained on all 150 training examples: training error = 4.00%, **test error = 31.03%** (18 of 58 test examples misclassified).

## Kernel SVM, polynomial (libsvm -s 0 -t 1)

| C | 5-fold avg. training error (%) | 5-fold avg. validation error (%) |
|---|---|---|
| 0.1 | 44.00 | 44.00 |
| 1 | 43.50 | 43.33 |
| 10 | 16.67 | 23.33 |
| 100 **(best)** | 0.00 | 16.00 |
| 1000 | 0.00 | 18.00 |

Best C = 100; retrained on all 150 training examples: training error = 0.00%, **test error = 12.07%** (7 of 58 test examples misclassified).

## Kernel SVM, RBF (libsvm -s 0 -t 2)

| C | 5-fold avg. training error (%) | 5-fold avg. validation error (%) |
|---|---|---|
| 0.1 | 44.00 | 44.00 |
| 1 | 16.00 | 24.67 |
| 10 | 2.83 | 24.00 |
| 100 **(best)** | 0.00 | 19.33 |
| 1000 | 0.00 | 19.33 |

Best C = 100; retrained on all 150 training examples: training error = 0.00%, **test error = 13.79%** (8 of 58 test examples misclassified).

## Comparison of test errors

| model | best C | training error (%) | test error (%) |
|---|---|---|---|
| Logistic regression (liblinear -s 0) | 10 | 5.33 | 27.59 |
| Linear SVM (liblinear -s 3) | 1000 | 4.00 | 31.03 |
| Kernel SVM, polynomial (libsvm -s 0 -t 1) | 100 | 0.00 | 12.07 |
| Kernel SVM, RBF (libsvm -s 0 -t 2) | 100 | 0.00 | 13.79 |
