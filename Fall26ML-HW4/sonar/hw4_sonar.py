"""
HW4 - Problems 2 & 3 on the sonar (scaled) data set.

Interfaces used (official Python bindings from PyPI):
    liblinear-official  -> liblinear.liblinearutil   (logistic regression "-s 0", linear SVM "-s 3")
    libsvm-official     -> libsvm.svmutil            (kernel SVM "-s 0", poly "-t 1", RBF "-t 2")

5-fold cross validation is implemented by hand (fixed random fold assignment,
seed 0) so that both the training error and the validation error can be reported
for every C. The same folds are used for every model. Ties in validation error
are broken in favour of the smaller C (stronger regularization).

Usage:  python hw4_sonar.py      -> prints all tables and writes results.md
"""
import os

import numpy as np
from liblinear.liblinearutil import predict as ll_predict, train as ll_train
from libsvm.svmutil import svm_predict, svm_read_problem, svm_train

HERE = os.path.dirname(os.path.abspath(__file__))
CS = [0.1, 1, 10, 100, 1000]
N_FOLDS = 5
SEED = 0

# name -> (library, base option string). C is appended as "-c <C>".
MODELS = {
    "Logistic regression (liblinear -s 0)": ("liblinear", "-s 0 -q"),
    "Linear SVM (liblinear -s 3)": ("liblinear", "-s 3 -q"),
    "Kernel SVM, polynomial (libsvm -s 0 -t 1)": ("libsvm", "-s 0 -t 1 -q"),
    "Kernel SVM, RBF (libsvm -s 0 -t 2)": ("libsvm", "-s 0 -t 2 -q"),
}


def load():
    y, x = svm_read_problem(os.path.join(HERE, "sonar_scale.txt"))
    read = lambda f: [int(s) - 1 for s in open(os.path.join(HERE, f)).read().split()]  # 1-based
    tr, te = read("sonar-scale-train-indices.txt"), read("sonar-scale-test-indices.txt")
    assert not set(tr) & set(te) and len(set(tr) | set(te)) == len(y)
    pick = lambda idx: ([y[i] for i in idx], [x[i] for i in idx])
    return pick(tr), pick(te)


def fit_predict(lib, opts, C, y_tr, x_tr, eval_sets):
    """Train on (y_tr, x_tr) and return the error rate (%) on each (y, x) in eval_sets."""
    opt = "{} -c {}".format(opts, C)
    if lib == "liblinear":
        m = ll_train(y_tr, x_tr, opt)
        pred = lambda y, x: ll_predict(y, x, m, "-q")[0]
    else:
        m = svm_train(y_tr, x_tr, opt)
        pred = lambda y, x: svm_predict(y, x, m, "-q")[0]
    return [100.0 * np.mean(np.array(pred(y, x)) != np.array(y)) for y, x in eval_sets]


def cross_validate(lib, opts, C, y, x, folds):
    tr_err, va_err = [], []
    for k in range(N_FOLDS):
        tr_idx = np.flatnonzero(folds != k)
        va_idx = np.flatnonzero(folds == k)
        y_tr, x_tr = [y[i] for i in tr_idx], [x[i] for i in tr_idx]
        y_va, x_va = [y[i] for i in va_idx], [x[i] for i in va_idx]
        e_tr, e_va = fit_predict(lib, opts, C, y_tr, x_tr, [(y_tr, x_tr), (y_va, x_va)])
        tr_err.append(e_tr)
        va_err.append(e_va)
    return float(np.mean(tr_err)), float(np.mean(va_err))


def main():
    (y_train, x_train), (y_test, x_test) = load()
    rng = np.random.RandomState(SEED)
    folds = rng.permutation(np.arange(len(y_train)) % N_FOLDS)  # balanced fold sizes (30 each)

    out = ["# Problems 2 & 3: sonar (scaled), {} train / {} test, {}-fold CV\n".format(
        len(y_train), len(y_test), N_FOLDS)]
    summary = []
    for name, (lib, opts) in MODELS.items():
        rows = [(C,) + cross_validate(lib, opts, C, y_train, x_train, folds) for C in CS]
        best_C = min(rows, key=lambda r: (r[2], r[0]))[0]  # lowest val error, then smallest C
        train_err, test_err = fit_predict(lib, opts, best_C, y_train, x_train,
                                          [(y_train, x_train), (y_test, x_test)])
        summary.append((name, best_C, train_err, test_err))

        out += ["## {}\n".format(name),
                "| C | 5-fold avg. training error (%) | 5-fold avg. validation error (%) |",
                "|---|---|---|"]
        for C, e_tr, e_va in rows:
            mark = " **(best)**" if C == best_C else ""
            out.append("| {}{} | {:.2f} | {:.2f} |".format(C, mark, e_tr, e_va))
        out += ["", "Best C = {}; retrained on all {} training examples: "
                "training error = {:.2f}%, **test error = {:.2f}%** ({} of {} test examples misclassified).\n".format(
                    best_C, len(y_train), train_err, test_err,
                    int(round(test_err * len(y_test) / 100)), len(y_test))]

    out += ["## Comparison of test errors\n",
            "| model | best C | training error (%) | test error (%) |", "|---|---|---|---|"]
    for name, C, e_tr, e_te in summary:
        out.append("| {} | {} | {:.2f} | {:.2f} |".format(name, C, e_tr, e_te))

    text = "\n".join(out) + "\n"
    with open(os.path.join(HERE, "results.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
