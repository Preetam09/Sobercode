# Problem 5 results (seed 123, 15 epochs)

## Hyper-parameter search (final-epoch scores)

| method | config | val AUROC | val AUPRC | test AUROC | test AUPRC |
|---|---|---|---|---|---|
| APLoss + SOAP (Adam) | {'lr': 0.0005, 'gamma': 0.2} | 0.9938 | 0.9980 | 0.9725 | 0.9824 |
| APLoss + SOAP (Adam) | {'lr': 0.0005, 'gamma': 0.5} | 0.9939 | 0.9979 | 0.9745 | 0.9834 |
| APLoss + SOAP (Adam) | {'lr': 0.0005, 'gamma': 0.8} | 0.9935 | 0.9977 | 0.9750 | 0.9838 |
| APLoss + SOAP (Adam) | {'lr': 0.001, 'gamma': 0.2} | 0.9931 | 0.9977 | 0.9734 | 0.9828 |
| APLoss + SOAP (Adam) | {'lr': 0.001, 'gamma': 0.5} | 0.9931 | 0.9976 | 0.9743 | 0.9836 |
| APLoss + SOAP (Adam) | {'lr': 0.001, 'gamma': 0.8} | 0.9950 | 0.9984 | 0.9744 | 0.9832 |
| APLoss + SOAP (Adam) | {'lr': 0.002, 'gamma': 0.2} | 0.9948 | 0.9983 | 0.9696 | 0.9804 |
| APLoss + SOAP (Adam) | {'lr': 0.002, 'gamma': 0.5} | 0.9930 | 0.9976 | 0.9708 | 0.9815 |
| APLoss + SOAP (Adam) | {'lr': 0.002, 'gamma': 0.8} | 0.9930 | 0.9976 | 0.9691 | 0.9800 |
| AUCMLoss + PESG (Adam) | {'lr': 0.02} | 0.9869 | 0.9956 | 0.9536 | 0.9690 |
| AUCMLoss + PESG (Adam) | {'lr': 0.05} | 0.9830 | 0.9940 | 0.9408 | 0.9624 |
| AUCMLoss + PESG (Adam) | {'lr': 0.1} | 0.9673 | 0.9881 | 0.9171 | 0.9381 |
| CE + Adam | {'lr': 0.0005} | 0.9965 | 0.9988 | 0.9785 | 0.9854 |
| CE + Adam | {'lr': 0.001} | 0.9958 | 0.9986 | 0.9768 | 0.9846 |
| CE + Adam | {'lr': 0.002} | 0.9958 | 0.9986 | 0.9811 | 0.9877 |

## Selected configurations (chosen on validation set)

| method | selected by | config | train AUROC | train AUPRC | test AUROC | test AUPRC | best test AUROC (ep) | best test AUPRC (ep) |
|---|---|---|---|---|---|---|---|---|
| APLoss + SOAP (Adam) | val AUPRC | {'lr': 0.001, 'gamma': 0.8} | 0.9962 | 0.9988 | 0.9744 | 0.9832 | 0.9744 (14) | 0.9832 (14) |
| AUCMLoss + PESG (Adam) | val AUROC | {'lr': 0.02} | 0.9856 | 0.9953 | 0.9536 | 0.9690 | 0.9557 (13) | 0.9703 (13) |
| CE + Adam | val AUROC | {'lr': 0.0005} | 0.9982 | 0.9994 | 0.9785 | 0.9854 | 0.9785 (14) | 0.9854 (14) |
