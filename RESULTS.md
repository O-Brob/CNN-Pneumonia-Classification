## Model Performance Results
This section details the diagnostic performance of the trained Convolutional Neural Network on the test subset of the data. The model's classification effectiveness is benchmarked against the ground-truth annotations of the test subset to evaluate its accuracy and reliability in identifying pneumonia from chest radiographs.

## Test Set Evaluation

The model shows good diagnostic performance, achieving a balance between high sensitivity and precision, while maintaining a low false positive rate.

- **Accuracy:** 94.47% (393/416 total cases correctly classified)

- **Recall (Sensitivity):** 93.46% (143/153 pneumonia cases caught)

- **Precision:** 91.67% (143/156 pneumonia predictions were correct)

- **False Positive Rate:** 4.94% (13/263 healthy patients incorrectly flagged with pneumonia)

- **F1-Score:** 92.56%

## Confusion Matrix
Probability scores alone do not represent reality. The four possible outcomes of the binary classifier are presented below, to more clearly present the distribution of positive and negative cases, as well as whether or not these were classified correctly.

|                        |Actual Positive | Actual Negative |
|------------------------|:--------------:|:---------------:|
| **Predicted Positive** | **143** (TP)   | **13** (FP)     |
| **Predicted Negative** | **10** (FN)    | **250** (TN)    |
|