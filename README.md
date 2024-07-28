# MetaOE-L
Python implement of MetaOE-L for few-shot HSI classification.

## File Description
- dataDes/
-   plt_false_color.py: Plotting false color images of hyperspectral data
-   plt_gt.py: draw a label map
- HyperDataAnalysis
-   acc.py: calculate overall accuracy, averaged accuracy, Kappa coefficient and accuracy per class
-   paint_loss.py: draw loss function curves of all the optimizers
- TrainTestSplits
-   TrainTestSplit.py: split the hyperspectral data into training and testing sets
- find_bestlr_normal_HSI.py: find out the best learning rates for human-designed optimizers
- fit_ensemble_test.py: meta-optimizer ensemble for test
- model.py: predictive models
- optimizer2: network structure of meta-optimizers
- 
