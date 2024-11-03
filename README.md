# MOE-U
The code for the paper "Few-Shot Hyperspectral Remote Sensing Image Classification
via an Ensemble of Meta-Optimizers with Update Integration". [pdf](https://www.mdpi.com/2072-4292/16/16/2988/pdf)

Citation: Hao, T.; Zhang, Z.; Crabbe, M.J.C. Few-Shot Hyperspectral Remote Sensing Image Classification via an Ensemble of Meta-Optimizers with Update Integration. Remote Sens. 2024, 16, 2988. https://doi.org/10.3390/rs16162988

## File Description
- dataDes/
  - plt_false_color.py: Plotting false color images of hyperspectral data
  - plt_gt.py: draw a label map
- HyperDataAnalysis/
  - acc.py: calculate overall accuracy, averaged accuracy, Kappa coefficient and accuracy per class
  - paint_loss.py: draw loss function curves of all the optimizers
- TrainTestSplits/
  - TrainTestSplit.py: split the hyperspectral data into training and testing sets
- find_bestlr_normal_HSI.py: find out the best learning rates for human-designed optimizers
- fit_ensemble_test.py: meta-optimizer ensemble for test
- model.py: predictive models
- optimizer2.py: network structure of meta-optimizers
- TEST_LSTMO_AVERAGE_HSI.py: test the performance of the MOE-A (meta-optimizer ensemble with the average method)
- TEST_LSTMO_ENSEMBLE_HSI.py: test the performance of the MOE-U (meta-optimizer ensemble with the proposed update integration algorithm)
- TEST_LSTMO_HSI.py: test the performance of the LSTM optimizer
- TEST_normal_HSI.py: test the performance of the humand-designed optimizers
- tools.py: some tools for processing the datasets
- train_LSTMoptimizer.py: train the LSTM optimizer
- train_meta_optimizer_ensemble.py: train a meta-optimizer ensemble
- utils.py: gradients normalization and generating HSI data for training the meta-optimizers

We reference and use some of the code in the following repositories, thanks！
> https://github.com/szubing/ED-DMM-UDA/blob/master/datasets/hyperXDatas.py
> https://github.com/kostagiolasn/MetaLearning-MScThesis/blob/master/MetaLearning.ipynb

