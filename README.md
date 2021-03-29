# Credit_Risk_Analysis
Credit risk analysis using scikit-learn and, imbalanced-learn.

## Overview
The purpose of this analysis was to build and evaluate various machine learning models to evaluate credit risk. The dataset used to train the models was from LendingClub, "a peer-to-peer lending services company." The algorithms used were:
* RandomOverSampler
* SMOTE
* ClusterCentroids
* SMOTEENN
* BalancedRandomForestClassifier
* EasyEnsembleClassifier
The models were run and then evaluated for performance and accuracy at predicting credit risk.

## Results

* Naive Random Oversampling
![Random Oversampling Balanced Accuracy Score](images/ros_bal_acc.png)
![Random Oversampling Imbalanced Classifications Report](images/ros_imbal_class.png)

* SMOTE Oversampling
![SMOTE Oversampling Balanced Accuracy Score](images/smote_bal_acc.png)
![SMOTE Oversampling Imbalanced Classifications Report](images/smote_imbal_class.png)

* Cluster Centroids Undersampling
![Cluster Centroids Undersampling Balanced Accuracy Score](images/ccu_bal_acc.png)
![Cluster Centroids Undersampling Imbalanced Classifications Report](images/ccu_imbal_class.png)

* Combination Sampling
![Combination Sampling Balanced Accuracy Score](images/combsamp_bal_acc.png)
![Combination Sampling Imbalanced Classifications Report](images/combsamp_imbal_class.png)

* Balanced Random Forest Classifier
![Balanced Random Forest Classifier Balanced Accuracy Score](images/brfc_bal_acc.png)
![Balanced Random Forest Classifier Imbalanced Classifications Report](images/brfc_imbal_class.png)

* Easy Ensemble AdaBoost Classifier
![Easy Ensemble AdaBoost Classifier Balanced Accuracy Score](images/eec_bal_acc.png)
![Easy Ensemble AdaBoost Classifier Imbalanced Classifications Report](images/eec_imbal_class.png)

## Summary
