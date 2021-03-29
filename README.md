# Credit_Risk_Analysis
Credit risk analysis using scikit-learn and imbalanced-learn on sample credit data.

## Overview
The purpose of this analysis was to build and evaluate various machine learning models to evaluate individual customer credit risk. The dataset used to train the models was from LendingClub, "a peer-to-peer lending services company." The algorithms used were:
* RandomOverSampler
* SMOTE
* ClusterCentroids
* SMOTEENN
* BalancedRandomForestClassifier
* EasyEnsembleClassifier

The models were run and then evaluated for performance and accuracy at predicting credit risk.

## Results
In examining the results we will look at the Balanced Accuracy Score as well as the Imbalanced Classification Report (ICR) from each model. Of particular interest in the ICR are two figures from the "f1" (F-score) column - the number from the bottom "avg / total" row, as well as the f-score from the "high risk" row, since we're primarily interested in our ability to detect high credit risk individuals.

* Naive Random Oversampling
    * ![Random Oversampling Balanced Accuracy Score](images/ros_bal_acc.png)
    * ![Random Oversampling Imbalanced Classifications Report](images/ros_imbal_class.png)

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
