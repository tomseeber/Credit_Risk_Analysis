# Credit Risk Analysis

In this analysis, we are looking to improve the performance of current algorithms to predict the credit risk of a new loan ( high or low risk ) based on our previous loan data.  

## Analysis 
### ([Sampling Algorithms](credit_risk_resampling.ipynb))
We performed an analysis of different sampling techniques with Logistic Regression model to improve the effectiveness of models in determining the credit risk.

The following were the sampling methods we used with the results:

* RandomOverSampler (Oversampling) 
![Random Over Sampler Results](randomOverSampler.png)

* SMOTE (Oversampling) 
![SMOTE Over Sampler Results](SMOTE.png)

* ClusterCentroids ( Undersampling) 
![ClusterCentroids UnderSampler Results](ClusterCentroids.png)

* SMOTEENN ( Under and Over Sampling Combination)
![SMOTEENN OVER and Under sampling Results](SMOTEEN.png)1

### ([Ensembling Algorithms](credit_risk_ensemble.ipynb))
We performed an analysis of different ensemble techniques to determine the credit risk.

The following were the sampling methods we used with the results:

* Balanced Random Forest
![Balanced Random Forest](BalancedRandomForest.png)

* Easy Ensemble AdaBoost Classifier
![Easy Ensemble AdaBoost Classifier](EasyEnsembleAdaBoostClassifier.png)


## Conclusion
 None of the algorithms configured in the current ways are commercially viable to predict the credit risk for a new loan.  Each of the models suffered from extremely low precision for high-risk situations, although Easy Ensemble AdaBoost Classifier did have better Recall (0.92), and marginally improved prevision (0.09)
