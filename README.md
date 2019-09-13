# atypical_doseresponse_classifier
This repo comprises a method to identify atypical dose-response curves using a Convolutional Neural Network trained from simulation data. Most of the implementation is aimed at 7-dose response curves and aim to predict hermetic transition points.

# run 

1. Follow the instructions provided by the [synthetic_doseresponse_generator](https://github.com/nathanieljevans/synthetic_doseresponse_generator) github repo to produce your training data or use the data provided  [here](https://drive.google.com/drive/folders/1bF-OeHiamdALTz2jKnEzqkbnq_rBEBxl?usp=sharing).

2. Run the following scripts (assumes you've cloned the repo)
```bash
# This will preprocess as many chained datasets (produced by the generator above) together and split training/test dataset
python atyp_gen_preprocessing_classifier.py ../../data/synth_set1.csv ../../data/typical_noisy_set.csv 

# This will train a model, use tensorboard to choose the optimal model
python atyp_train_classifier.py

# This step will take a while to package the results into a dataframe, expect 30+ minutes
                                #path/to/model                 #number of plots to display  
python atyp_test_classifier.py ./models/best_model.06-0.09.h5 10
```

This will produce the csv file: `classifier_test_results.csv` in `./python/classifier/`. Using this data, open the Rmd file `./R/atyp_classifier_performance.Rmd` to review performance and choose an optimal cutoff. Be wary of class imbalances. 

# data
Data for training and testing the CNN can be found [here](https://drive.google.com/drive/folders/1bF-OeHiamdALTz2jKnEzqkbnq_rBEBxl?usp=sharing).
    This is a synthetic dataset, prduced using the methods described in [synthetic_doseresponse_generator](https://github.com/nathanieljevans/synthetic_doseresponse_generator) 

# performance 

## Classifier 

This is the model we suggest using. 

## Regression [DEPRECATED]

This model attempts to predict the transition point, it was trained with a class imbalance that renders it non-generalizable. 

Details on how these metrics were caculated can be found in `hermetic_cnn_test.py`.  

> average observation dose point classification accuracy: **89.7%** 

> dose point classification specificity: **73.9%** 

> dose point classfication sensitivity: **99.8%** 

A more in-depth performance analysis can be found in [here](./python/performance_analysis.ipynb). As a general overview, the parameter space regression is shown below. 

![sum1](./figures/ols_sum1.PNG)

![sum2](./figures/ols_sum2.PNG)
