This repo was created to store the codes used to create the results shown in the paper on the application of transfer learning to forecasting models of arboviral diseases. 

This study is an updated version of this work (available in pre-print): https://www.medrxiv.org/content/10.1101/2020.02.03.20020164v1.full.pdf

In this work, we applied the transfer learning technique to predict the chikungunya cases using a model trained to predict the dengue cases.

Our repo is organized as follows:

* In the `data` folder, we have `.csv` tables with the data used to train and apply the models;

* `compute_cluster.ipynb`: In this notebook is applied a hierarchical clustering techinique to cluster the cities by dengue cases. The cities in the cluster will be used as predictors for the cases of each city in the cluster while applying the models. 

* `download_data_with_copernicus.ipynb`: notebook to download the data from infodengue and format it in the right format for the models. 

* In the `notebooks` folder, we have jupyter notebooks with the results of our analysis. The notebooks are organized by folders: 

- `comp_models.ipynb`: In this notebook we have the code used to create the figures to compare the predictions, the qq plots and compute the skill scores.

- `train_and_apply_models.ipynb`: In this notebook we have the code to apply the following models: 
* train a model on dengue and use it to predict chik using a lstm model;
* apply transfer learning to predict chik using a lstm model;
* train a model on chik and use it to predict chik using a lstm model;
* train a model on dengue and use it to predict chik using a pgbm model;
* train a model on chik and use it to predict chik using a pgbm model;

The notebook above uses the functions saved in the module `train_models.py`. 

- `cities_with_more_cases_by_region.ipynb`: In this notebook there is the code to select the cities with more chik cases in 2023 and that had epidemics in the previous years to apply the transfer learning techinique. 


To apply the models in the cities selected on the notebook above use: 

* `train_dl.py` to train the DL model on dengue, chik and apply the transfer;
* `apply_dl.py` to generate predictions in train and test for the models trained above;
* `train_pgbm.py` to train the PGBM model on dengue, chik and apply the transfer;
* `apply_pgbm.py` to generate predictions in train and test for the models trained above;

* In the `plots` folder, we saved the figures;
* In the `predictions` folder, we save the predictions of the modes in `.pkl` files; 
* In the `saved_models` folder, we save the models trained. 
* In the `lstm.py` file, we have the definition of the functions used to apply the neural networks models used. 
* In the `pgbm_model.py` file, we have the definition of the functions used to apply the probabilistic grandient boosting tree models. 
* In the `plots_lstm.py` file, we have the definition of the functions used to plot the results of the neural networks models used. 
* In the `plots_pgbm.py` file, we have the definition of the functions used to plot the results of the gradient boosting tree models. 
In the `preprocessing.py` file, we have the definition of the functions use to process the data saved in the `data` folder and transform it in the format accepted by the models in `lstm.py` and `pgbm_model.py`. 
