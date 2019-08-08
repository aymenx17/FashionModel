# FashionModel
Transfer learning with fashion dataset and metadata


### intro

#### Transfer learning

Train a classification model on a relatively large dataset; use the learned embedding to adapt to new classes.
Create methods that learn to generalize when only a small number of samples are available.

##### Steps


- Create master train and test splits of the valid image data, with everything in even years (e.g.
  2010) used for the training set, and everything in an odd year (e.g. 2011) used for the test split.

- Create sub-splits of the training data for pre-training and fine tuning as follows:
  the top 20 classes (top most frequent) - about 3/4 of the data; and all other classes - about 1/4 of the data.

- Train a classifier network using the top 20 classes and report Top-1 Top-5 accuracy you get on each of these
  classes, and the average across classes, using the test split.

- Finetune this network to classify the remaining classes not used so far. Report the
  same Top-1 and Top-5 accuracy figures for each class, and the average across all classes.


### To do

- Training on top 20 classes and report accuracy (Done)
- Finetuning on remaining classes and report accuracy (Done)
- Integrating class weights and focal loss in the cost function (Done)
- Integrating metadata into classification.
- Overall comparison between different techniques.

- Do the previous points on the big dataset.   

Note: I'm currently working with small fashion dataset, which is a small version of the original. And instead of 20 classes
you will notice that I'm using 19, since one class was present in the test set but not in the training set.

### Dataset

The fashion images dataset contains images from an online product catalog.

##### Sample of the csv file

![](https://github.com/aymenx17/detectAnomaly/blob/master/project_imgs/sample_csv.png)

As shown in the figure, each data example correspond to an id. As class label I use the fifth column category type
(right before color column). In addition to categories and subcategories, extra metadata is provided with the dataset as well.  
There are 143 different article Types in this list, some have only a few examples, while others have thousands.



### Results

##### Sample Visualizations

Top 19 class training:
![](https://github.com/aymenx17/detectAnomaly/blob/master/project_imgs/top20_results.png)

Finetuning:
![](https://github.com/aymenx17/detectAnomaly/blob/master/project_imgs/finetune_results.png)
