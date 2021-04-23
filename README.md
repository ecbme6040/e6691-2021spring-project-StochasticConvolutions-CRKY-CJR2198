# e6691-2021Spring-project-StochasticConvolutions-CRKY-CJR2198

Repository for CRKY Project (CJR2198) 'Stochastic Convolutions for High Resolution Image Classification'.

In this project I propose a novel Convolutional Layer named 'Stochastic Convolution' and architectural guidelines to implement the approach into existing lightweight CNN backbones. The approach achieves significant improvements in classification performance in high resolution dermatological image classification over the EfficientNet-B4 model at less than 50% of the model size.

The graph below demonstrates validation performance improvement of the stochastic convolutional approach to high resolution image classification over EfficientNet B0 and B4 base models evaluated using the ISIC skin lesion dataset on images of size 1024x1024 and greater.

![alt text](https://github.com/ecbme6040/e6691-2021spring-project-StochasticConvolutions-CRKY-CJR2198/blob/main/figures/All%20Models%20Compared.jpg?raw=true)


# Directory Organization

## Notebooks

The 'Notebooks' folder contains all Jupyter notebooks used to clean and prepare training data and analyze & aggregate results.

* The 'data_preparation' folder contains 'Data Preparation' for aggregating all image datasets and  'Resize Images' used to resize and crop training images to 1024x1024.
* The 'illustration' folder contains 'Melanoma Data Augmentation Illsutration Notebook' which contains examples of image augmentation used in training.
* The 'results_analysis' folder contains 'Results Analysis Workbook', which contains results analysis and aggreation functions and graph production.

## Model Files 

The main training loop / script can be found in 'TrainScript.py' in the root.

The folder 'model_files' contains all model architecture and layer related code:
 
 * efficientnet_model.py contains class declaration for all model architectures evaluated in this project.
 * layers.py contains derived classes for all layers used in this implementation, including the stochastic convolutional layers.
 * model_architectures.py contains the dictionaries and supporting methods to instantiate the efficientnet model classes.

The folder 'utils' contains the 'utils.py' file which contains the declaration for the custom dataset used for preparation, augmentation and batching of the training images.

## Model Implementation Validation

The 'model_validation' folder contains the 'EfficientNetValidation.py' file with the code used to validate the efficientnet implementation.




