# e6691-2021Spring-project-StochasticConvolutions-CRKY-CJR2198

Repository for CRKY Project (CJR2198) 'Stochastic Convolutions for High Resolution Image Classification'.

In this project I propose a novel Convolutional Layer named 'Stochastic Convolution' and architectural guidelines to implement the approach into existing lightweight CNN backbones. The approach achieves significant improvements in classification performance in high resolution dermatological image classification over the EfficientNet-B4 model at less than 50% of the model size.

The graph below demonstrates validation performance improvement of the stochastic convolutional approach to high resolution image classification over EfficientNet B0 and B4 base 
models evaluated using the ISIC skin lesion dataset on images of size 1024x1024 and greater.

# Key Results

![alt text](https://github.com/ecbme6040/e6691-2021spring-project-StochasticConvolutions-CRKY-CJR2198/blob/main/figures/ValidationResults.png?raw=true)

![alt text](https://github.com/ecbme6040/e6691-2021spring-project-StochasticConvolutions-CRKY-CJR2198/blob/main/figures/All%20Models%20Compared.jpg?raw=true)


# Directory Organization

## Notebooks

The 'Notebooks' folder contains all Jupyter notebooks used to clean and prepare training data and analyze & aggregate results.

* The 'data_preparation' folder contains 'Data Preparation' for aggregating all image datasets and  'Resize Images' used to resize and crop training images to 1024x1024.
* The 'illustration' folder contains 'Melanoma Data Augmentation Illsutration Notebook' which contains examples of image augmentation used in training.
* The 'results_analysis' folder contains 'Results Analysis Workbook', which contains results analysis and aggreation functions and graph production.

## Model Files 

All model files can be files can be found within the 'Project Code' sub-directory.

The main training loop / script can be found in 'TrainScript.py'.

The folder 'model_files' contains all model architecture and layer related code:
 
 * efficientnet_model.py contains class declaration for all model architectures evaluated in this project.
 * layers.py contains derived classes for all layers used in this implementation, including the stochastic convolutional layers.
 * model_architectures.py contains the dictionaries and supporting methods to instantiate the efficientnet model classes.

The folder 'utils' contains the 'utils.py' file which contains the declaration for the custom dataset used for preparation, augmentation and batching of the training images.

## Model Implementation Validation

The 'model_validation' folder contains the 'EfficientNetValidation.py' file with the code used to validate the efficientnet implementation.

## Figures

The 'figures' folder contains a sampling of graphed model results.

## Results

The 'results' folder contains the collected per-epoch results for the model training runs, organized by base model type (B0 or B4). 'Base' folders are the base model runs, the other runs are named corresponding to the model type 'full split','stochastic stem' etc.


## Directory Treee

'''
+---figures
|       All Models Compared.jpg
|       b0FullSplitVsBaseSpreadFull.jpg
|       b0OfficialvsProjectSpread.jpg
|       b0StochsaticSkipVsBaseSpreadFull.jpg
|       b0StochsaticStemVs32BaseSpread.jpg
|       b0StochsaticStemVs64BaseSpread.jpg
|       b0TTAFullSplitStochastic.jpg
|       b4OfficialvsProjectSpread.jpg
|       b4StochsaticStemVs48BaseSpread.jpg
|       ValidationResults.png
|       
+---Notebooks
|   +---data_preparation
|   |       Data Preparation.ipynb
|   |       Resize Images.ipynb
|   |       
|   +---illustration
|   |       Melanoma Data Augmentation Illsutration Notebook.ipynb
|   |       
|   \---results_analysis
|           Results Analysis Workbook.ipynb
|           
+---Project Code
|   |   TrainScript.py
|   |   
|   +---model_files
|   |   |   efficientnet_model.py
|   |   |   layers.py
|   |   |   model_architectures.py
|   |   |   
|   |   \---__pycache__
|   |           layers.cpython-37.pyc
|   |           model_architectures.cpython-37.pyc
|   |           
|   +---model_validation
|   |       EfficientNetValidation.py
|   |       
|   \---utils
|           utils.py
|           
\---results
    +---B0
    |   +---base
    |   |       base10_per_epoch.csv
    |   |       base11_per_epoch.csv
    |   |       base12_per_epoch.csv
    |   |       base13_per_epoch.csv
    |   |       base14_per_epoch.csv
    |   |       base1_per_epoch.csv
    |   |       base2_per_epoch.csv
    |   |       base3_per_epoch.csv
    |   |       base4_per_epoch.csv
    |   |       base5_per_epoch.csv
    |   |       base6_per_epoch.csv
    |   |       base7_per_epoch.csv
    |   |       base8_per_epoch.csv
    |   |       base9_per_epoch.csv
    |   |       
    |   +---base 64 input channels
    |   |       b0_base_64_channel0_per_epoch.csv
    |   |       b0_base_64_channel1_per_epoch.csv
    |   |       b0_base_64_channel2_per_epoch.csv
    |   |       b0_base_64_channel3_per_epoch.csv
    |   |       b0_base_64_channel4_per_epoch.csv
    |   |       
    |   +---full split
    |   |       b0_fullsplit0_per_epoch.csv
    |   |       b0_fullsplit1_per_epoch.csv
    |   |       b0_fullsplit2_per_epoch.csv
    |   |       b0_fullsplit3_per_epoch.csv
    |   |       b0_fullsplit4_per_epoch.csv
    |   |       b0_fullsplit5_per_epoch.csv
    |   |       b0_fullsplit6_per_epoch.csv
    |   |       b0_fullsplit7_per_epoch.csv
    |   |       
    |   +---full split tta
    |   |       fs_tta0_per_epoch.csv
    |   |       fs_tta1_per_epoch.csv
    |   |       fs_tta2_per_epoch.csv
    |   |       
    |   +---stochastic skip
    |   |       b0_stochasticskip_640_per_epoch.csv
    |   |       b0_stochasticskip_641_per_epoch.csv
    |   |       b0_stochasticskip_642_per_epoch.csv
    |   |       b0_stochasticskip_643_per_epoch.csv
    |   |       b0_stochasticskip_644_per_epoch.csv
    |   |       
    |   \---stochastic stem
    |           b0_stochastic_640_per_epoch.csv
    |           b0_stochastic_641_per_epoch.csv
    |           b0_stochastic_642_per_epoch.csv
    |           b0_stochastic_643_per_epoch.csv
    |           b0_stochastic_644_per_epoch.csv
    |           b0_stochastic_645_per_epoch.csv
    |           
    \---B4
        +---base
        |       b4_base0_per_epoch.csv
        |       b4_base1_per_epoch.csv
        |       b4_base2_per_epoch.csv
        |       b4_base3_per_epoch.csv
        |       b4_base4_per_epoch.csv
        |       
        \---stochastic_stem
                b4_stochastic0_per_epoch.csv
                b4_stochastic1_per_epoch.csv
                b4_stochastic2_per_epoch.csv
                b4_stochastic3_per_epoch.csv
                b4_stochastic4_per_epoch.csv
'''

