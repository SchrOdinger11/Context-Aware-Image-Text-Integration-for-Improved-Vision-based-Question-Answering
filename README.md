



# Context-Aware Image-Text Integration for Improved Vision-based
Exploring
question-aware image-text integration techniques
that prioritize relevant image features in response
to specific queries.


## Installation

Install libraries from requirement.txt . Preferable to create a new conda environment.

Python version for each approach should be <3.11

```bash
pip install -r requirements.txt

```
# Datasets:
A. combined_data1.json: This json is pre-processed version of the complete dataset.

B. filtered_dataForTest.json : This json file contains information of the validation dataset used for making inferences.

# Requirements
A. Create a folder in src directory with the name modelCheckpoints and this checkpoint [SAM MODEL CHECKPOINT ](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

B. Create a folder in src directory with the name segmentedImages
    
# Results of the model

The modelResults folder in src directory contains the csv files which comprise results of all the models. These are namely:

A. inference_qavit_full_image_setting.csv : Results of the QAVIT architecutre on unsegmented images.

B. inference_qavit_segmented_image_setting.csv: Results of the QAVIT architecutre on segmented images. 

C. SegmentedImage_Text_Prompts_PreTrainedLLM.csv: Results of the VLIT architecutre on segmented images. 

  
D. Baseline_OriginalImage_Text_Prompts_PreTrainedLLM.csv: Results of the VLIT architecutre on un-segmented images. 

# Running the Scripts

To run the codes, follow these steps:

A. **Approach 1**: 
    - Run `trainQAVIT.py`

B. **To generate the segmented images**:
    - Run `SAM_NER_DATASET.py`

C. **Approach 2**: 
    - Run `preTrainedVILT.ipynb` on the segmented images from Approach 2

D. **Approach 3**: 
    - To run the mixture approach, run the segmented images on `trainQAVIT.py`

E. **Baseline Model**: 
    - Run unsegmented images on `preTrainedVILT.ipynb`

F. **Model Evaluation**: 
    - Script `eval.py` is used to evaluate the accuracy of the results which are mentioned in the .csv files (taken from the models).



## Authors

- [@Sudhanshu Kulkarni](https://www.github.com/octokatherine)
- [@Dhruvin Gandhi](https://www.github.com/dhruvin5)
- [@Samveg Shah](https://www.github.com/Samveg12)
- [@Dishant Padalia](https://www.github.com/dishant26)


