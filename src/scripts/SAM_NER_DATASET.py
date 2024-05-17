# %%
import sys
print(sys.path)


import torch
import cv2
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np
import os


# %%
from models import NERModel, SamModelPrediction,NERRModel
from segmentation import create_segmented_image
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
ner = NERRModel("../lang-segment-anything/fineTunedT5ForNER")
#ner=NERModel()
sam_predictor = SamModelPrediction()


# %%
# Function to extract entities using Transformer model


def extract_entities(sentence):
    instruction = f" Extract exact words from the sentence that are place, animal, thing or location"
     # Encode the prompt
    prompt = f"""
    1. Sentence: "The cat sat on the mat?" Common Nouns: cat, sat, mat.
    2. Sentence: "He drinks a lot of coffee and reads many books." Common Nouns:  coffee, reads, books.
    3. Sentence: "Sky is blue and grass is green." Common Nouns: sky, blue, grass, green.
    4. Sentence: "Is the  Boy is skateboarding on the wall?" Common Nouns: boy, skateboarding, wall .
    5. Sentence: "What color is teddy bear " Common Nouns: teddy bear, color
    6. Sentence: "Teddy bear is pink ?" Common Nouns: teddy bear, pink
    Sentence: "{sentence}"
    Common Nouns:"""
   
    
    
    # Generate output
    outputs = ner.extract_entities(instruction,prompt)

    if ',' in outputs:
        # Split the string at each comma and strip whitespace
        entity_list = [entity.strip() for entity in outputs.split(',')]
    else:
        # Return the entire string as a single element list, also strip any whitespace
        entity_list = [outputs.strip()]
    return entity_list

 # Function to display image and entities

def display_image_with_sam(image_id, entities, model):
    base_url = "http://images.cocodataset.org/val2014"
    image_url = f"{base_url}/COCO_val2014_{image_id:012d}.jpg"
    
    # Fetch the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')

   
    masks = [model.predict_image(image, ent) for ent in entities]
    
    masks = [item for sublist in masks for item in sublist]
   
    segmentedImageWithEntities = create_segmented_image(image,masks)
  
    # Display the original image and masks


    return segmentedImageWithEntities
 # Create a unique filename using the image ID 
def create_unique_filename(image_id):
   
   
    return f"{image_id}_image.png"

# %%


# %%

file_path = '../dataset/combined_data.json'
with open(file_path, 'r') as file:
    data = json.load(file)
directory = "../segmentedImage"
# Extract questions and image IDs
questions_and_ids_question_id = [(item['question'], item['image_id'],item['question_id']) for item in data['annotations']]

questionsSegmentedImageMapping ={}
questions_idSegmentedImageMapping ={}


for question, image_id,question_id in questions_and_ids_question_id:
    # Extract entities from the question
    entities = extract_entities(question)
    print(entities)
    #extract the entities 
    entities =  [item[0] for item in entities]
    


    name= str(image_id)+"_"+ str(question_id)
    unique_filename = create_unique_filename(name)
    # Generate a segmented image based on the image ID and entities
    segmented_image = display_image_with_sam(image_id, entities, sam_predictor)
    segmented_image.convert('RGB')
      # # Save the segmented image with a unique filename
    segmented_image.save(os.path.join(directory, unique_filename))
  

    
    # # Map the unique image filename to the question
    questionsSegmentedImageMapping[unique_filename] = question
    questions_idSegmentedImageMapping[unique_filename] =question_id

print("done")

# %%


# %%
with open('../dataset/question_image_mapping.json', 'w') as f:
    json.dump(questionsSegmentedImageMapping, f)

with open('../dataset/questionId_image_mapping.json', 'w') as f:
    json.dump(questions_idSegmentedImageMapping, f)


