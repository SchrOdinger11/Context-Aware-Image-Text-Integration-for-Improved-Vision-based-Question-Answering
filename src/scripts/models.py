from transformers import pipeline
import torch
import sys
import sys

sys.path.append('/Users/sudhanshu/Desktop/UMASS_COURSES_SEMESTERS/SEM_2/NLP/VITLLMs/src/lang-segment-anything/lang_sam')

import numpy as np
from segment_anything import SamPredictor, sam_model_registry
print(torch.cuda.is_available())
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
#from src/lang-segment-anything/lang_sam/lang_sam.py  import LangSAM
from lang_sam import LangSAM


model_name = "google/flan-t5-base"  # You can choose from different sizes (base, large, etc.)
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)




# class NERModel:
#     def __init__(self):
#         self.ner = pipeline("ner", model=AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER"),tokenizer=AutoTokenizer.from_pretrained("dslim/bert-base-NER") )

#     def extract_entities(self, text):
#         return self.ner(text)



class NERModel:
    def __init__(self):
        self.ner = pipeline("ner", model=AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER"),tokenizer=AutoTokenizer.from_pretrained("dslim/bert-base-NER") )

    def extract_entities(self, text):
        return self.ner(text)
    
class SamModelPrediction:
    def __init__(self):
        
        self.sam = LangSAM("vit_h","/Users/sudhanshu/Desktop/UMASS_COURSES_SEMESTERS/SEM_2/NLP/VITLLMs/src/modelCheckpoints/sam_vit_h_4b8939.pth")
        #self.predictor = SamPredictor(self.sam)

    def predict_image(self, image, prompt):
        self.mask, _, _, _ = self.sam.predict(image, prompt)
        return self.mask
        # self.predictor.set_image(image)  
        # input_box = np.array([100, 100, 400, 400])
        # self.masks, _, _ = self.predictor.predict(prompt) 
        # return self.masks 


class NERRModel:
    def __init__(self, model_path="google/flan-t5-base"):
        self.ner = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)

    
    def extract_entities(self, instruction, text):
        # Formulate the prompt according to FLAN T5 expected format
        prompt = f"{instruction}: {text}"
        
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate the output from the model
        outputs = self.ner.generate(input_ids, max_length=512, num_return_sequences=1)
        
        # Decode the generated ids to text
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    def save_model(self, save_path):
        self.ner.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)