import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["WORLD_SIZE"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline,GenerationConfig,T5ForConditionalGeneration
import pandas as pd



class negative():
    def __init__(self, mode, subset,device):
        self.mode = mode
        self.subset = subset
        self.device = device

        #Load data
        columns = ['molecule', 'caption']
        if self.subset == "train":
            self.data = pd.read_csv("data/train.csv", usecols=columns) 
        elif self.subset == "val":
            self.data = pd.read_csv("data/val.csv", usecols=columns) 
        # self.data = self.data.sample(frac=0.1, replace=False)    
    
        if self.mode=='smiles2caption':
                model_id = "language-plus-molecules/molt5-large-smiles2caption-LPM24"  
                self.prefix = "Translate the following molecule smile string to a descriptive caption: "
        elif self.mode=='caption2smiles':  
                model_id = "language-plus-molecules/molt5-large-caption2smiles-LPM24"  
                self.prefix = "Translate  the following descriptive caption to a molecule smile string: "   


        self.model = T5ForConditionalGeneration.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
         

        self.model.to(self.device)
       
        
       

    def generate_negative(self):

        df = pd.DataFrame(columns=['source','chosen', 'rejected'])
         
        for index, row in self.data.iterrows():
                if self.mode=='smiles2caption':
                    #  source = self.prefix+ row['molecule']
                     source = row['molecule']
                     target = row['caption']
                elif self.mode=='caption2smiles':
                    #  source = self.prefix+ row['caption']
                     source =   row['caption']
                     target = row['molecule']     

                inputs = self.tokenizer.encode(source, return_tensors="pt").to(self.device)   
                outputs = self.model.generate(inputs, max_new_tokens=250)  
                predicted = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                df.loc[len(df)] = pd.Series({'source':source, 'chosen':target, 'rejected':predicted})

        df.to_csv(os.path.join("data/all",  "_".join([self.subset, self.mode, '.csv'])))
               



        



if __name__=="__main__":
    mode = "smiles2caption" #caption2smiles, smiles2caption
    subset = "val"
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    self = negative(mode, subset,device)
    self.generate_negative()


# def merge(caption2smiles,smiles2caption):
#      columns = ['source', 'chosen', 'rejected']
#      df_1  = pd.read_csv(caption2smiles, usecols=columns)
#      df_2  = pd.read_csv(smiles2caption, usecols=columns)
#      df = pd.concat([df_1,df_2], axis=0)

#      df.to_csv(os.path.join("data/big", "_".join(["dpo", caption2smiles.split("_")[0].split("/")[2], '.csv'])))
    
  
     
     
# caption2smiles = "data/big/train_caption2smiles_.csv"    
# smiles2caption = "data/big/train_smiles2caption_.csv"

# # #Create test    
# import numpy as np    
# def test(path_all,path_train):
#     columns = ['molecule', 'caption']
#     df_all  = pd.read_csv(path_all, usecols=columns)

#     df_train  = pd.read_csv(path_train)
#     #excluded_indices = df_train['Unnamed: 0'].to_list()
#     excluded_indices = df_train['chosen'].to_list()

   

#     filtered_indices = [idx for idx in df_all.index if idx not in excluded_indices]
#     random_indices = np.random.choice(filtered_indices, size=3000, replace=False)

#     text_data = df_all.loc[random_indices]
#     text_data.to_csv('data/test_.csv')

# path_all= "data/val.csv"
# path_train= "data/big/dpo_val_.csv"     


# from datasets import load_dataset
# import pandas as pd

# dataset = load_dataset("language-plus-molecules/LPM-24_train-extra")
# val = dataset['split_valid']
# df = val.to_pandas()


# data = df.groupby(['molecule'], as_index=False).agg(
#             caption = ('caption', 'first')
#             ) 

# random_samples = data.sample(n=3000)

# random_samples.to_csv(os.path.join("data/big",  "test.csv"))


# from datasets import load_dataset

# dataset = load_dataset("liupf/ChEBI-20-MM")
# test = dataset['test']
# test = test.to_pandas()
# test = test[['SMILES', 'description']]
# test = test.rename(columns={'SMILES': 'molecule', 'description': 'caption'})

# test.to_csv(os.path.join("data/big",  "test2.csv"))