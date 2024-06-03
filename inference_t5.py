import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["WORLD_SIZE"] = "1"
from argument_parser import parse_arguments
#### COMMENT IN TO MERGE PEFT AND BASE MODEL ####
from peft import PeftModel, PeftConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
import torch
import pandas as pd
from peft import AutoPeftModelForCausalLM


# # Load PEFT model on CPU
# model = AutoPeftModelForCausalLM.from_pretrained(
#     args.output_dir,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
# )
# # Merge LoRA and base model and save
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained(args.output_dir,safe_serialization=True, max_shard_size="2GB")


class run():
    def __init__(self,args,checkpoint,mode,device,export):
        self.args = args
        self.checkpoint = checkpoint
        self.mode = mode
        self.device = device
        self.model_id = checkpoint
        self.export = export

        print('Load from checkpoint')
        self.tokenizer, self.model = self.load_model()
        self.model.to(self.device)
       

        if self.mode=='smiles2caption':
            self.prefix = "Translate the following molecule smile string to a descriptive caption: " 
        elif self.mode=='caption2smiles':  
            self.prefix = "Translate the following descriptive caption to a molecule smile string: "   

           

    def inderence(self):
        columns = ['molecule','caption']
        df = pd.read_csv('data/big/test.csv', usecols=columns)

        if self.mode == "smiles2caption":
            columns = ['SMILES', 'ground truth', 'output']
            sources = df['molecule'].tolist()
            targets = df['caption'].tolist()
        elif self.mode=='caption2smiles':  
             
            columns = ['description', 'ground truth', 'output']
            sources = df['caption'].tolist()
            targets = df['molecule'].tolist()

       

        col1 = []
        col2 = []
        col3 = []
        for i in range (len(sources)):
            source = self.prefix + sources[i]    
            target = targets[i]
            
        
            inputs = self.tokenizer.encode(source, return_tensors="pt").to(self.device) 
            outputs = self.model.generate(inputs, max_new_tokens=200)
            predicted = self.tokenizer.decode(outputs[0], skip_special_tokens=True)


            col1.append(source)
            col2.append(target)
            col3.append(predicted)

        data = [col1,col2,col3]
        transposed_data  = list(map(list, zip(*data)))
        output = pd.DataFrame(transposed_data, columns=columns)
        output.to_csv(self.export)


    def load_model(self):    
       
        model = T5ForConditionalGeneration.from_pretrained(self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        return tokenizer, model
    



if __name__ =="__main__":
    args = parse_arguments()   
    #language-plus-molecules/molt5-large-smiles2caption-LPM24
    #language-plus-molecules/molt5-large-caption2smiles-LPM24
    #GT4SD/multitask-text-and-chemistry-t5-base-standard

    checkpoint = "GT4SD/multitask-text-and-chemistry-t5-base-standard"     
    mode = "smiles2caption" #caption2smiles, smiles2caption
    export = "output/big/smiles2caption_multiT5"
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    self = run(args,checkpoint,mode,device,export) 
    self.inderence()