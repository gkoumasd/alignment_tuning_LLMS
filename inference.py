import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["WORLD_SIZE"] = "1"
from argument_parser import parse_arguments
#### COMMENT IN TO MERGE PEFT AND BASE MODEL ####
from peft import PeftModel, PeftConfig
from transformers import  AutoTokenizer,GenerationConfig
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
        model_id = "language-plus-molecules/Meditron7b-caption2smiles-LPM24"
       

        self.export = export

        print('Load from checkpoint')
        self.tokenizer, self.model = self.load_model()
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.to(self.device)
       

        if self.mode=='smiles2caption':
            self.prefix = "Translate to a descriptive caption for the given SMILES:" 
        elif self.mode=='caption2smiles':  
            self.prefix = "Translate to SMILES for the given caption:"   

        self.generation_config = GenerationConfig.from_pretrained(model_id,best_of=1,
                presence_penalty=0.0,
                frequency_penalty=1.0,
                top_p=0.9,
                temperature=1e-10,
                do_sample = True,
                stop=["###", self.tokenizer.eos_token, self.tokenizer.pad_token],
                use_beam_search=False,
                max_new_tokens=300,
                logprobs=5)      

    def inference(self):
        columns = ['molecule','caption']
        df = pd.read_csv('data/big/test2.csv', usecols=columns)

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
           
        
            inputs = self.tokenizer(source, return_tensors="pt").to(self.device) 
            outputs = self.model.generate(**inputs, generation_config=self.generation_config)  
            predicted = self.tokenizer.batch_decode(outputs[:,inputs.input_ids.shape[1]:], skip_special_tokens=True) 
            col1.append(source)
            col2.append(target)
            col3.append(predicted)   

         
        data = [col1,col2,col3]
        transposed_data  = list(map(list, zip(*data)))
        output = pd.DataFrame(transposed_data, columns=columns)
        output.to_csv(self.export)      

    def load_model(self):    
        model = AutoPeftModelForCausalLM.from_pretrained(
            self.checkpoint,
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        return tokenizer, model
    



if __name__ =="__main__":
    args = parse_arguments()   
    checkpoint = "tmpbig/CPO_Meditron7b_reverse/checkpoint-1188"     
    mode = "smiles2caption" #caption2smiles, smiles2caption
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    export = "output/big/smiles2caption_CPO_reverse_test2"
    self = run(args,checkpoint,mode,device,export) 
    self.inference()
