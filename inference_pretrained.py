import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["WORLD_SIZE"] = "1"
from argument_parser import parse_arguments
#### COMMENT IN TO MERGE PEFT AND BASE MODEL ####
from peft import PeftModel, PeftConfig
from transformers import  AutoTokenizer,GenerationConfig, AutoModelForCausalLM
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
    def __init__(self,args,checkpoint,mode,device,export, config_model):
        self.args = args
        self.checkpoint = checkpoint
        self.mode = mode
        self.device = device
        self.model_id = checkpoint
        self.export = export
        self.config_model = config_model

        print('Load from checkpoint')
        self.tokenizer, self.model = self.load_model()
        self.model.to(self.device)
       

        if self.mode=='smiles2caption':
            self.prefix = "Translate the following molecule smile string to a descriptive caption: " 
        elif self.mode=='caption2smiles':  
            self.prefix = "Translate the following descriptive caption to a molecule smile string: " 

        self.generation_config = GenerationConfig.from_pretrained(self.config_model ,best_of=1,
                presence_penalty=0.0,
                frequency_penalty=1.0,
                top_p=0.9,
                temperature=1e-10,
                do_sample = True,
                stop=[self.tokenizer.eos_token, self.tokenizer.pad_token],
                use_beam_search=False,
                # max_length=300,
                # max_tokens=200,
                max_new_tokens=300,
                logprobs=5)      

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

            source = sources[i]    
            target = targets[i]

            if self.mode == "smiles2caption":
                  prompt_template =     f"""Below is an instruction that describes a task, paired with an input that provides further context.
                    Write a response that appropriately completes the request.\n\n
                    ### Instruction:\n You are a researcher. You can come up captions based on your existing knowledge. Captions are given against the following input. You should be as detailed as possible.\n\n
                    ### Input:\nMolecule: {source} \nIn that molecule, could you formulate a caption about?\n\n\n### Response:"""
            elif self.mode=='caption2smiles':        
                 prompt_template =  f"""Below is an instruction that describes a task, paired with an input that provides further context. 
                    Write a response that appropriately completes the request.\n\n
                    ### Instruction:\nYou are a researcher. You can come up molecule smile strings based on your existing knowledge. 
                    Molecule smile strings are given against the following input. You should be as detailed as possible.\n\n
                    ### Input:\nCaption: {source} \nIn that caption, could you generate a molecule smile string?\n\n\n### Response:"""

           
            inputs = self.tokenizer(prompt_template, return_tensors="pt").to(self.device) 
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
        # model = AutoModelForCausalLM.from_pretrained(
        #      self.model_id,
        #      device_map="auto",
        #      torch_dtype=torch.float16,
        #      trust_remote_code = True
        # )
        model = AutoPeftModelForCausalLM.from_pretrained(
             self.checkpoint,
             device_map="auto",
             torch_dtype=torch.float16
         )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id,trust_remote_code=True)

        return tokenizer, model
    



if __name__ =="__main__":
    args = parse_arguments()   
    checkpoint = "tmpbig/CPO_SLERP2/checkpoint-396"     
    mode = "smiles2caption" #caption2smiles, smiles2caption
    export = "output/big/smiles2caption_CPO_SLERP2.csv"
    #language-plus-molecules/Meditron7b-caption2smiles-LPM24
    #for reverse: language-plus-molecules/Meditron7b-smiles2caption-LPM24
    config_model = "language-plus-molecules/Meditron7b-caption2smiles-LPM24"
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    self = run(args,checkpoint,mode,device,export,config_model) 
    self.inderence()