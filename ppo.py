import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["WORLD_SIZE"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["WANDB_DISABLED"]="false"
from argument_parser import parse_arguments
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer,BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import torch
from peft import LoraConfig

class run():
    def __init__(self,args):
        self.args = args

        print('Train Initialization...')
        # self.tokenizer, self.model = self.train_preparation()

        print("Preparing the data...")
        # self.data_train = self.laod_dataset(self.args.train_path)
        # self.data_val = self.laod_dataset(self.args.val_path)

    def train_preparation(self):
        if self.args.model_name=="Meditron7b":
            model_id = "language-plus-molecules/Meditron7b-caption2smiles-LPM24"
        else:
            print("model not defined")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" #Pad to left due to 16f error


        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_id,
            device_map={"": 0},
            peft_config=lora_config,
            quantization_config=nf4_config,
            # reward_adapter=script_args.rm_adapter,
            # use_safetensors=script_args.use_safetensors,
        ) 

        model.config.use_cache = False
        #a value different than 1 will activate the more accurate but slower computation of the linear layers, which should better match the original logits
        model.config.pretraining_tp = 1     

    def laod_dataset(self,path:str):   
        columns = ['source', 'chosen'] 
        data = pd.read_csv(os.path.join(path), usecols=columns)     
        columns_new = ['query', 'chosen','rejected'] 

if __name__ =="__main__":
    args = parse_arguments()        
    self = run(args)      