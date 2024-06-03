import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["WORLD_SIZE"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["WANDB_DISABLED"]="false"
from argument_parser import parse_arguments
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer,BitsAndBytesConfig,DataCollatorWithPadding
from peft import LoraConfig, TaskType
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
from accelerate import Accelerator
import wandb
import torch
from numpy import percentile

def preprocess_function(examples):

    
    model_id = "language-plus-molecules/Meditron7b-caption2smiles-LPM24"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right" #Pad to left due to 16f error
    
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
        }
            
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, padding='max_length', truncation=True, max_length=900)
        tokenized_rejected = tokenizer(rejected, padding='max_length', truncation=True, max_length=900)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
            


    return new_examples

class run():
    def __init__(self,args):
        self.args = args


        print('Train Initialization...')
        self.tokenizer, self.model = self.train_preparation()

        print("Preparing the data...")
        # self.max_seq_length_train  = self.calculate_lengths(self.data_train)
        # self.max_seq_length_val  = self.calculate_lengths(self.data_val)
        self.max_seq = 890

        self.data_train = self.laod_dataset(self.args.train_path)
        self.data_val = self.laod_dataset(self.args.val_path)


    def laod_dataset(self,path):
        columns = ['source', 'chosen','rejected']
        data = pd.read_csv(os.path.join(path), usecols=columns) 

        chosen = []
        rejected = []
        for i in range (len(data)):
            chosen.append(data['source'].iloc[i] + data['chosen'].iloc[i])
            rejected.append(data['source'].iloc[i] + data['rejected'].iloc[i])

        dataset = pd.DataFrame(columns=['chosen','rejected'])  
        dataset['chosen'] = chosen 
        dataset['rejected'] = rejected 
        dataset = Dataset.from_pandas(pd.DataFrame(dataset)) 

        dataset = dataset.map(preprocess_function,num_proc=4)
 
        return dataset

    
    
            

    def train_preparation(self):
        if self.args.model_name=="Meditron7b":
            model_id = "language-plus-molecules/Meditron7b-caption2smiles-LPM24"
        else:
            print("model not defined")

       

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right" #Pad to left due to 16f error
        

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

       
        model_kwargs = dict(
            trust_remote_code=True,
            device_map={"": 0},
            quantization_config=quantization_config,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=1,
            **model_kwargs
        ) 

        model.config.use_cache = False
        #a value different than 1 will activate the more accurate but slower computation of the linear layers, which should better match the original logits
        model.config.pretraining_tp = 1  

        return tokenizer, model
    
    def trainer(self):
        collator =  DataCollatorWithPadding(self.tokenizer, max_length=self.max_seq, return_tensors = 'pt')
        
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        #Create the folder for saving the best models
        save_path = os.path.join('./tmp',self.args.setting+'_'+ self.args.model_name)

        args = RewardConfig(
            output_dir=save_path,   # directory to save and repository id
            overwrite_output_dir = True,
            load_best_model_at_end=self.args.load_best_model_at_end,
            save_total_limit=self.args.save_total_limit,
            num_train_epochs=self.args.epochs,  # number of training epochs
            per_device_train_batch_size=self.args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=self.args.batch_size,  # batch size for evaluation
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,  # number of steps before performing a backward/update pass
            gradient_checkpointing=self.args.gradient_checkpointing,            # use gradient checkpointing to save memory
            optim="adamw_torch_fused",              # use fused adamw optimizer
            learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
            #max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
            #warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
            #lr_scheduler_type="cosine",             # use cosine learning rate scheduler
            #Report
            save_strategy=self.args.save_strategy,
            evaluation_strategy=self.args.evaluation_strategy,
            logging_steps=self.args.logging_steps,
            metric_for_best_model= self.args.metric_for_best_model, 
            greater_is_better=self.args.greater_is_better,
            disable_tqdm = self.args.disable_tqdm,
            #bf16=True,                              # use bfloat16 precision
            #tf32=True,                              # use tf32 precision
            push_to_hub=False,                      # push model to hub
            report_to="wandb", # enables logging to W&B 😎
            gradient_checkpointing_kwargs={"use_reentrant": self.args.gradient_checkpointing_kwargs},
            max_length = self.max_seq
        )

        trainer = RewardTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            train_dataset=self.data_train,
            eval_dataset=self.data_val,
            peft_config=peft_config,
            
        )

        return trainer



if __name__ =="__main__":
    args = parse_arguments()        
    self = run(args)     

    
    project_name = "reward_rl"
    run_name="_".join((args.setting, args.model_name))
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name, 
                              config=vars(args), 
                              init_kwargs={"wandb":{"name":run_name}}) 
    
    trainer = self.trainer()
    trainer.train()     