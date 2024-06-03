import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["WORLD_SIZE"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["WANDB_DISABLED"]="false"
from argument_parser import parse_arguments
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig,AutoModelForCausalLM,TrainingArguments,AutoModelForSequenceClassification, AutoConfig
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from numpy import percentile
from trl import DPOTrainer,CPOTrainer,CPOConfig
from accelerate import Accelerator
import wandb

class run():
    def __init__(self, args):
        self.args = args

        print('Train Initialization...')
        self.tokenizer, self.model = self.train_preparation()

        print("Preparing the data...")
        self.data_train = self.laod_dataset(self.args.train_path)
        self.data_val = self.laod_dataset(self.args.val_path)

        self.prompt_length_train, self.max_seq_length_train  = self.calculate_lengths(self.data_train)
        self.prompt_length_val, self.max_seq_length_val  = self.calculate_lengths(self.data_val)

        self.prompt_length = max([self.prompt_length_train,self.prompt_length_val])
        self.max_seq_length = max([self.max_seq_length_train,self.max_seq_length_val])

        


    def calculate_lengths(self, data, perc=100):
        # # lets find the p95 length of the prompt

        prompt_length = int(percentile([len(self.tokenizer(x)["input_ids"]) for x in data["prompt"]], perc))  
        max_seq_length_chosen = int(percentile([len(self.tokenizer(x["prompt"] + x["chosen"])["input_ids"]) for x in data], perc))
        max_seq_length_rejected = int(percentile([len(self.tokenizer(x["prompt"] + x["rejected"])["input_ids"]) for x in data], perc))
        max_seq_length = max(max_seq_length_chosen, max_seq_length_rejected)

        print(f"Percentile {perc} prompt length: {prompt_length}")
        print(f"Percentile {perc} prompt + chosen length: {max_seq_length}")    
        return prompt_length, max_seq_length    

    def laod_dataset(self,path:str):     
        columns = ['source', 'chosen','rejected']
        columns_new = ['prompt', 'chosen','rejected']
        dict_columnes = {columns[i]: columns_new[i] for i in range(len(columns))}
        data = pd.read_csv(os.path.join(path), usecols=columns)  
        data = data.rename(columns=dict_columnes)
        data = data.sample(frac=1).reset_index(drop=True)
        dataset = Dataset.from_pandas(pd.DataFrame(data)) 
        original_columns = dataset.column_names

        dataset = dataset.map(self.data_format,remove_columns=original_columns)

        return dataset

    def data_format(self, datum):

        if "mol2cap:" in datum['prompt']:
            source = datum['prompt'].replace("mol2cap:","").strip()
            prompt_template =     f"""Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.\n\n
            ### Instruction:\n
            You are a researcher. You can come up captions based on your existing knowledge.
            Captions are given against the following input. You should be as detailed as possible.\n\n
            ### Input:\n
            Molecule: {source} \n
            In that molecule, could you formulate a caption about?\n\n\n
            ### Response:"""
        elif "cap2mol:" in datum['prompt']:   
            source = datum['prompt'].replace("cap2mol:","").strip() 
            prompt_template =  f"""Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.\n\n
            ### Instruction:\n
            You are a researcher. You can come up molecule smile strings based on your existing knowledge. 
            Molecule smile strings are given against the following input. You should be as detailed as possible.\n\n
            ### Input:\nCaption: {source} \n
            In that caption, could you generate a molecule smile string?\n\n\n
            ### Response:"""
            
        
        prompt = prompt_template
        chosen = datum['chosen'] + self.tokenizer.eos_token
        rejected = datum['rejected'] + self.tokenizer.eos_token

        datum = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                } 
        
        return datum
    
    def train_preparation(self):

        if self.args.model_name=="Meditron7b":
            model_id = "language-plus-molecules/Meditron7b-caption2smiles-LPM24"
        elif self.args.model_name=="Meditron7b_reverse":    
            model_id = "language-plus-molecules/Meditron7b-smiles2caption-LPM24"
        elif self.args.model_name== "Hybrid_Meditron7b":
            model_id = "models/hybrid_meditron7b"
        elif  self.args.model_name== "SLERP":  
            model_id = "mergekit/output_folder/SLERP"  
        elif  self.args.model_name== "TIES":  
            model_id = "mergekit/output_folder/TIES"      
        elif  self.args.model_name== "Passthrough":   
            model_id = "mergekit/output_folder/Passthrough"     
        elif  self.args.model_name== "TIES2":  
            model_id = "mergekit/output_folder/TIES2" 
        elif  self.args.model_name== "TIES3":  
            model_id = "mergekit/output_folder/TIES3"     
        elif  self.args.model_name== "SLERP2":  
            model_id = "mergekit/output_folder/SLERP2"        
        else:
            print("model not defined")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # tokenizer.pad_token_id = (
        #     0  # unk. we want this to be different from the eos token
        # )
        tokenizer.padding_side = "right" #Pad to left due to 16f error


        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # if self.args.setting =="DPO":
        model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quantization_config, device_map={"": 0}, trust_remote_code=True)  
        # elif self.args.setting =="CPO":   
        #     model = AutoModelForSequenceClassification.from_pretrained(model_id,quantization_config=quantization_config, device_map={"": 0}, trust_remote_code=True)  
        #     model.config.pad_token_id = model.config.eos_token_id



        model.config.use_cache = False
        #a value different than 1 will activate the more accurate but slower computation of the linear layers, which should better match the original logits
        model.config.pretraining_tp = 1  




        return tokenizer, model 
    
    def trainer(self):

       

        peft_config = LoraConfig(
            lora_alpha=self.args.lora_alpha,
            r = self.args.r,
            lora_dropout=self.args.lora_dropout,
            task_type=self.args.task_type,
            bias=self.args.bias,
            target_modules= "all-linear" #self.find_all_linear_names(self.model)

        )

        # self.model = prepare_model_for_kbit_training(self.model)
        # self.model = get_peft_model(self.model, peft_config)
        # self.model.print_trainable_parameters()

        #Create the folder for saving the best models
        save_path = os.path.join('./tmpbig',self.args.setting+'_'+ self.args.model_name)

        args = TrainingArguments(
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
            max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
            warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
            lr_scheduler_type="cosine",             # use cosine learning rate scheduler
            #Report
            save_strategy=self.args.save_strategy,
            evaluation_strategy=self.args.evaluation_strategy,
            logging_steps=self.args.logging_steps,
            metric_for_best_model= self.args.metric_for_best_model, 
            greater_is_better=self.args.greater_is_better,
            disable_tqdm = self.args.disable_tqdm,
            bf16=True,                              # use bfloat16 precision
            tf32=True,                              # use tf32 precision
            push_to_hub=False,                      # push model to hub
            report_to="wandb", # enables logging to W&B 😎
            gradient_checkpointing_kwargs={"use_reentrant": self.args.gradient_checkpointing_kwargs}
        )

        
        if self.args.setting == "DPO":
            dpo_args = {
                "beta": 0.1,    # The beta factor in DPO loss. Higher beta means less divergence
                "loss_type": "kto_pair",  # specify the "cdpo" loss type for handling noisy preference labels
            }

            trainer = DPOTrainer(
                self.model,
                ref_model=None, # set to none since we use peft
                peft_config=peft_config,
                args=args,
                train_dataset=self.data_train,
                eval_dataset=self.data_val,
                tokenizer=self.tokenizer,
                max_length=self.max_seq_length,
                max_prompt_length=self.prompt_length,
                beta=dpo_args["beta"],
                loss_type=dpo_args["loss_type"]
            )
        elif self.args.setting == "CPO":
            
            cpo_config = CPOConfig(
                beta=0.1,
                output_dir=save_path,
                overwrite_output_dir = True,
                load_best_model_at_end=self.args.load_best_model_at_end,
                save_total_limit=self.args.save_total_limit,
                num_train_epochs=self.args.epochs,  # number of training epochs
                per_device_train_batch_size=self.args.batch_size,  # batch size per device during training
                per_device_eval_batch_size=self.args.batch_size,  # batch size for evaluation
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,  # number of steps before performing a backward/update 
                #Report
                save_strategy=self.args.save_strategy,
                evaluation_strategy=self.args.evaluation_strategy,
                logging_steps=self.args.logging_steps,
                metric_for_best_model= self.args.metric_for_best_model, 
                greater_is_better=self.args.greater_is_better,
                disable_tqdm = self.args.disable_tqdm,
                bf16=True,                              # use bfloat16 precision
                tf32=True,                              # use tf32 precision
                report_to="wandb", # enables logging to W&B 😎
                gradient_checkpointing_kwargs={"use_reentrant": self.args.gradient_checkpointing_kwargs},
                max_length=self.max_seq_length,
                max_prompt_length=self.prompt_length,
                optim="adamw_torch_fused",   
                warmup_ratio=0.1,   
            )
            trainer = CPOTrainer(
                self.model,
                peft_config=peft_config,
                args=cpo_config,
                train_dataset=self.data_train,
                eval_dataset=self.data_val,
                tokenizer=self.tokenizer
            )


        return trainer



if __name__ =="__main__":
    args = parse_arguments()        
    self = run(args)        


    project_name = "ALMol"
    run_name="_".join((args.setting, args.model_name))
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name, 
                              config=vars(args), 
                              init_kwargs={"wandb":{"name":run_name}}) 
    
    trainer = self.trainer()
    trainer.train()  