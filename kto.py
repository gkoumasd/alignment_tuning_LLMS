import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["WORLD_SIZE"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["WANDB_DISABLED"]="false"
from argument_parser import parse_arguments
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig,AutoModelForCausalLM
import torch
from peft import LoraConfig
from trl import KTOConfig, KTOTrainer, setup_chat_format
from accelerate import Accelerator
import wandb

class run():
    def __init__(self, args):
        self.args = args

        print('Train Initialization...')
        self.tokenizer, self.model, self.model_ref = self.train_preparation()
        if self.tokenizer.chat_template is None:
            self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)
        self.model.resize_token_embeddings(len(self.tokenizer))    
        self.model_ref.resize_token_embeddings(len(self.tokenizer))    

        print("Preparing the data...")
        self.data_train = self.laod_dataset(self.args.train_path)
        self.data_val = self.laod_dataset(self.args.val_path)
        print("Chat format...")
        self.data_train_formated = self.data_train.map(self.format_dataset)
        self.data_val_formated = self.data_val.map(self.format_dataset)

        #self.prompt_length_train, self.max_seq_length_train  = self.calculate_lengths(self.data_train)
        #self.prompt_length_val, self.max_seq_length_val  = self.calculate_lengths(self.data_val)

        #self.prompt_length = max([self.prompt_length_train,self.prompt_length_val])
        #self.max_seq_length = max([self.max_seq_length_train,self.max_seq_length_val])

          

    def laod_dataset(self,path:str):     
        #Load the data
        columns = ['source', 'chosen','rejected']
        data = pd.read_csv(os.path.join(path), usecols=columns) 

        # Reshape the DataFrame
        melted_df = pd.melt(data, id_vars=['source'], value_vars=['chosen', 'rejected'],  value_name='completion',var_name='label')
        melted_df.rename(columns={'source': 'prompt'}, inplace=True)
        melted_df['label'] = melted_df['label'].apply(lambda x: True if x == 'chosen' else False)
        melted_df = melted_df[['prompt', 'completion', 'label']]
        melted_df = melted_df.sample(frac=1).reset_index(drop=True)


        prompts = []
        completions = []
        labels = []
        for i in range (len(melted_df)):
            if "mol2cap:" in melted_df['prompt'].iloc[i]:
                source = melted_df['prompt'].iloc[i].replace("mol2cap:","").strip()
                completion = melted_df['completion'].iloc[i]
                label = melted_df['label'].iloc[i]
                prompt_template =     f"""Below is an instruction that describes a task, paired with an input that provides further context.
                Write a response that appropriately completes the request.\n\n
                ### Instruction:\n
                You are a researcher. You can come up captions based on your existing knowledge.
                Captions are given against the following input. You should be as detailed as possible.\n\n
                ### Input:\n
                Molecule: {source} \n
                In that molecule, could you formulate a caption about?\n\n\n
                ### Response:"""
            elif "cap2mol:" in melted_df['prompt'].iloc[i]:  
                source = melted_df['prompt'].iloc[i].replace("cap2mol:","").strip()
                completion = melted_df['completion'].iloc[i]
                label = melted_df['label'].iloc[i]
                prompt_template =  f"""Below is an instruction that describes a task, paired with an input that provides further context.
                Write a response that appropriately completes the request.\n\n
                ### Instruction:\n
                You are a researcher. You can come up molecule smile strings based on your existing knowledge. 
                Molecule smile strings are given against the following input. You should be as detailed as possible.\n\n
                ### Input:\nCaption: {source} \n
                In that caption, could you generate a molecule smile string?\n\n\n
                ### Response:"""
            prompts.append(prompt_template)
            completions.append(completion)
            labels.append(label) 
        df_new = pd.DataFrame()       
        df_new['prompt'] = prompts
        df_new['completion'] = completions
        df_new['label'] = labels

        dataset = Dataset.from_pandas(pd.DataFrame(df_new)) 

        original_columns = dataset.column_names
        dataset = dataset.map(self.data_format,remove_columns=original_columns)
        

        return dataset

    def data_format(self, datum):
        
        prompt = [{"content":datum['prompt'] ,"role":"user"}]
        completion = [{"content":datum['completion'] ,"role":"assistant"}]
        label = datum['label'] 

        datum = {
                    "prompt": prompt,
                    "completion": completion,
                    "label": label,
                } 
        
        return datum
    
    def format_dataset(self,example):
        example["prompt"] = self.tokenizer.apply_chat_template(example["prompt"], tokenize=False)
        example["completion"] = self.tokenizer.apply_chat_template(example["completion"], tokenize=False)
        return example
    
    def train_preparation(self):

        if self.args.model_name=="Meditron7b":
            model_id = "language-plus-molecules/Meditron7b-caption2smiles-LPM24"
        elif self.args.model_name== "Hybrid_Meditron7b":
            model_id = "models/hybrid_meditron7b"
        else:
            print("model not defined")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
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

        model_ref = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quantization_config, device_map={"": 0}, trust_remote_code=True)




        return tokenizer, model ,model_ref
    
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
       

        training_args = KTOConfig(
            beta=0.1,
            desirable_weight=1.0,
            undesirable_weight=1.0,
            max_length = 1024,
            max_prompt_length = 400,
            remove_unused_columns = False,
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

        kto_trainer = KTOTrainer(
            self.model,
            self.model_ref,
            args=training_args,
            train_dataset=self.data_train_formated,
            eval_dataset=self.data_val_formated,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
        )

        return kto_trainer



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

     