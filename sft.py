import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["WORLD_SIZE"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["WANDB_DISABLED"]="false"
from argument_parser import parse_arguments
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig,AutoModelForCausalLM,TrainingArguments,DataCollatorForLanguageModeling
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
from trl import SFTTrainer
from accelerate import Accelerator
import wandb
import random


class run():
    def __init__(self, args):
        self.args = args

        print("Preparing the data...")
        self.data_train = self.dataset_format(self.args.train_path)
        self.data_val = self.dataset_format(self.args.val_path)

        print('Train Initialization...')
        self.tokenizer, self.model = self.train_preparation()

    def dataset_format(self,path:str):  

        #Load the data
        columns = ['source', 'chosen','rejected']
        data = pd.read_csv(os.path.join(path), usecols=columns) 

        columns_new = ['instruction', 'input','output']
        data_new = pd.DataFrame(columns=columns_new) 

        instsructions = []
        inputs = []
        outputs = []
        
        for i in range (len(data)):
            if "mol2cap:" in data['source'].iloc[i]:
                source = data['source'].iloc[i].replace("mol2cap:","").strip()
                target = data['chosen'].iloc[i]
                instruction = f"""You are a researcher. You can come up captions based on your existing knowledge.
                Captions are given against the following input. You should be as detailed as possible.\n\n"""
                input = f"""Molecule: {source} \n
                In that molecule, could you formulate a caption about?\n\n\n"""
                output = f"""{target}"""
               
            elif "cap2mol:" in data['source'].iloc[i]: 
                source = data['source'].iloc[i].replace("cap2mol:","").strip()
                target = data['chosen'].iloc[i]
                instruction = f"""You are a researcher. You can come up molecule smile strings based on your existing knowledge.
                Molecule smile strings are given against the following input. You should be as detailed as possible.\n\n"""
                input = f"""Caption: {source} \n
                In that caption, could you generate a molecule smile string?\n\n\n"""
                output = f"""{target}"""
            instsructions.append(instruction)
            inputs.append(input)   
            outputs.append(output)       


        data_new['instruction'] = instsructions    
        data_new['input'] = inputs    
        data_new['output'] = outputs   
        data_new = data_new.sample(frac=1).reset_index(drop=True) 
        

        dataset_data = Dataset.from_pandas(pd.DataFrame(data_new)) 

       
        return dataset_data  
    
    def format_instruction(self,data):
        output_texts = []
        for i in range(len(data['input'])):
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.\n\n
            ### Instruction:\n
            {data["instruction"][i]} 
            ### Input:\n
            {data["input"][i]} 
            ### Response:\n
            {data["output"][i]}""".strip()

            output_texts.append(prompt)
        return output_texts    
    
    def train_preparation(self):

        if self.args.model_name=="Meditron7b":
            model_id = "language-plus-molecules/Meditron7b-caption2smiles-LPM24"
        elif self.args.model_name=="Meditron7b_reverse":    
            model_id = "language-plus-molecules/Meditron7b-smiles2caption-LPM24"
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

        model = AutoModelForCausalLM.from_pretrained(model_id,  quantization_config=quantization_config, device_map={"": 0}, trust_remote_code=True)  

        model.config.use_cache = False
        #a value different than 1 will activate the more accurate but slower computation of the linear layers, which should better match the original logits
        model.config.pretraining_tp = 1  




        return tokenizer, model   

    def trainer(self):
        collator =  DataCollatorForLanguageModeling(self.tokenizer, mlm=False, return_tensors = 'pt')

        peft_parameters = LoraConfig(
            use_dora=True,
            lora_alpha=self.args.lora_alpha, 
            lora_dropout=self.args.lora_dropout,
            r=self.args.r,
            bias=self.args.bias,
            task_type=self.args.task_type,
            target_modules= "all-linear"
        )

        #Prepare the model for training
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, peft_parameters)
        self.model.print_trainable_parameters()

        #Create the folder for saving the best models
        save_path = os.path.join('./tmpbig',self.args.setting+'_'+ self.args.model_name)

        training_args = TrainingArguments(
            report_to="wandb", # enables logging to W&B 😎
            output_dir=save_path,
            overwrite_output_dir = True,
            load_best_model_at_end=self.args.load_best_model_at_end,
            save_total_limit=self.args.save_total_limit,
            #Training
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,  
            learning_rate=self.args.lr,
            num_train_epochs=self.args.epochs,
            lr_scheduler_type=self.args.lr_scheduler_type,
            optim=self.args.optim,
            weight_decay=self.args.weight_decay,
            warmup_ratio = self.args.warmup_ratio,
            fp16=self.args.fp16, 
            #Report
            save_strategy=self.args.save_strategy,
            evaluation_strategy=self.args.evaluation_strategy,
            logging_steps=self.args.logging_steps,
            metric_for_best_model= args.metric_for_best_model, 
            greater_is_better=self.args.greater_is_better,
            disable_tqdm = self.args.disable_tqdm,
            #Skip wornings
            gradient_checkpointing=self.args.gradient_checkpointing,  
            gradient_checkpointing_kwargs={"use_reentrant": self.args.gradient_checkpointing_kwargs}
        )

        #Trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.data_train,
            eval_dataset=self.data_val,
        #     compute_metrics=self.compute_metrics, #The trainer returns in eval_pred everything that the model returns expecpt of loss  
        #     preprocess_logits_for_metrics = self.preprocess_logits_for_metrics,
        #     peft_config=peft_parameters,
        #     #dataset_text_field="text", #That is when the data have a format [{'text'}:'....'], here that is not the case.
            tokenizer=self.tokenizer,
            args=training_args,
            packing=False, # this tells the trainer to pack sequences of `max_seq_lenght` when it's true, Never packing=True when using DataCollatorForCompletionOnlyLM
            formatting_func=self.format_instruction, # The instruction template to apply to the datum
            max_seq_length = self.args.max_seq_length,
            data_collator = collator #the dataset_text_field overrides the use of the collator, so don't use it when dataset_text_field is on
        #     callbacks=[EarlyStoppingCallback(early_stopping_patience=self.args.stop)],
            
            
        )

        return trainer

     
    def find_all_linear_names(self, model):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])


        # lm_head is often excluded.
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        return list(lora_module_names)    

      
        



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