import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
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




def laod_dataset(path):
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

    return dataset    

def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_j = tokenizer(chosen, truncation=True)
        tokenized_k = tokenizer(rejected, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

    return new_examples



if __name__=="__main__":
    args = parse_arguments()  
    data_train = laod_dataset(args.train_path)   
    data_val = laod_dataset(args.val_path)      

    model_id = "language-plus-molecules/Meditron7b-caption2smiles-LPM24"
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
    model.config.pretraining_tp = 1  

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right" #Pad to left due to 16f error

    peft_config = LoraConfig(
            task_type="SEQ_CLS",
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
    
    train_dataset = data_train.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )

    test_dataset = data_val.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )

    #Create the folder for saving the best models
    save_path = os.path.join('./tmp',args.setting+'_'+ args.model_name)

    training_args = RewardConfig(
            output_dir=save_path,   # directory to save and repository id
            overwrite_output_dir = True,
            load_best_model_at_end=args.load_best_model_at_end,
            save_total_limit=args.save_total_limit,
            num_train_epochs=args.epochs,  # number of training epochs
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
            gradient_accumulation_steps=args.gradient_accumulation_steps,  # number of steps before performing a backward/update pass
            gradient_checkpointing=args.gradient_checkpointing,            # use gradient checkpointing to save memory
            optim="adafactor",              # use fused adamw optimizer
            learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
            #max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
            #warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
            #lr_scheduler_type="cosine",             # use cosine learning rate scheduler
            #Report
            save_strategy=args.save_strategy,
            evaluation_strategy=args.evaluation_strategy,
            logging_steps=args.logging_steps,
            metric_for_best_model= args.metric_for_best_model, 
            greater_is_better=args.greater_is_better,
            disable_tqdm = args.disable_tqdm,
            #bf16=True,                              # use bfloat16 precision
            #tf32=True,                              # use tf32 precision
            push_to_hub=False,                      # push model to hub
            report_to="wandb", # enables logging to W&B
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_kwargs},
            remove_unused_columns=False,
        )
    
       
    
    trainer = RewardTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=peft_config,
            max_length=1024,
        )
    trainer.train()