import argparse

def parse_arguments(): 
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default='0', type=str, help='enables cuda')
    parser.add_argument('--epochs', default=1, type=int, help='number of epochs to train for')
    parser.add_argument('--model_name', type=str, default='SLERP2', choices=['Meditron7b', 'Hybrid_Meditron7b' , 'Meditron7b_reverse', 'SLERP', 'SLERP2', 'TIES', 'TIES2', 'TIES3', 'Passthrough'])
   
    parser.add_argument('--setting', type=str, default='CPO', choices=['SFT', 'DPO', 'CPO', 'Reward', 'KTO'], help='training method')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=64, help='Number of updates steps to accumulate the gradients for, before performing a backward/update pass.')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--lr_scheduler_type', type=str, default='constant', choices=['linear','cosine', 'constant'], help='Learning rate scheduler modifing the learning rate during training, there different types of schedulers (linear,constant)')
    parser.add_argument('--optim', type=str, default='paged_adamw_32bit', choices=['adamw_torch_fused', 'paged_adamw_32bit'], help='optimizer') 
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Preventing  from fitting the training data too closely and improve its ability to generalize to new, unseen data.')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='Ratio of total training steps used for a linear warmup from 0 to learning_rate.')
    parser.add_argument('--fp16', type=bool, default=True, help='Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.')
    parser.add_argument('--gradient_checkpointing', type=bool, default=False, help='Leads to reduction in memory at slighly decrease in speed')
    parser.add_argument('--gradient_checkpointing_kwargs', type=bool, default=True, help='Leads to reduction in memory at slighly decrease in speed')
    parser.add_argument('--max_seq_length', type=int, default=1024, help='Context Length')


    #Data
    parser.add_argument('--train_path', type=str, default='data/big/dpo_train_.csv', help='train data')
    parser.add_argument('--val_path', type=str, default='data/big/dpo_val_.csv', help='val data')
    parser.add_argument('--test_path', type=str, default='data/test.csv', help='test data')

    #Quantization
    parser.add_argument('--load_in_4bit', type=bool, default=True, help='4-bit quantization')
    parser.add_argument('--bnb_4bit_use_double_quant', type=bool, default=True, help='This argument suggests using the nested quantization technique, which offers even greater memory efficiency without compromising performance.')
    parser.add_argument('--bnb_4bit_quant_type', type=str, default='nf4', help='NF4 data type')


    #LoRA config based on QLoRA paper
    #LoRA Config PEFT train a smaller subset of parameters or use low-rank adaptation (LoRA) 
    #LoRA’s approach to fine-tuning uses low-rank decomposition to represent weight updates with two smaller matrices.
    parser.add_argument('--lora_alpha', type=int, default=16, help='This parameter represents the scaling factor for the weight matrices in LoRA, which is adjusted by alpha to control the magnitude of the combined output from the base model and low-rank adaptation.')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='Drop out for regularization')
    parser.add_argument('--r', type=int, default=64, help='This represents the LoRA rank of the update matrices, expressed in int. Lower rank results in smaller update matrices with fewer trainable parameters', choices=[16,64])
    parser.add_argument('--bias', type=str, default="none", help='bias')
    parser.add_argument('--task_type', type=str, default='CAUSAL_LM', help='')

    #Report
    parser.add_argument('--save_strategy', type=str, default='epoch', help='When to save a model')
    parser.add_argument('--evaluation_strategy', type=str, default='epoch', help='Number of update steps between two evaluations if evaluation_strategy="steps". Will default to the same value as logging_steps if not set. Should be an integer or a float in range [0,1). If smaller than 1, will be interpreted as ratio of total training steps.')
    parser.add_argument('--logging_steps', type=int, default=10, help='How often the training logs are printed or recorded (e.g., training loss, evaluation metrics, etc.)')
    parser.add_argument('--metric_for_best_model', type=str, default='eval_loss', choices=['eval_loss'], help="Metric to choose the best model")
    parser.add_argument('--disable_tqdm', type=bool, default=True, help="Disable or enable tqdm progress")
    parser.add_argument('--greater_is_better', type=bool, default=False, help='Regarding the evaluation metric, e.g., when loss->False')
    parser.add_argument('--load_best_model_at_end', type=bool, default=True, help='Whether or not to load the best model found during training at the end of training. When this option is enabled, the best checkpoint will always be saved.')
    parser.add_argument('--save_total_limit', type=int, default=1, help="Here, two checkpoints are saved: the last one and the best one (if they are different).")

    opt = parser.parse_args("")
    
    return opt