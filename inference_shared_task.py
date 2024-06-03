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
                temperature=1e-9,
                do_sample = True,
                stop=[self.tokenizer.eos_token, self.tokenizer.pad_token],
                use_beam_search=False,
                # max_length=300,
                # max_tokens=200,
                max_new_tokens=300,
                logprobs=5)      

    def inderence(self):
        if self.mode=='smiles2caption':
            columns = ['molecule',]
            df = pd.read_csv('data/big/test_caption.csv', usecols=columns)
        elif self.mode=='caption2smiles':   
            columns = ['caption',]
            df = pd.read_csv('data/big/test_molecule.csv', usecols=columns)  
       

        if self.mode == "smiles2caption":
            #columns = ['SMILES', 'ground truth', 'output']
            sources = df['molecule'].tolist()
            #targets = df['caption'].tolist()
        elif self.mode=='caption2smiles':  
             
            #columns = ['description', 'ground truth', 'output']
            sources = df['caption'].tolist()
            #targets = df['molecule'].tolist()

       

        col1 = []
    
        for i in range (len(sources)):
            
            source = sources[i]    
           
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

           
            col1.append(predicted)

        data = pd.DataFrame()
        data['predictions'] = col1
        # transposed_data  = list(map(list, zip(*data)))
        # output = pd.DataFrame(transposed_data, columns=columns)
        # output.to_csv(self.export)
        data.to_csv(self.export , index=False, header=True)


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
    export = "shared_CPO_SLERP2_smiles2caption.csv"
    #language-plus-molecules/Meditron7b-caption2smiles-LPM24
    #for reverse: language-plus-molecules/Meditron7b-smiles2caption-LPM24
    config_model = "language-plus-molecules/Meditron7b-caption2smiles-LPM24"
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    self = run(args,checkpoint,mode,device,export,config_model) 
    self.inderence()


    # from datasets import load_dataset
    # import pandas as pd
    # dataset = load_dataset("language-plus-molecules/LPM-24_eval-molgen")
    # dataset = load_dataset("language-plus-molecules/LPM-24_eval-caption")

    # df = pd.DataFrame(dataset['train'])
    # df.to_csv("data/big/test_caption.csv")


# from torchmetrics.text import CharErrorRate
# preds = ["CCC(C)CCCCCCCCCCCCC(=O)OC[C@H](COC(=O)CCCCCCCCCCCC(C)C)OC(=O)CCCCCCCCCCCCCCCCCCC(C)C"]
# target = ["CCCCCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](COC(=O)CCCCCCCCCCCCCCCCCCCCC(C)C)OC(=O)CCCCCCCCCCCCCCCCCCCCC(C)C"]
# cer = CharErrorRate()
# cer(preds, target)
    
# from rdkit import Chem
# from rdkit.Chem import rdDepictor
# rdDepictor.SetPreferCoordGen(True)
# from rdkit.Chem.Draw import IPythonConsole
# IPythonConsole.drawOptions.addBondIndices = True

# mol = Chem.MolFromSmiles('CCCCCC/C=C\CCCCCCCC(=O)OC[C@H](COC(=O)CCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCC/C=C\CCCCCCCC')
# mol

# smiles = "(=O)NC(C(=O)N1CCCC1c1ncc(C#CC#Cc2ccc(-c3cnc(C4CCCN4C(=O)C(NC(=O)OC)C(C)C)[nH]3)cc2-c2cccc(C(C)(C)C)c2)[nH]1)c1ccccc1"  
# Chem.MolFromSmiles(smiles)


# from rdkit import Chem
# from rdkit.Chem import Draw

# # Define the SMILES string for the molecule
# smiles = "COc1cnc(-c2cc(Cl)ccc2F)nc1Nc1ccncn1"
# smiles = "CC(C)(C)OC(=O)N1CCN(c2ccc(C(=O)Nc3ccc(C(=O)N4CCN(C(=O)OC(C)(C)C)CC4)cc3)cc2)CC1"
# smiles = "Cc1ccc(C(=O)Nc2cccc(C(F)(F)F)c2)cc1-c1cc(Cl)ccc1Cl"





# base = "CC(=O)N1C(=O)N(C(=O)Cl)CC1C"
# Chem.MolFromSmiles(base)

# med = "CCCCCCCCCCCC(N)OCC(CC)CCCC"
# Chem.MolFromSmiles(med)
# 2,0.26,1


# cpo = "CC(C)C(=S)NCC1CN(c2ccc(N3CCN(S(N)(=O)=O)CC3)c(F)c2)C(=O)O1"
# Chem.MolFromSmiles(cpo)
# 31,0.33,1

# cpo_slerp = "Cc1cc(N)c(C)c(OC(=O)C(Cl)C)c1C"
# Chem.MolFromSmiles(cpo_slerp)
# 2,0.26,1

# cpo_slerp = "CCN1c2cc(-c3c(F)ccc(F)c3C#N)ccc2N(Cc1cc)S1(O)O"
# Chem.MolFromSmiles(cpo_slerp)






# smiles = "CCCCc1cn(-c2c(C(C)C)cnn2C)c(=O)n1Cc1cc(-c2cccc(-c3nn[nH]n3)c2)ccn1"
# smiles = "Cc1ccc(-c2ccn(C)n2)c(C/C=C\\C=C\\C(=O)c2ccccc2)c1"
# smiles = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"

# from torchmetrics.text import CHRFScore
# smile1 = "CCC(C)CCCCCCCCCCCCCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCC(C)C)COP(=O)(O)OC[C@@H](O)COP(=O)(O)OC[C@@H](COC(=O)CCCCCCCCC(C)C)OC(=O)CCCCCCCCCCCCCCCCCC(C)C"
# smile2 = "CCC(C)CCCCCCCCCCCCCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCC(C)C)COP(=O)(O)OC[C@@H](O)COP(=O)(O)OC[C@@H]"
# chrf = CHRFScore(n_char_order=2)
# chrf([smile1], [smile2])

# # Create the molecule object from the SMILES string
# mol = Chem.MolFromSmiles(smiles)

# # Generate 2D coordinates for the molecule
