import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["WORLD_SIZE"] = "1"
import os
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline,GenerationConfig
from argument_parser import parse_arguments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import pandas as pd
from datasets import Dataset

class evaluate():
    def __init__(self, args):
        self.args = args

        if self.args.model_name=="Meditron7b":
            self.model_id = "language-plus-molecules/Meditron7b-caption2smiles-LPM24"
        else:
            print("model not defined")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            return_dict=True,
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)  


        # self.generation_config, unused_kwargs = GenerationConfig.from_pretrained(
        #     self.model_id, do_sample=True, temperature=0.7, top_p=0.9, num_return_sequences=1,truncation=True, max_length=500,return_unused_kwargs=True
        #     )

        

        self.test_data = self.data(self.args.val_path)

    def data(self,path):
        tasks = {
            "caption-generation": "Generate a caption given a molecule: ",
            "molecule-generation": "Generate a molecule given a description: "
        }   
        columns = ['molecule', 'caption']
        data = pd.read_csv(os.path.join(path), usecols=columns) 

        dataset_data = []
        for key, value in tasks.items():
            task_data = [
                {
                    "instruction": value,
                    "input": row_dict["molecule"] if key == "caption-generation" else row_dict["caption"],
                    "output": row_dict["caption"] if key == "caption-generation" else row_dict["molecule"]
                }
                for row_dict in data.to_dict(orient="records")
            ]
            dataset_data.extend(task_data)

        dataset_data = Dataset.from_pandas(pd.DataFrame(dataset_data))  
       
       
        
        original_columns = dataset_data.column_names
        dataset_data = dataset_data.map(self.chatml_format,remove_columns=original_columns)

        return dataset_data
    
    def chatml_format(self, datum):

        message = [{"role":"system", "content":"This is a competitive yet complementary shared task based on 'translating' between molecules and natural language."},
            {"role": "user", "content":datum['instruction'] + datum['input']}] 
        
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False,add_generation_prompt=True)

        reference = datum['output']

        datum = {
                    "prompt": prompt,
                    "reference": reference
                }
        
        return datum
        
    def inference(self):
        generator = pipeline("text-generation", model=self.model,tokenizer=self.tokenizer,device_map={"": 1})  

        generated = generator(
                "Generate a molecule given a description: The molecule is a factor xa inhibitor and belongs to the anti thrombotic class of molecules. ",
                do_sample=True,
                temperature=1e-10,
                top_p=0.9,
                num_return_sequences=1,
                truncation=True,
                max_length=500,
        )    

        print(generated[0]['generated_text'])


if __name__ == "__main__":
    args = parse_arguments()
    self = evaluate(args)       
