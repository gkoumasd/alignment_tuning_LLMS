import pandas as pd

path = "../output/big/caption2smiles_CPO_TIES.csv"
path_out = "../output/big/caption2smiles_CPO_TIES.txt"
columns = ["SMILES","ground truth",	"output"]
columns = ["description", "ground truth","output"]
df = pd.read_csv(path, usecols=columns)

molecules = []
for i in range(len(df)):
   
    str =df['output'].iloc[i]
    
    
    str =  str.replace("['", "")
    str =  str.replace("']", "")
    #str =  str.replace(" ", "")
    str =  str.replace("#", "")
    str =  str.replace("\\n", "")
    molecules.append(str)
df['output'] = molecules    
    
    
df.to_csv(path_out, sep='\t', index=False)


#Inference for KTO
import pandas as pd

path = "../output/smiles2caption_SFT.csv"
path_out = "../output/smiles2caption_SFT.txt"
columns = ["SMILES","ground truth",	"output"]
columns = ["description", "ground truth","output"]
df = pd.read_csv(path, usecols=columns)

molecules = []
for i in range(len(df)):
   
    str =df['output'].iloc[i]
    str =  str.replace("['", "")
    str =  str.replace("']", "")
    str =  str.replace("\\n", "")
    str =  str.replace("assistant", "")
    str = str.strip()
    str = str.split(" ")[0]
    molecules.append(str)

df['output'] = molecules  
df.to_csv(path_out, sep='\t', index=False)     
