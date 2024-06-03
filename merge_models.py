from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tqdm



def load_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


model_name_1= "language-plus-molecules/Meditron7b-caption2smiles-LPM24"
model_name_2= "language-plus-molecules/Meditron7b-smiles2caption-LPM24"

model1, tokenizer1 = load_model(model_name_1, device=0)
model2, tokenizer2 = load_model(model_name_2, device=0)


def merge_models(model1, model2, alpha=0.5):
    print("Instantiating merged model")
    merged_model = AutoModelForCausalLM.from_config(model1.config).to("cpu")

    total_params = sum(1 for _ in model1.parameters())
    progress_bar = tqdm.tqdm(total=total_params, desc="Merging parameters")

    with torch.no_grad():
        for param1, param2, merged_param in zip(
            model1.parameters(), model2.parameters(), merged_model.parameters()
        ):
            merged_param.data = alpha * param1.data + (1 - alpha) * param2.data
            progress_bar.update(1)

    progress_bar.close()
    return merged_model


merged_model = merge_models(model1, model2, alpha=0.5)

merged_model.save_pretrained(save_directory="models/hybrid_meditron7b")
tokenizer1.save_pretrained(save_directory="models/hybrid_meditron7b")