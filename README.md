# This repository contains the implementation code for the paper titled [Feedback-aligned Mixed LLMs for Machine Language-Molecule Translation](https://arxiv.org/pdf/2405.13984)



## Table of Contents

- [Data](#data)
- [Usage](#usage)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

## Data
The models are trained with aligment tuning for language-molecule translation on LPM-24-Dataset.
The dataset is now available at https://github.com/language-plus-molecules/LPM-24-Dataset.
A manuscript detailing the dataset’s creation can be found [here](https://arxiv.org/pdf/2403.00791).


## Usage

### Run the models

1. Standard-finetune: `python train.py --model_name RoBERTa`
2. Multitask MLM: `python train.py --model_name RoBERTa_Multitask`
3. Entailment: `python train.py --model_name RoBERTa_entail`
4. Standard−prompt: `python train_prompt.py --model_name RoBERTa_Prompt`
5. Prompt−demonstrations: `python train_prompt.py --model_name RoBERTa_Prompt_dem`
6. Prompt−inverse: `python train_prompt.py --model_name RoBERTa_Prompt_inverse`

### Test the models

1. Standard-finetune: `python test.py --model_name RoBERTa`
2. Multitask MLM: `python test.py --model_name RoBERTa_Multitask`
3. Entailment: `python test.py --model_name RoBERTa_entail`
4. Standard−prompt: `python test_prompt.py --model_name RoBERTa_Prompt`
5. Prompt−demonstrations: `python test_prompt.py --model_name RoBERTa_Prompt_dem`
6. Prompt−inverse: `python test_prompt.py --model_name RoBERTa_Prompt_inverse`

## Contributing
1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit them: `git commit -m 'Add your feature`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request



## Author
Dimitris Gkoumas. For more information, please visit [gkoumasd.github.io](https://gkoumasd.github.io)  


## License
If you find this project useful for your research, please consider citing it using the following BibTeX entry:


```bibtex
@inproceedings{gkoumas2023reformulating,
  title={Reformulating NLP tasks to Capture Longitudinal Manifestation of Language Disorders in People with Dementia.},
  author={Gkoumas, Dimitris and Purver, Matthew and Liakata, Maria},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={15904--15917},
  year={2023}
}
