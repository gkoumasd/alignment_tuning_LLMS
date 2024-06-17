# This repository contains the implementation code for the paper titled [Feedback-aligned Mixed LLMs for Machine Language-Molecule Translation](https://arxiv.org/pdf/2405.13984)



## Table of Contents

- [Data](#data)
- [Usage](#usage)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

## Data
LLMs are trained with aligment tuning for language-molecule translation on LPM-24-Dataset.
The dataset is now available at https://github.com/language-plus-molecules/LPM-24-Dataset.
A manuscript detailing the dataset’s creation can be found [here](https://arxiv.org/pdf/2403.00791).


## Usage

The repo includes codes for training LLMs across different strategies: a) Supervised Fine-Tuning (SFT), b) Direct Preference Optimisation (DPO), c) Contrastive Preference Optimisation (CPO) and d) Kahneman-Tversky Optimisation (KTO)

### Train LLMs with alignment tuning.

Follow the following steps to train a model:

1. Chose the appropriate model and training strategy as listed in `argument_parser.py`
2. For SFT run `sft.py`, for DPO and CPO run `dpo.py` and for KTO run `kto.py` 
3. Compbine the previous steps to train the LLMs. For example if your LLM is Meditron and you want to perform aligment tunning with CPO, run
     `python dpo.py ----model_name Meditron --setting CPO`

### LLMs inference

For inference, run `python indeference.py` 
1. Load the pretrained model in the python script e.g,.` checkpoint = "tmpbig/CPO_Meditron7b_reverse/checkpoint-1188"  `
2. Define also the task, i.e., either language2molecule or molecule2language e.g., ` mode = "smiles2caption" `

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
  title={Feedback-aligned Mixed LLMs for Machine Language-Molecule Translation},
  author={Gkoumas, Dimitris and Liakata, Maria},
  year={2024}
}
