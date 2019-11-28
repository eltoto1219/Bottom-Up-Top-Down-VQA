# Bottom-Up-Top-Down-VQA
This is a replication of the winning 2017 VQA model. This project more specifically investigates how different weight normalization techniques and the use of different parameter optimizers effect overall validation performance 

### Instructions for getting preprocessed data
Use this library: https://pypi.org/project/gdown/<br/>
For preprocessed question ID to answer, question, image ID, etc... dictionaries, run the following (.pkl): <br/>
`gdown https://drive.google.com/uc?id=1c5GXpL3p8YoYq3FIMDBwMjP684k1Gt5g` <br/>
For GloVe toekn embeddings (300d), run the following (.pkl): <br/>
`gdown https://drive.google.com/uc?id=1rm_txmQD5t9-jxq5YDbhyNtQ6i7Sc3EM` <br/>
For preprocessed adaptive features, run the following (.h5): <br/>
`gdown https://drive.google.com/uc?id=1YBQFZOqIxEoXgd_5bBOvaskbje1tbCeP` <br/>
### General Config
See config.py 
### Preprocesed Question/Answer Specifications
1. Max Question tokens: 14 
2. Quesiton processing: remove all special characters
3. Question padding: padding token = 0, righ-hand-side padding
4. Answer processing: remove all special characters, remove all punctuation
5. Min Answer freqecny: 9
6. Ansswer Score type: softscore (multiple answers per question, so we weight answer based on confidence)
7. Total Answers to predict: 3133









