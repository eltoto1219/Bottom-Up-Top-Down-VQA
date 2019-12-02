# Bottom-Up-Top-Down-VQA
This is a replication of the winning 2017 VQA model. This project more specifically investigates how different weight normalization techniques and the use of different parameter optimizers effect overall validation performance 

### Instructions for getting preprocessed data
Put all of this data in one directory and point the data_path config option in config.py to it <br/><br/>
Use this library: https://pypi.org/project/gdown/<br/>
For preprocessed question ID to answer, question, image ID, etc... dictionaries, run the following (.pkl): <br/>
`gdown https://drive.google.com/uc?id=1fyGqPZxEcz5sCYKNZQBvkD74q2Q6hhu2` <br/>
For GloVe token embeddings (300d), run the following (.pkl): <br/>
`gdown https://drive.google.com/uc?id=1rm_txmQD5t9-jxq5YDbhyNtQ6i7Sc3EM` <br/>
For preprocessed adaptive features, run the following (.h5): <br/>
`gdown https://drive.google.com/uc?id=1YBQFZOqIxEoXgd_5bBOvaskbje1tbCeP` <br/>
Training Annotations <br/>
`wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip` <br/>
Validation Annotations <br/>
`wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip` <br/>
Training Questions <br/>
`wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip`<br/>
Validation Questions <br/>
`wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip` <br/>
Test Questions <br/>
`wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip` <br/>

### General Config
See config.py 
### Preprocesed Question/Answer/Image Specifications
1. Max Question tokens: 14 
2. Quesiton processing: remove all special characters
3. Question padding: padding token = 0, rigth-hand-side padding
4. Answer processing: remove all special characters, remove all punctuation
5. Min Answer frequnecy: 9
6. Ansswer Score type: softscore (multiple answers per question, so we weight answer based on confidence)
7. Total Answers to predict: 3133
8. Tokenizer for Answers and Questions: Spacey Enlglish tokenizer
9. Image embedding dimension: 2048
10. Max number of objects per image: 100
11. Pre-computed image embeddings consist of Bottom up features on the MSCOCO dataset extracted using a Faster R-CNN object detection model trained on the Visual Genome dataset 









