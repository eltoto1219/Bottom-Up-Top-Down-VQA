# Bottom-Up-Top-Down-VQA
This is a replication of the winning 2017 VQA model. This project more specifically investigates how different weight normalization techniques and the use of different parameter optimizers effect overall validation performance 
### Instructions for getting preprocessed data
set the `data_root` argument in config.py to ./data. 
wget the following in the ./data directory and unzip. <br/>
For preprocessed question ID to answer, question, image ID, etc... dictionaries, run the following (.pkl): <br/>
`wget https://vqa-dnac.s3.amazonaws.com/dicts/dicts.zip` <br/>
For GloVe token embeddings (300d), run the following (.pkl): <br/>
`wget https://vqa-dnac.s3.amazonaws.com/token-weights/token_weights.zip` <br/>
For preprocessed adaptive features, run the following (.h5): <br/>
`wget https://vqa-dnac.s3.amazonaws.com/features/adaptive_features.zip` <br/>
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
### Run Expirement
To run expirements: <br>
`./scripts/EXP_NAME.sh`
For Slrum: <br>
`sbatch slurm/EXP_NAME.sh`












