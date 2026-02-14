# sentiment_analysis
version - 1.0
LICENSE - GNU AFFERO GENERAL PUBLIC LICENSE
built status - finish
description - Text classification based on BERT

## features
- fast train
- easy embed
- needed

## requirements for project
git
python3.12.10
pip
- for support GPU in project use nvidia cart
- my cart
nvidia rtx 3050

## installation
- in powershell or bash
git clone https://github.com/Dimasic-Cloud/sentiment_analysis.git
cd sentiment_analysis

- created venv (vertual environment)
--     windows
python -m venv your_name_for_vertual_environment
-- linux
python3 -m venv your_name_for_vertual_environment

- activate
-- windows
your_name_for_vertual_environment/scripts/activate
-- linux
source your_name_for_vertual_environment/bin/activate


- installing requirements
pip install -r requirements.txt

- installing pytorch GPU
pip install torch --index-url https://download.pytorch.org/whl/cu118


## usage
python train.py
- for train model and saved it into folder
python eval.py
- for download model and tokenizer and test model

## structure
sentiment-analysis/
    emotion_dataset_raw.csv
    eval.py
    LICENSE
    README.md
    requirements.txt
    test_dataset.csv
    train.py


# tech stack
python
torch
transformers
scikit-learn
pandas

## author
in profile github
