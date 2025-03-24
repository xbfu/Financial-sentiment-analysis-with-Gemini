## Introduction
Code for financial sentiment analysis using Germini


## Dependencies
- numpy==1.26.3  
- datasets==3.0.2
- google-genai==1.1.0
- scikit-learn==1.5.1

## Usage
##### 1. Install dependencies
```
conda create --name FSA -y python=3.9
conda activate FSA
pip install numpy==1.26.3 datasets==3.0.2 google-genai==1.1.0 scikit-learn==1.5.1
```
##### 2. Run code
```
python run_FSA.py --model_name=gemini-1.5-flash --dataset_name=fiqa
```
