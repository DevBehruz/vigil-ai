## VigilAI — Harmful Message Detection with NLP

VigilAI is a machine learning system for detecting abusive and explicit content in text messages. The project demonstrates an end-to-end natural language processing pipeline for content safety using classical machine learning techniques and feature engineering.

## Purpose

The goal of VigilAI is to automatically identify harmful language in online communication to support safer digital platforms, moderation tools, and research on content safety.

## Technologies Used

- Python  
- Scikit-learn  
- Natural Language Processing (NLP)  
- TF-IDF and N-gram feature engineering  
- Classical ML models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
  - Gradient Boosting  

## Key Features

- Text preprocessing and cleaning  
- Multiple feature extraction methods  
- Training and evaluation of several classifiers  
- Modular pipeline for experimentation  

## Benefits

- Helps detect abusive or unsafe messages at scale  
- Supports automated moderation systems  
- Provides a foundation for building advanced AI safety tools  
- Easily extensible to deep learning and transformer models  

## Usage

Run the training pipeline:

```
python main.py
```

## Project Structure

```
src/        Core modules for data processing, features, and models  
notebooks/  Experimental notebook  
main.py     Entry point  
```
