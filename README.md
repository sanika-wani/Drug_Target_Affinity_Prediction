# Drug-Target Affinity Prediction

## Overview
This project implements a deep learning pipeline to predict **drug-target binding affinities**. The model combines graph-based representations of drug molecules with protein sequence embeddings to generate accurate predictions of binding strength. It leverages advanced neural architectures including Graph Neural Networks (GNNs) and ProtBERT for feature extraction, along with a cross-attention fusion mechanism.

## Features
- **Drug Representation:** Graph-based encoding using GINEConv with optional virtual nodes and flexible readout strategies.  
- **Protein Representation:** Sequence-based embeddings using the pretrained **ProtBERT** model.  
- **Fusion Mechanism:** Cross-attention module to integrate drug and protein features before regression.  
- **Performance Metrics:** Model performance is evaluated using RMSE, MSE, and Concordance Index (CI).  

## Dataset
- Trained and evaluated on the **KIBA dataset**, which contains drug-target interaction scores.  
- Data preprocessing includes feature extraction for drugs and tokenization for protein sequences.  

## Results
- Achieved **RMSE ≈ 0.70** on the KIBA dataset.  
- Supports evaluation with **RMSE** and **Train Loss**
- Caching of embeddings speeds up training and evaluation.  

## Future Improvements
- Experiment with additional protein embeddings or alternative GNN architectures.  
- Implement hyperparameter tuning for optimal model performance.  
- Explore transfer learning to adapt the model for other datasets.  

