# Conductivity_AEM
ncovering Structure–Conductivity Relationships in AEMs Using Interpretable Machine Learning
This repository accompanies the paper:

“Uncovering Structure–Conductivity Relationships in AEMs Using Interpretable Machine Learning”
P. Naghshnejad, J. A. Romagnoli, R. Kumar, J. J. Chen

We present a hybrid framework that integrates graph neural networks (GCN/GAT) and descriptor-based machine learning with unsupervised clustering and SHAP analysis to predict and interpret the ionic conductivity of anion exchange membranes (AEMs).
🚀 Workflow
1. Descriptor Generation
Compute Mordred molecular descriptors for all SMILES in the dataset:

bash
Copy
Edit
jupyter notebook notebooks/mordred.ipynb
2. Unsupervised Clustering & SHAP Analysis
Explore structure–performance relationships using t-SNE, DBSCAN, SOM, and SHAP:

bash
Copy
Edit
jupyter notebook notebooks/unsupervised_shap.ipynb
3. Train Graph Neural Network Models
Execute the GCN and GAT training pipeline:

bash
Copy
Edit
bash execute.sh
Or run manually:

bash
Copy
Edit
python src/GNN_workflow.py --data data/modified_data.csv --model GCNReg
f you use this repository, please cite
