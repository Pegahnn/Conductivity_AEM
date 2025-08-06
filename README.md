# Conductivity_AEM
Uncovering Structure–Conductivity Relationships in AEMs Using Interpretable Machine Learning
This repository accompanies the paper:

“Uncovering Structure–Conductivity Relationships in AEMs Using Interpretable Machine Learning”
P. Naghshnejad, J. A. Romagnoli, R. Kumar, J. J. Chen

We propose a hybrid machine learning framework combining:

Graph Neural Networks (GCN & GAT) for learning molecular topology

Descriptor-based models (CatBoost, XGBoost, Random Forest) for interpretability

Unsupervised clustering (t-SNE, DBSCAN, SOM) to uncover structure–performance patterns

SHAP and saliency maps for chemical interpretability


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
