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


🚀 Usage
1️⃣ Generate Molecular Descriptors
Compute Mordred descriptors for all AEM SMILES:

bash
Copy
Edit
jupyter notebook notebooks/mordred.ipynb
2️⃣ Perform Unsupervised Analysis
Explore clustering with t-SNE, DBSCAN, and SHAP:

bash
Copy
Edit
jupyter notebook notebooks/unsupervised_shap.ipynb
3️⃣ Train Graph Neural Networks
Run GCN/GAT models from terminal:

bash
Copy
Edit
bash execute.sh
Or manually specify a model:

bash
Copy
Edit
python src/GNN_workflow.py --data data/modified_data.csv --model GCNReg
📝 Citation
If you use this repository, please cite
