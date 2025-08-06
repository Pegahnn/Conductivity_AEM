 python ./src/GNN_workflow.py --gpu 0 --train --randSplit --path './models/EarlyStop_rd42/' --data './modified_data.csv' --seed 42 --test_size 0.2 --epochs 1000 --early_stop --epochs 1000  --add_features --num_feat 4 --gnn_model GCNReg_add  --batch_size 20
 python ./src/opt.py --model GCNReg_add --add_features --num_feat 4 --split 0.2 --n_trials 200 --data_path './modified_data.csv' --save_dir './reports/GCN' 
 [I 2025-05-08 17:33:04,491] Trial 199 finished with value: 5.697591041098349e-05 and parameters: {'batch_size': 32, 'lr': 0.004614787496001549, 'unit_per_layer': 256, 'epochs': 1500, 'patience': 40, 'seed': 42}. Best is trial 139 with value: 4.321204869484063e-06.

Best trial:
  Value: 4.321204869484063e-06
  Params:
    batch_size: 32
    lr: 0.004446263137859366
    unit_per_layer: 512
    epochs: 1500
    patience: 20
    seed: 2024
python ./src/GNN_workflow.py --gpu 0 --train --randSplit --path './models/EarlyStop_bestparams_skipcv/' --data './modified_data.csv' --seed 2024 --test_size 0.2 --epochs 1500 --early_stop --add_features --num_feat 4 --gnn_model GCNReg_add --batch_size 32 --lr 0.004446263137859366 --unit_per_layer 512 --patience 20 --skip_cv
python ./src/GCN/GNNparameterTuning4_GCN_add.py --model GCNReg_add --split 0.1 --n_trials 200 --data_path './modified_data.csv'  --save_dir './reports/sur_tent/GCN_add_opt' --num_feat 4

Best trial:
  Value: 0.015532042416699159
  Params:
    batch_size: 5
    lr: 0.002153475691636952
    unit_per_layer: 384
    epochs: 500
    patience: 30
    seed: 2021
    seed: 2021
    weight_decay: 1.48963565767138e-06
    optimizer: adamw
    seed: 2021
    weight_decay: 1.48963565767138e-06
    seed: 2021
    seed: 2021
    weight_decay: 1.48963565767138e-06
    seed: 2021
    weight_decay: 1.48963565767138e-06
    seed: 2021
    weight_decay: 1.48963565767138e-06
    seed: 2021
    weight_decay: 1.48963565767138e-06
    optimizer: adamw
    seed: 2021
    weight_decay: 1.48963565767138e-06
    optimizer: adamw
    scheduler: none
python ./src/GCN/GNNparameterTuning4_GCN_add.py --model GCNReg_add --split 0.1 --n_trials 200 --data_path './modified_data.csv'  --save_dir './reports/sur_tent/GCN_add_opt' --num_feat 
python ./src/GCN/GNN_workflow.py --gpu 0 --train --randSplit --path './models/EarlyStop_2021_skipcv/' --data './modified_data.csv' --seed 2021 --test_size 0.1 --epochs 1000  --add_features --num_feat 4 --gnn_model GCNReg_add  --skip_cv
python ./src/GCN/GNN_workflow.py --gpu 0 --train --randSplit --path './models/EarlyStop_42_skipcv/' --data './modified_data.csv' --seed 42 --test_size 0.1 --epochs 1000  --add_features --num_feat 4 --gnn_model GCNReg_add  --skip_cv

python ./src/GCN/GNN_workflow.py --gpu 0 --train --randSplit --path './models/EarlyStop_28_skipcv/' --data './modified_data.csv' --seed 28 --test_size 0.1 --epochs 1000  --add_features --num_feat 4 --gnn_model GCNReg_add  --skip_cv
python ./src/GCN/GNN_workflow.py --gpu 0 --train --randSplit --path './models/EarlyStop_2020_skipcv/' --data './modified_data.csv' --seed 2020 --test_size 0.1 --epochs 1000  --add_features --num_feat 4 --gnn_model GCNReg_add  --skip_cv
./src/GCN/GNN_workflow.py --gpu 0 --train --randSplit --path './models/EarlyStop_2020_skipcv/' --data './modified_data.csv' --seed 2020 --test_size 0.1 --epochs 1000  --add_features --num_feat 4 --gnn_model GCNReg_add  --skip_cv
python ./src/GCN/GNN_workflow.py --gpu 0 --train --randSplit --path './models/EarlyStop_2023_skipcv/' --data './modified_data.csv' --seed 2023 --test_size 0.1 --epochs 1000  --add_features --num_feat 4 --gnn_model GCNReg_add  --skip_cv
python ./src/GCN/GNN_workflow.py --gpu 0 --train --randSplit --path './models/EarlyStop_1337_skipcv/' --data './modified_data.csv' --seed 1337 --test_size 0.1 --epochs 1000  --add_features --num_feat 4 --gnn_model GCNReg_add  --skip_cv
python ./src/GCN/GNN_workflow.py --gpu 0 --train --randSplit --path './models/EarlyStop_999_skipcv/' --data './modified_data.csv' --seed 999 --test_size 0.1 --epochs 1000  --add_features --num_feat 4 --gnn_model GCNReg_add  --skip_cv
#opt
python ./src/GCN/GNNparameterTuning4_GCN_add.py --model GCNReg_add --split 0.1 --n_trials 200 --data_path './modified_data.csv'  --save_dir './reports/sur_tent/GCN_add_opt2023' --num_feat 4 
#result
183	0.007504115	6	0.000842118	256	500	40	2023	2.94E-06	adamw	cosine
python ./src/GCN3/GNNparameterTuning4_GAT_add5.py --model GATReg_add --split 0.1 --n_trials 200 --data_path './modified_data.csv'  --save_dir './reports/sur_tent/GAT_add_opt2023' --num_feat 4 
Trial 199 finished with value: inf and parameters: {'batch_size': 4, 'lr': 0.0013849735185198885, 'unit_per_layer': 512, 'epochs': 1500, 'patience': 40, 'seed': 2023, 'weight_decay': 0.00038611107827960596, 'optimizer': 'adam', 'scheduler': 'cosine'}. Best is trial 142 with value: 0.014175273275294656.

Best trial:
  Value: 0.014175273275294656
  Params:
    batch_size: 8
    lr: 0.0011928650474374166
    unit_per_layer: 128
    epochs: 1500
    patience: 20
    seed: 2023
    weight_decay: 1.5003028709004243e-06
    optimizer: adamw
    scheduler: cosine
  
  python ./src/GCN3/GNNparameterTuning4_GAT_add5.py --model GATReg_add --split 0.1 --n_trials 200 --data_path './modified_data_added.csv'  --save_dir './reports/sur_tent/GAT_add_opt2023' --num_feat 11 
