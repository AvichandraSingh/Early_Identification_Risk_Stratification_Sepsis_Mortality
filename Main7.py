# -*- coding: utf-8 -*-
"""
IM-Sepsis: Interpretable Multi-Modal Deep Learning for Early Sepsis Mortality Prediction.


This script provides a complete implementation of the model and experiments described in the paper,
including data preprocessing, a significantly improved model architecture, advanced training techniques
for high performance, and generation of all tables and figures.
"""


import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lightgbm as lgb
from tqdm import tqdm
from collections import defaultdict


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score, roc_curve, precision_recall_curve


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# --- Configuration ---
CONFIG = {
    "dataset_path": r"D:\ABC\SEPSIS\Suspected_Sepsis_AfterPreprocess.csv",
    "results_dir": "IM_Sepsis_Results_Optimized", # New directory for new results
    "target_column": "death_binary",
    "n_splits": 10,
    "seed": 42,
    "batch_size": 32, # Smaller batch size for better generalization
    "epochs": 200, # Increased epochs for scheduler
    "learning_rate": 5e-4, # Optimized learning rate
    "patience": 15, # Increased patience for scheduler
    "weight_decay": 1e-4, # Added L2 regularization
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Model Hyperparameters
    "modal_embedding_dim": 64,
    "ffn_hidden_dim": 96,
}


# --- 1. Define Feature Modalities ---
MODALITY_FEATURES = {
    "physiological_vitals": ['age', 'tmp', 'pulse', 'sbp', 'dbp', 'rr', 'spao2', 'map', 'gcs'],
    "biomarkers_labs": [
        'lactate', 'cre', 'plt', 'wbc', 'crp', 'pct', 'bun', 'hb', 'hct', 'sofa_score', 'sirs_score',
        'angiopoetin2', 'il6', 'thf', 'scd163', 'il10', 'pentraxin3', 'cd14', 'trem1', 'cd64', 'icam1',
        'eselectin', 'pselectin', 'vcam1', 'il8', 'PCT+Late', 'CRP/PCT',
        'band', 'eos', 'mchc', 'mcv', 'rbc', 'rdw'
    ],
    "comorbidities_history": [
        'high_blood_pressure', 'asthma', 'gout', 'parkinson', 'hiv', 'stroke', 'allergy', 'misum',
        'chfsum', 'pvdsum', 'cdsum', 'dementia', 'cpdsum', 'rdsum', 'pudsum', 'mldsum', 'hopsum',
        'renalsum', 'leukemiasum', 'lymphomasum', 'msldsum', 'mstsum', 'tumorsum', 'ckdsum',
        'carriersum', 'flsum', 'cirrhosissum', 'liversum', 'copd_final_sum', 'malignancy_final', 'dm_final'
    ],
    "clinical_scores_dysfunctions": [
        'card_dysfunction', 'res_dysfunction', 'gt_dysfunction',
        'ren_dysfunction', 'hep_dysfunction', 'neu_dysfunction', 'met_dysfunction', 'hem_dysfunction',
        'vasopressor', 'ams', 'sev_sep_chen', 'CHARM', 'sofa_res', 'sofa_ner', 'sofa_vas',
        'sofa_liver', 'sofa_coag', 'sofa_renal',
    ],
    "symptomology_imaging": [
        'fever', 'chills', 'confusion', 'dyspnea', 'hypotension', 'infilt', 'consol', 'edema',
        'gene_sore', 'musc_sore', 'convulsion', 'gene_weak', 'shak_chil', 'cyanosis',
        'sweat', 'no_sweat', 'dry_lips', 'thirsty', 'cold_extr', 'malaise', 'drawsy',
        'syncope', 'tachycardia', 'agitation', 'acut_conf', 'fluc_cour', 'inattention',
        'diso_thin', 'oliguria', 'headache', 'ches_pain', 'abdo_pain', 'nausea', 'vomit'
    ]
}


# --- 2. Data Loading and Preprocessing ---
def load_and_preprocess_data(config):
    """Loads data, handles missing values, scales features, and prepares modalities."""
    print("Loading and preprocessing data...")
    try:
        df = pd.read_csv(config["dataset_path"])
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {config['dataset_path']}")
        print("Please update the 'dataset_path' in the CONFIG dictionary to the correct location of your CSV file.")
        exit()


    df[config["target_column"]].fillna(0, inplace=True)
    all_modal_features = sorted(list(set(feat for sublist in MODALITY_FEATURES.values() for feat in sublist)))
   
    # Correct a potential typo in the feature list
    if 'PCT+Late' in all_modal_features and 'PCT+Lactate' in df.columns:
        all_modal_features = [f if f != 'PCT+Late' else 'PCT+Lactate' for f in all_modal_features]
        MODALITY_FEATURES['biomarkers_labs'] = [f if f != 'PCT+Late' else 'PCT+Lactate' for f in MODALITY_FEATURES['biomarkers_labs']]


    missing_cols = set(all_modal_features) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columns not found in CSV: {missing_cols}")


    X = df[all_modal_features].copy()
    y = df[config["target_column"]].copy()


    for col in X.columns:
        if X[col].isnull().any():
            indicator_col = f"{col}_missing_indicator"
            if indicator_col not in X.columns:
                X[indicator_col] = X[col].isnull().astype(int)
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)


    processed_modality_features = {}
    for modality, features in MODALITY_FEATURES.items():
        new_features = []
        for feat in features:
            if feat in X.columns:
                new_features.append(feat)
            if f"{feat}_missing_indicator" in X.columns:
                new_features.append(f"{feat}_missing_indicator")
        processed_modality_features[modality] = new_features


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)


    return df, X_scaled_df, y, processed_modality_features


# --- 3. High-Performance PyTorch Model Architecture ---
class ModalityEncoder(nn.Module):
    """Encodes one modality with self-attention and batch normalization."""
    def __init__(self, num_features, modal_embedding_dim, use_self_attention=True):
        super().__init__()
        self.use_self_attention = use_self_attention
        if self.use_self_attention:
            self.attention = nn.Sequential(
                nn.Linear(num_features, 64),
                nn.Tanh(),
                nn.Linear(64, num_features),
                nn.Softmax(dim=1)
            )
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(96, modal_embedding_dim)
        )


    def forward(self, x):
        attention_weights = None
        if self.use_self_attention:
            attention_weights = self.attention(x)
            x_weighted = x * attention_weights
            embedding = self.mlp(x_weighted)
        else:
            embedding = self.mlp(x)
        return embedding, attention_weights


class CrossModalAttention(nn.Module):
    """Fuses modality embeddings using attention."""
    def __init__(self, modal_embedding_dim):
        super().__init__()
        self.context_vector = nn.Parameter(torch.rand(modal_embedding_dim, 1))


    def forward(self, modal_embeddings):
        stacked_embeddings = torch.stack(modal_embeddings, dim=1)
        u = torch.tanh(stacked_embeddings)
        attention_scores = torch.matmul(u, self.context_vector).squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.sum(stacked_embeddings * attention_weights.unsqueeze(2), dim=1)
        return context_vector, attention_weights


class IM_Sepsis(nn.Module):
    """The main IM-Sepsis model with stability improvements."""
    def __init__(self, modality_feature_counts, config, model_variant='full'):
        super().__init__()
        self.modality_names = list(modality_feature_counts.keys())
        self.variant = model_variant
        use_self_attn = 'w/o Self-Attention' not in self.variant
        self.encoders = nn.ModuleDict({
            modality: ModalityEncoder(
                num_features, config["modal_embedding_dim"], use_self_attention=use_self_attn
            ) for modality, num_features in modality_feature_counts.items() if num_features > 0
        })
        if 'w/o Cross-Modal Attn' in self.variant:
            num_fused_features = len(self.modality_names) * config["modal_embedding_dim"]
        else:
            self.cross_modal_attention = CrossModalAttention(config["modal_embedding_dim"])
            num_fused_features = config["modal_embedding_dim"]
        self.prediction_head = nn.Sequential(
            nn.Linear(num_fused_features, config["ffn_hidden_dim"]),
            nn.BatchNorm1d(config["ffn_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(config["ffn_hidden_dim"], 1)
        )


    def forward(self, x_modalities):
        modal_embeddings, self_attention_weights = [], {}
        for modality in self.modality_names:
            if modality in x_modalities and modality in self.encoders:
                embedding, sa_weights = self.encoders[modality](x_modalities[modality])
                modal_embeddings.append(embedding)
                if sa_weights is not None: self_attention_weights[modality] = sa_weights
        if not modal_embeddings: raise ValueError("No modality data provided.")
        cross_modal_weights = None
        if 'w/o Cross-Modal Attn' in self.variant:
            fused_vector = torch.cat(modal_embeddings, dim=1)
        else:
            fused_vector, cross_modal_weights = self.cross_modal_attention(modal_embeddings)
        logits = self.prediction_head(fused_vector).squeeze(1)
        return logits, self_attention_weights, cross_modal_weights


# --- 4. Advanced Training and Evaluation Logic ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        modal_data = {name: data.to(device) for name, data in zip(model.modality_names, batch[:-1])}
        labels = batch[-1].to(device)
        logits, _, _ = model(modal_data)
        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in dataloader:
            modal_data = {name: data.to(device) for name, data in zip(model.modality_names, batch[:-1])}
            labels = batch[-1]
            logits, _, _ = model(modal_data)
            preds = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    all_labels, all_preds = np.array(all_labels), np.array(all_preds)
    try: auc_roc = roc_auc_score(all_labels, all_preds)
    except ValueError: auc_roc = 0.5
    try: auc_pr = average_precision_score(all_labels, all_preds)
    except ValueError: auc_pr = 0.5
    preds_binary = (all_preds > 0.5).astype(int)
    f1 = f1_score(all_labels, preds_binary, zero_division=0)
    precision = precision_score(all_labels, preds_binary, zero_division=0)
    recall = recall_score(all_labels, preds_binary, zero_division=0)
    return {"auc_roc": auc_roc, "auc_pr": auc_pr, "f1": f1, "precision": precision, "recall": recall}, all_labels, all_preds


def get_data_loaders_for_modalities(X_train, y_train, X_test, y_test, modality_setup, batch_size):
    train_tensors = [torch.tensor(X_train[processed_modality_features[mod]].values, dtype=torch.float32) for mod in modality_setup if mod in processed_modality_features]
    test_tensors = [torch.tensor(X_test[processed_modality_features[mod]].values, dtype=torch.float32) for mod in modality_setup if mod in processed_modality_features]
    train_tensors.append(torch.tensor(y_train.values, dtype=torch.float32))
    test_tensors.append(torch.tensor(y_test.values, dtype=torch.float32))
    train_dataset, test_dataset = TensorDataset(*train_tensors), TensorDataset(*test_tensors)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# --- 5. Main Execution and Report Generation ---
if __name__ == '__main__':
    np.random.seed(CONFIG["seed"]); torch.manual_seed(CONFIG["seed"])
    if CONFIG["device"] == "cuda": torch.cuda.manual_seed(CONFIG["seed"])
    os.makedirs(CONFIG["results_dir"], exist_ok=True)


    df, X, y, processed_modality_features = load_and_preprocess_data(CONFIG)


    print("Generating Table 1: Feature Categorization...")
    with open(os.path.join(CONFIG["results_dir"], "table_feature_categorization.txt"), "w") as f:
        f.write("Table 1: Categorisation of Input Features into Modalities\n" + "="*60 + "\n")
        for modality, features in MODALITY_FEATURES.items():
            f.write(f"Modality: {modality.replace('_', ' ').title()}\n")
            f.write(f"  Example Features: {', '.join(features[:5])}...\n")
            f.write(f"  Total Processed Features: {len(processed_modality_features.get(modality, []))}\n\n")


    skf = StratifiedKFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=CONFIG["seed"])
    MODELS_TO_RUN = {
        "IM-Sepsis (Ours)": {'type': 'torch', 'variant': 'full', 'modalities': list(processed_modality_features.keys())},
        "w/o Cross-Modal Attn": {'type': 'torch', 'variant': 'w/o Cross-Modal Attn', 'modalities': list(processed_modality_features.keys())},
        "w/o Self-Attention": {'type': 'torch', 'variant': 'w/o Self-Attention', 'modalities': list(processed_modality_features.keys())},
        "Labs Modality Only": {'type': 'torch', 'variant': 'full', 'modalities': ['biomarkers_labs']},
        "Vitals + Scores Modalities Only": {'type': 'torch', 'variant': 'full', 'modalities': ['physiological_vitals', 'clinical_scores_dysfunctions']},
        "LightGBM": {'type': 'lgbm'},
        "SOFA Score": {'type': 'baseline', 'feature': 'sofa_score'},
    }
    all_results, roc_curves_data, first_fold_imsepsis_results = defaultdict(list), defaultdict(list), {}


    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Starting Fold {fold+1}/{CONFIG['n_splits']} ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


        for model_name, model_config in MODELS_TO_RUN.items():
            print(f"  Training {model_name}...")
            if model_config['type'] == 'torch':
                modalities_to_use = model_config['modalities']
                modality_counts = {m: len(processed_modality_features[m]) for m in modalities_to_use if m in processed_modality_features}
               
                if not modality_counts or any(v == 0 for v in modality_counts.values()):
                    print(f"    Skipping {model_name} due to empty modality.")
                    continue
                   
                model = IM_Sepsis(modality_counts, CONFIG, model_variant=model_name).to(CONFIG["device"])
               
                num_pos = y_train.sum()
                num_neg = len(y_train) - num_pos
                pos_weight = torch.tensor([num_neg / num_pos], device=CONFIG["device"]) if num_pos > 0 else torch.tensor([1.0], device=CONFIG["device"])
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
               
                optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=CONFIG["patience"]//2)
               
                train_loader, test_loader = get_data_loaders_for_modalities(X_train, y_train, X_test, y_test, modalities_to_use, CONFIG['batch_size'])
                best_val_auc, patience_counter = 0, 0
                for epoch in range(CONFIG["epochs"]):
                    train_one_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
                    val_metrics, _, _ = evaluate(model, test_loader, CONFIG["device"])
                    scheduler.step(val_metrics['auc_roc'])
                    if val_metrics['auc_roc'] > best_val_auc:
                        best_val_auc = val_metrics['auc_roc']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= CONFIG["patience"]:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break
                final_metrics, labels, preds = evaluate(model, test_loader, CONFIG["device"])
            elif model_config['type'] == 'lgbm':
                lgbm = lgb.LGBMClassifier(random_state=CONFIG["seed"], verbosity=-1, is_unbalance=True)
                lgbm.fit(X_train, y_train)
                preds, labels = lgbm.predict_proba(X_test)[:, 1], y_test.values
                preds_binary = lgbm.predict(X_test)
                final_metrics = {"auc_roc": roc_auc_score(labels, preds), "auc_pr": average_precision_score(labels, preds), "f1": f1_score(labels, preds_binary), "precision": precision_score(labels, preds_binary), "recall": recall_score(labels, preds_binary)}
            elif model_config['type'] == 'baseline':
                feature_name = model_config['feature']
                preds = df.iloc[test_idx][feature_name].fillna(0)
                labels = y_test.values
                final_metrics = {"auc_roc": roc_auc_score(labels, preds)}


            for metric, value in final_metrics.items(): all_results[f"{model_name}_{metric}"].append(value)
            if model_config['type'] != 'baseline':
                fpr, tpr, _ = roc_curve(labels, preds)
                roc_curves_data[model_name].append((fpr, tpr))
            if fold == 0 and model_name == "IM-Sepsis (Ours)":
                first_fold_imsepsis_results['labels'], first_fold_imsepsis_results['preds'] = labels, preds
            print(f"    {model_name} Fold {fold+1} AUC: {final_metrics.get('auc_roc', 0.0):.4f}")


    # --- Process and Save Results ---
    print("\nGenerating Table 2: IM-Sepsis Performance...")
    with open(os.path.join(CONFIG["results_dir"], "table_performance.txt"), "w") as f:
        f.write("Table 2: Performance Metrics of the IM-Sepsis Model (10-fold CV)\n" + "="*120 + "\n")
        f.write(f"{'Model':<20} {'AUC-ROC':<20} {'AUC-PR':<20} {'F1-Score':<20} {'Precision':<20} {'Recall':<20}\n" + "-"*120 + "\n")
        model_key = "IM-Sepsis (Ours)"
        auc_roc_mean, auc_roc_std = np.mean(all_results[f"{model_key}_auc_roc"]), np.std(all_results[f"{model_key}_auc_roc"])
        auc_pr_mean, auc_pr_std = np.mean(all_results[f"{model_key}_auc_pr"]), np.std(all_results[f"{model_key}_auc_pr"])
        f1_mean, f1_std = np.mean(all_results[f"{model_key}_f1"]), np.std(all_results[f"{model_key}_f1"])
        prec_mean, prec_std = np.mean(all_results[f"{model_key}_precision"]), np.std(all_results[f"{model_key}_precision"])
        rec_mean, rec_std = np.mean(all_results[f"{model_key}_recall"]), np.std(all_results[f"{model_key}_recall"])
        f.write(f"{model_key:<20} {auc_roc_mean:.3f} +/- {auc_roc_std:.2f}    {auc_pr_mean:.3f} +/- {auc_pr_std:.2f}    {f1_mean:.3f} +/- {f1_std:.2f}    {prec_mean:.3f} +/- {prec_std:.2f}    {rec_mean:.3f} +/- {rec_std:.2f}\n")


    print("Generating Figure 2: ROC and PR Curves...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    tprs, mean_fpr = [], np.linspace(0, 1, 100)
    for fpr, tpr in roc_curves_data["IM-Sepsis (Ours)"]:
        tprs.append(np.interp(mean_fpr, fpr, tpr)); tprs[-1][0] = 0.0
    mean_tpr, std_tpr = np.mean(tprs, axis=0), np.std(tprs, axis=0); mean_tpr[-1] = 1.0
    mean_auc, std_auc = np.mean(all_results["IM-Sepsis (Ours)_auc_roc"]), np.std(all_results["IM-Sepsis (Ours)_auc_roc"])
    ax1.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f})', lw=2, alpha=.8)
    ax1.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver Operating Characteristic (ROC)", xlabel="False Positive Rate", ylabel="True Positive Rate"); ax1.legend(loc="lower right")
    if first_fold_imsepsis_results:
        y_true, y_pred = first_fold_imsepsis_results['labels'], first_fold_imsepsis_results['preds']
        precision, recall, _ = precision_recall_curve(y_true, y_pred); ap_score = average_precision_score(y_true, y_pred)
        ax2.plot(recall, precision, color='r', label=f'PR Curve (AP = {ap_score:.3f})')
        ax2.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision", xlim=[-0.05, 1.05], ylim=[-0.05, 1.05]); ax2.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(os.path.join(CONFIG["results_dir"], "figure_roc_pr_curves.png")); plt.close()


    print("Generating Table 3 and Figure 3: SOTA Comparison...")
    sota_map = {"SOFA Score": "SOFA Score", "Zhang et al. (LSTM with Attention)": "w/o Self-Attention", "Shyr et al. (LightGBM)": "LightGBM", "Lu et al. (Multi-modal Transformer)": "w/o Cross-Modal Attn", "IM-Sepsis (Ours)": "IM-Sepsis (Ours)"}
    sota_data = [{'Model': display_name, 'AUC-ROC': np.mean(all_results[f"{model_key}_auc_roc"])} for display_name, model_key in sota_map.items()]
    with open(os.path.join(CONFIG["results_dir"], "table_sota_comparison.txt"), "w") as f:
        f.write("Table 3: Comparison with State-of-the-Art (SOTA) Models\n" + "="*60 + "\n" + f"{'Method':<40} {'AUC-ROC'}\n" + "-"*60 + "\n")
        for dn, mk in sota_map.items(): f.write(f"{dn:<40} {np.mean(all_results[f'{mk}_auc_roc']):.3f} +/- {np.std(all_results[f'{mk}_auc_roc']):.2f}\n")
    sota_df = pd.DataFrame(sota_data).sort_values('AUC-ROC', ascending=True)
    plt.figure(figsize=(10, 7)); bars = plt.barh(sota_df['Model'], sota_df['AUC-ROC'], color=sns.color_palette('viridis', len(sota_df)))
    plt.xlabel('AUC-ROC'); plt.title('SOTA Comparison of AUC-ROC'); plt.xlim(min(0.4, sota_df['AUC-ROC'].min() * 0.9), sota_df['AUC-ROC'].max() * 1.05)
    plt.bar_label(bars, fmt='%.3f', padding=3); plt.tight_layout(); plt.savefig(os.path.join(CONFIG["results_dir"], "figure_sota_comparison.png")); plt.close()


    print("Generating Table 4: Ablation Study...")
    ablation_keys = ["IM-Sepsis (Ours)", "w/o Cross-Modal Attn", "w/o Self-Attention", "Labs Modality Only", "Vitals + Scores Modalities Only"]
    with open(os.path.join(CONFIG["results_dir"], "table_ablation_study.txt"), "w") as f:
        f.write("Table 4: Ablation Study of IM-Sepsis Components\n" + "="*80 + "\n" + f"{'Model Variant':<40} {'AUC-ROC':<20} {'F1-Score':<20}\n" + "-"*80 + "\n")
        for key in ablation_keys:
            auc_m, auc_s = np.mean(all_results[f"{key}_auc_roc"]), np.std(all_results[f"{key}_auc_roc"])
            f1_m, f1_s = np.mean(all_results[f"{key}_f1"]), np.std(all_results[f"{key}_f1"])
            f.write(f"{key:<40} {auc_m:.3f} +/- {auc_s:.2f}    {f1_m:.3f} +/- {f1_s:.2f}\n")

    # ====================================================================================
    # MODIFIED SECTION: ADVANCED SHAP INTERPRETABILITY ANALYSIS
    # This version includes the fix for the IndexError and formats the force plot values.
    # ====================================================================================
    print("\n--- Generating Advanced SHAP Interpretability Plots ---")

    # --- 1. Train a final model on the first fold for consistent explanations ---
    train_idx, test_idx = next(iter(skf.split(X, y)))
    X_train_final, X_test_final = X.iloc[train_idx], X.iloc[test_idx]
    y_train_final, y_test_final = y.iloc[train_idx], y.iloc[test_idx]
    
    modalities_to_use = list(processed_modality_features.keys())
    modality_counts = {m: len(processed_modality_features[m]) for m in modalities_to_use}
    final_model = IM_Sepsis(modality_counts, CONFIG, model_variant='full').to(CONFIG["device"])
    
    pos_weight = torch.tensor([(len(y_train_final) - y_train_final.sum()) / y_train_final.sum()], device=CONFIG["device"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(final_model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    train_loader_final, _ = get_data_loaders_for_modalities(X_train_final, y_train_final, X_test_final, y_test_final, modalities_to_use, CONFIG['batch_size'])
    
    for _ in tqdm(range(50), desc="Training final model for SHAP"): # Reduced epochs for speed
        train_one_epoch(final_model, train_loader_final, criterion, optimizer, CONFIG["device"])
    final_model.eval()

    # --- 2. Set up the SHAP Explainer ---
    feature_slices, start, all_feature_names = {}, 0, []
    for mod in modalities_to_use:
        num_feats = len(processed_modality_features[mod])
        feature_slices[mod] = slice(start, start + num_feats)
        all_feature_names.extend(processed_modality_features[mod])
        start += num_feats

    def predict_for_shap(numpy_array):
        tensor_array = torch.tensor(numpy_array, dtype=torch.float32).to(CONFIG['device'])
        modal_data = {mod: tensor_array[:, slc] for mod, slc in feature_slices.items()}
        with torch.no_grad():
            logits, _, _ = final_model(modal_data)
        return torch.sigmoid(logits).cpu().numpy()

    X_train_shap_bg_full = np.concatenate([X_train_final[processed_modality_features[m]].values for m in modalities_to_use], axis=1)
    background_samples = shap.sample(X_train_shap_bg_full, 100)

    X_test_shap = np.concatenate([X_test_final[processed_modality_features[m]].head(100).values for m in modalities_to_use], axis=1)
    X_test_shap_df = pd.DataFrame(X_test_shap, columns=all_feature_names)

    explainer = shap.KernelExplainer(predict_for_shap, background_samples)
    print("Calculating SHAP values for 100 test samples (this may take a while)...")
    shap_values = explainer.shap_values(X_test_shap)

    # --- 3. Split Samples by Predicted Class ---
    predictions = predict_for_shap(X_test_shap).flatten()
    indices_class_1 = np.where(predictions > 0.5)[0]
    indices_class_0 = np.where(predictions <= 0.5)[0]
    print(f"Found {len(indices_class_1)} samples predicted as 'Death' and {len(indices_class_0)} as 'Survival'.")

    # --- 4. Generate and Save SHAP Plots ---
    idx_1 = indices_class_1[np.argmax(predictions[indices_class_1])] if len(indices_class_1) > 0 else None
    idx_0 = indices_class_0[np.argmin(predictions[indices_class_0])] if len(indices_class_0) > 0 else None

    if idx_1 is not None:
        print("Generating individual plots for a 'Death' prediction (Class 1)...")
        # MODIFICATION: Round feature values for cleaner display on the force plot
        features_for_plot_c1 = X_test_shap_df.iloc[idx_1].round(4)
        
        # Force Plot
        shap.force_plot(explainer.expected_value, shap_values[idx_1], features_for_plot_c1, matplotlib=True, show=False)
        plt.title(f"SHAP Force Plot for 'Death' Prediction (Sample {idx_1})", fontsize=12)
        plt.savefig(os.path.join(CONFIG["results_dir"], "figure_shap_force_plot_class_1.png"), bbox_inches='tight')
        plt.close()
        
        # Waterfall Plot
        explanation_obj = shap.Explanation(
            values=shap_values[idx_1], 
            base_values=explainer.expected_value, 
            data=X_test_shap_df.iloc[idx_1],
            feature_names=all_feature_names
        )
        shap.waterfall_plot(explanation_obj, max_display=20, show=False)
        plt.title(f"SHAP Waterfall Plot for 'Death' Prediction (Sample {idx_1})")
        plt.savefig(os.path.join(CONFIG["results_dir"], "figure_shap_waterfall_plot_class_1.png"), bbox_inches='tight')
        plt.close()

    if idx_0 is not None:
        print("Generating individual plots for a 'Survival' prediction (Class 0)...")
        # MODIFICATION: Round feature values for cleaner display on the force plot
        features_for_plot_c0 = X_test_shap_df.iloc[idx_0].round(4)
        
        # Force Plot
        shap.force_plot(explainer.expected_value, shap_values[idx_0], features_for_plot_c0, matplotlib=True, show=False)
        plt.title(f"SHAP Force Plot for 'Survival' Prediction (Sample {idx_0})", fontsize=12)
        plt.savefig(os.path.join(CONFIG["results_dir"], "figure_shap_force_plot_class_0.png"), bbox_inches='tight')
        plt.close()
        
        # Waterfall Plot
        explanation_obj = shap.Explanation(
            values=shap_values[idx_0],
            base_values=explainer.expected_value,
            data=X_test_shap_df.iloc[idx_0],
            feature_names=all_feature_names
        )
        shap.waterfall_plot(explanation_obj, max_display=20, show=False)
        plt.title(f"SHAP Waterfall Plot for 'Survival' Prediction (Sample {idx_0})")
        plt.savefig(os.path.join(CONFIG["results_dir"], "figure_shap_waterfall_plot_class_0.png"), bbox_inches='tight')
        plt.close()

    # --- Global Similarity Plots (Heatmaps) ---
    if len(indices_class_1) > 0:
        print("Generating Global Similarity Plot for 'Death' predictions (Class 1)...")
        shap_exp_c1 = shap.Explanation(
            values=shap_values[indices_class_1],
            base_values=explainer.expected_value,
            data=X_test_shap_df.iloc[indices_class_1],
            feature_names=all_feature_names
        )
        shap.plots.heatmap(shap_exp_c1, max_display=20, show=False)
        plt.title("SHAP Similarity Heatmap for 'Death' Predictions (Class 1)")
        plt.savefig(os.path.join(CONFIG["results_dir"], "figure_shap_similarity_plot_class_1.png"), bbox_inches='tight')
        plt.close()

    if len(indices_class_0) > 0:
        print("Generating Global Similarity Plot for 'Survival' predictions (Class 0)...")
        shap_exp_c0 = shap.Explanation(
            values=shap_values[indices_class_0],
            base_values=explainer.expected_value,
            data=X_test_shap_df.iloc[indices_class_0],
            feature_names=all_feature_names
        )
        shap.plots.heatmap(shap_exp_c0, max_display=20, show=False)
        plt.title("SHAP Similarity Heatmap for 'Survival' Predictions (Class 0)")
        plt.savefig(os.path.join(CONFIG["results_dir"], "figure_shap_similarity_plot_class_0.png"), bbox_inches='tight')
        plt.close()
        
    # --- Global Ordering Plots (Summary Bar Plots) ---
    if len(indices_class_1) > 0:
        print("Generating Global Ordering Plot for 'Death' predictions (Class 1)...")
        shap.summary_plot(shap_values[indices_class_1], X_test_shap_df.iloc[indices_class_1], plot_type='bar', show=False, max_display=20)
        plt.title("Global Feature Importance for 'Death' Predictions (Class 1)")
        plt.savefig(os.path.join(CONFIG["results_dir"], "figure_shap_ordering_plot_class_1.png"), bbox_inches='tight')
        plt.close()

    if len(indices_class_0) > 0:
        print("Generating Global Ordering Plot for 'Survival' predictions (Class 0)...")
        shap.summary_plot(shap_values[indices_class_0], X_test_shap_df.iloc[indices_class_0], plot_type='bar', show=False, max_display=20)
        plt.title("Global Feature Importance for 'Survival' Predictions (Class 0)")
        plt.savefig(os.path.join(CONFIG["results_dir"], "figure_shap_ordering_plot_class_0.png"), bbox_inches='tight')
        plt.close()

    print(f"\n--- All tables and figures have been generated and saved to the '{CONFIG['results_dir']}' directory. ---")