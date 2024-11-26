import json
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
from pysentimiento.preprocessing import preprocess_tweet
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Union
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import torch.nn.functional as F

# ---- CONFIGURAÇÕES ----

@dataclass
class LoadDataConfig:
    json_path: str = './resultados.json'
    batch_size: int = 16
    label_columns: list = field(
        default_factory=lambda: [
            'Sexism',
            'Body',
            'Racism',
            'Ideology',
            'Homophobia'
        ]
    )
    min_characters: int = 5
    pred_path: str = ''

@dataclass
class TrainerConfig:
    load_data_config: LoadDataConfig = field(
        default_factory=lambda: LoadDataConfig()
    )
    save_path: str = './artifacts'
    save_weights_interval: int = 10
    device: str = "cuda"
    epochs: int = 10
    use_pretrained_classifier: bool = True
    model_name: str = 'pysentimiento/bertabaporu-pt-hate-speech'
    num_labels: int = 5
    classes: list = field(
        default_factory=lambda: [
            'Sexism',
            'Body',
            'Racism',
            'Ideology',
            'Homophobia'
        ]
    )
    start_lr: float = 0.00001
    ref_lr: float = 0.0001
    warm_up_lr: float = 2
    supervised_pid: str = None
    train: bool = True
    pred: bool = False


def parse_eval_trainer_args():
    parser = argparse.ArgumentParser(description="Configure Trainer parameters.")
    parser.add_argument(
        "--start_lr",
        type=float,
        required=False,
        help="Start learning rate for the cosine annealing schedule.",
        default=0.000001,
    )
    parser.add_argument(
        "--ref_lr",
        type=float,
        required=False,
        help="Reference learning rate for the cosine annealing schedule.",
        default=0.00001,
    )
    parser.add_argument(
        "--warm_up_lr",
        type=float,
        default=2,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='pysentimiento/bertabaporu-pt-hate-speech',
    )  
    parser.add_argument(
        "--supervised_pid",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        required=False,
        help="",
    ) 
    parser.add_argument(
        "--pred",
        action="store_true",
        required=False,
        help="",
    ) 
    parser.add_argument(
        "--use_pretrained_classifier",
        action="store_true",
        required=False,
        help="",
    )   
    args = parser.parse_args()

    loader_config = LoadDataConfig()

    trainer_config =  TrainerConfig(       
                                load_data_config=loader_config,
                                start_lr=args.start_lr,
                                ref_lr=args.ref_lr,
                                warm_up_lr=args.warm_up_lr,
                                supervised_pid=args.supervised_pid,
                                model_name=args.model_name,
                                train=args.train,
                                pred=args.pred,
                                use_pretrained_classifier=True
    )
    return trainer_config

# ---- CARREGAMENTO E PRÉ-PROCESSAMENTO DE DADOS ----

class LoadData:
    def __init__(self, json_path, pred_path, batch_size, label_columns, min_characters):
        self.json_path = json_path
        self.pred_path = pred_path
        self.batch_size = batch_size
        self.label_columns = label_columns
        self.min_characters = min_characters

    def preprocess_text(self, df):   
        def clean_message(msg):
            msg = re.sub(r'<[^>]+>', 'user_mention', msg)
            msg = preprocess_tweet(msg, lang='pt')
            msg = re.sub(r'[@#$*]', '', msg)
            return msg

        df['message'] = df['message'].apply(clean_message)
        df = df[df['message'].str.len() >= self.min_characters]
        return df

    def load_and_split_json(self, prediction):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.json_normalize(data)
        df['message'] = df['message'].astype(str)
        df = self.preprocess_text(df)
        df = df.dropna()
        df = df.sample(frac=1).reset_index(drop=True)  # Resetando o índice após o shuffle

        if prediction:
            return None, None, df

        # Inicializa as colunas de labels como 0
        for col in self.label_columns:
            df[col] = 0

        # Preenche as colunas de labels com 1 para as categorias correspondentes
        def set_labels(grupos, index):
            labels = {'R': 'Racism', 'H': 'Homophobia', 'G': 'Body', 'I': 'Ideology', 'M': 'Sexism'}
            for g in grupos:
                if g in labels:
                    df.at[index, labels[g]] = 1
        
        for index, row in df.iterrows():
            set_labels(row['grupos'], index)

        train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42)

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        return train_df, val_df, test_df



    def create_dataloaders(self, tokenizer, prediction=False):
        self.train_df, self.val_df, self.test_df = self.load_and_split_json(prediction)

        if prediction:
            test_set = CustomDataset(self.test_df, self.label_columns, tokenizer)
            test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
            return None, None, test_loader

        train_set = CustomDataset(self.train_df, self.label_columns, tokenizer)
        val_set = CustomDataset(self.val_df, self.label_columns, tokenizer)
        test_set = CustomDataset(self.test_df, self.label_columns, tokenizer)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


    def get_tokenizer(self):
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained(self.config['model_name'])

class CustomDataset(Dataset):
    def __init__(self, dataframe, label_columns, tokenizer, max_length=384):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_columns = label_columns
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dataframe = self.dataframe[self.dataframe.apply(self.filter_messages, axis=1)].reset_index(drop=True)

    def filter_messages(self, row):
        message = row['message']
        context_messages = [msg['message'] for msg in row['contexto']][-10:]
        context = ' '.join(context_messages)
        combined_message = f"{context} {message}"

        tokens = self.tokenizer(
            combined_message,
            truncation=False,
            padding=False,
            max_length=None,
            return_tensors='pt'
        )

        return tokens['input_ids'].size(1) <= self.max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        message = self.dataframe.loc[index, 'message']
        context_messages = [msg['message'] for msg in self.dataframe.loc[index, 'contexto']][-10:]
        context = ' '.join(context_messages)
        combined_message = f"{context} {message}"

        combined_tokens = self.tokenizer(
            combined_message,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Apenas a mensagem em destaque
        message_tokens = self.tokenizer(
            message,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        labels = self.dataframe.loc[index, self.label_columns].values.astype(float)

        return {
            'input_ids': combined_tokens['input_ids'].squeeze(0),
            'attention_mask': combined_tokens['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float),
            'message_input_ids': message_tokens['input_ids'].squeeze(0)
        }

from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F

class Trainer:
    def __init__(self, **config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.tokenizer = self.get_tokenizer()

        self.load_data = LoadData(
            json_path=config['load_data_config'].json_path,
            pred_path=config['load_data_config'].pred_path,
            batch_size=config['load_data_config'].batch_size,
            label_columns=config['load_data_config'].label_columns,
            min_characters=config['load_data_config'].min_characters
        )

        self.train_loader, self.val_loader, self.test_loader = self.load_data.create_dataloaders(self.tokenizer)

        # Usando o modelo BertForSequenceClassification do pysentimiento
        self.model = BertForSequenceClassification.from_pretrained(
            config['model_name'], 
            num_labels=config['num_labels'],
            output_hidden_states=True  # Para obter os hidden states (embeddings)
        ).to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=config['start_lr'], weight_decay=0.01)
        total_steps = len(self.train_loader) * config['epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=int(0.005 * total_steps), 
            num_training_steps=total_steps
        )

        self.train_loss = []
        self.val_loss = []

    def get_tokenizer(self):
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained(self.config['model_name'])

    def extract_message_embeddings(self, hidden_states, message_input_ids):
        # Extraímos a última camada de hidden states
        last_hidden_state = hidden_states[-1]
        # Extraímos os embeddings correspondentes à mensagem em destaque
        message_embeddings = last_hidden_state[:, :message_input_ids.size(1), :]
        # Podemos usar a média dos embeddings da mensagem ou outro método
        pooled_output = torch.mean(message_embeddings, dim=1)
        return pooled_output

    def train(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            message_input_ids = batch['message_input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_mask)
            hidden_states = outputs.hidden_states
            
            # Extrai os embeddings da última camada para a mensagem em destaque
            filtered_embeddings = self.extract_message_embeddings(hidden_states, message_input_ids)
            
            # Passa os embeddings filtrados pela camada de classificação
            logits = self.model.classifier(filtered_embeddings)

            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        avg_loss = total_loss / len(self.train_loader)
        self.train_loss.append(avg_loss)
        return avg_loss

    from sklearn.metrics import f1_score

# Alterando a função `validate` e `test` para incluir zero_division=1
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                message_input_ids = batch['message_input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                hidden_states = outputs.hidden_states

                # Extrai os embeddings da última camada para a mensagem em destaque
                filtered_embeddings = self.extract_message_embeddings(hidden_states, message_input_ids)

                # Passa os embeddings filtrados pela camada de classificação
                logits = self.model.classifier(filtered_embeddings)

                loss = F.binary_cross_entropy_with_logits(logits, labels)
                total_loss += loss.item()

                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        val_f1 = f1_score(all_labels, (all_preds >= 0.5).astype(int), average='weighted', zero_division=1)
        print(f"Validation F1-Score: {val_f1:.4f}")
        return avg_loss, all_preds, all_labels

    # Mesma alteração na função `test`
    def test(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                message_input_ids = batch['message_input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                hidden_states = outputs.hidden_states

                # Extrai os embeddings da última camada para a mensagem em destaque
                filtered_embeddings = self.extract_message_embeddings(hidden_states, message_input_ids)

                # Passa os embeddings filtrados pela camada de classificação
                logits = self.model.classifier(filtered_embeddings)

                loss = F.binary_cross_entropy_with_logits(logits, labels)
                total_loss += loss.item()

                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        avg_loss = total_loss / len(self.test_loader)
        test_f1 = f1_score(all_labels, (all_preds >= 0.5).astype(int), average='weighted', zero_division=1)
        print(f"Test F1-Score: {test_f1:.4f}")
        return avg_loss, all_preds, all_labels

    
    def run(self):
        best_val_loss = float('inf')

        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            train_loss = self.train()
            val_loss, _, _ = self.validate()

            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch)

            print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        test_loss, all_preds, all_labels = self.test()
        print(f"Test Loss: {test_loss:.4f}")

        # Calcular métricas no conjunto de teste por categoria
        threshold = 0.5
        all_preds_bin = (all_preds >= threshold).astype(int)
        
        metrics = {'Category': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 'AUC': []}
        
        for i, category in enumerate(self.config['classes']):
            metrics['Category'].append(category)
            metrics['Accuracy'].append(accuracy_score(all_labels[:, i], all_preds_bin[:, i]))
            metrics['Precision'].append(precision_score(all_labels[:, i], all_preds_bin[:, i]))
            metrics['Recall'].append(recall_score(all_labels[:, i], all_preds_bin[:, i]))
            metrics['F1-Score'].append(f1_score(all_labels[:, i], all_preds_bin[:, i]))
            metrics['AUC'].append(roc_auc_score(all_labels[:, i], all_preds[:, i]))

        # Criar e imprimir a tabela
        df_metrics = pd.DataFrame(metrics)
        print("\nMetrics by Category:")
        print(df_metrics.to_string(index=False))
        
        return df_metrics  # Retorna as métricas para posterior agregação


    def save_model(self, epoch):
        save_dir = "./artifacts"
        os.makedirs(save_dir, exist_ok=True)  # Cria o diretório se ele não existir
        
        save_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pt")
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

def create_pid_variable():
    if os.getenv('SLURM_JOB_ID') is None:
        now = datetime.now()
        formatted_time = now.strftime('%d%m%H%M')
        os.environ['SLURM_JOB_ID'] = formatted_time
        print(f'SLURM_JOB_ID has been set to: {formatted_time}')
    else:
        print(f'SLURM_JOB_ID is already set to: {os.getenv("SLURM_JOB_ID")}')

if __name__ == "__main__":
    create_pid_variable()
    config = parse_eval_trainer_args()
    
    all_metrics = []
    for run_idx in range(3):
        print(f"\n\n--- Run {run_idx + 1}/5 ---")
        trainer = Trainer(**config.__dict__)
        metrics = trainer.run()
        all_metrics.append(metrics)
    
    # Agrega os resultados de todas as execuções
    aggregated_metrics = pd.concat(all_metrics).groupby('Category').mean()

    # Calcula a média do AUC e a média da variância do AUC
    auc_values = [metrics['AUC'].values for metrics in all_metrics]
    auc_means = np.mean(auc_values, axis=0)
    auc_variances = np.var(auc_values, axis=0)

    mean_auc = np.mean(auc_means)
    mean_variance_auc = np.mean(auc_variances)
    
    print("\n--- Aggregated Metrics After 5 Runs ---")
    print(aggregated_metrics.to_string(index=True))
    
    print(f"\nAverage AUC across 5 categories: {mean_auc:.4f}")
    print(f"Average variance of AUC across 5 categories: {mean_variance_auc:.4f}")
