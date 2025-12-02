import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import obonet
from Bio import SeqIO
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel
from sklearn.model_selection import train_test_split

# --- Configuration ---
SEED = 42
MAX_LEN = 512
BATCH_SIZE = 2
EPOCHS = 10
LR = 5e-5
NUM_CLASSES = 1500
MODEL_CHECKPOINT = "facebook/esm2_t33_650M_UR50D"

# Paths (Adjust as needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "cafa-6-protein-function-prediction/data")
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TEST_DIR = os.path.join(DATA_DIR, "Test")
GO_OBO_PATH = os.path.join(TRAIN_DIR, "go-basic.obo")
TRAIN_FASTA_PATH = os.path.join(TRAIN_DIR, "train_sequences.fasta")
TRAIN_TERMS_PATH = os.path.join(TRAIN_DIR, "train_terms.tsv")
TEST_FASTA_PATH = os.path.join(TEST_DIR, "testsuperset.fasta")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ProteinDataset(Dataset):
    def __init__(self, df_seq, df_terms, term_to_idx, tokenizer, max_len=MAX_LEN):
        self.df_seq = df_seq
        self.df_terms = df_terms
        self.term_to_idx = term_to_idx
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.grouped_terms = df_terms.groupby('EntryID')['term'].apply(set).to_dict()

    def __len__(self):
        return len(self.df_seq)

    def __getitem__(self, idx):
        row = self.df_seq.iloc[idx]
        protein_id = row['EntryID']
        seq = row['Sequence']
        
        encoding = self.tokenizer(
            seq,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        label = np.zeros(len(self.term_to_idx), dtype=np.float32)
        if protein_id in self.grouped_terms:
            terms = self.grouped_terms[protein_id]
            for term in terms:
                if term in self.term_to_idx:
                    label[self.term_to_idx[term]] = 1.0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.float32)
        }

class ESM2ForProteinFunction(nn.Module):
    def __init__(self, model_checkpoint, num_classes):
        super(ESM2ForProteinFunction, self).__init__()
        self.esm = EsmModel.from_pretrained(model_checkpoint)
        self.fc = nn.Linear(self.esm.config.hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(cls_representation)
        return self.sigmoid(logits)

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Ontology
    print("Loading Ontology...")
    if not os.path.exists(GO_OBO_PATH):
        print(f"Error: {GO_OBO_PATH} not found.")
        return
    go_graph = obonet.read_obo(GO_OBO_PATH)

    # 2. Load Data
    print("Loading Data...")
    sequences = []
    ids = []
    for record in SeqIO.parse(TRAIN_FASTA_PATH, "fasta"):
        sequences.append(str(record.seq))
        ids.append(record.id.split("|")[1])
    df_seq = pd.DataFrame({"EntryID": ids, "Sequence": sequences})
    
    df_terms = pd.read_csv(TRAIN_TERMS_PATH, sep="\t")
    
    # 3. Prepare Labels
    term_counts = df_terms['term'].value_counts()
    top_terms = term_counts.head(NUM_CLASSES).index.tolist()
    term_to_idx = {term: i for i, term in enumerate(top_terms)}
    
    # 4. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # 5. Datasets
    train_df, val_df = train_test_split(df_seq, test_size=0.2, random_state=SEED)
    train_dataset = ProteinDataset(train_df, df_terms, term_to_idx, tokenizer)
    val_dataset = ProteinDataset(val_df, df_terms, term_to_idx, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 6. Model
    model = ESM2ForProteinFunction(MODEL_CHECKPOINT, NUM_CLASSES)
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 7. Training
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), "protein_esm2_script.pth")
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    main()
