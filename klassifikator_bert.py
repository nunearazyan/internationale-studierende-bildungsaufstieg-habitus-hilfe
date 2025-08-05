from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.nn import CrossEntropyLoss
import accelerate
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

# CSV-Datei einlesen
df = pd.read_csv("umfrage_featured.csv")

# Aufbereitung des Datensatzes
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
# Stärkeres Training bei unterrepräsentierten Klasses durch ihre höhere Gewichtung
class WeightedTrainer(Trainer):
    def __init__(self, *args, weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = CrossEntropyLoss(weight=self.weights.to(model.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

'''Klassen'''
# Vereinfachung von Mehrfachzugehörigkeiten und zu wenig repräsentierten Klassen
# Teilung auf Nicht-EU-Student*innen und Deutsche + EU in einer Klasse
def simplify_nation_category(value):
    parts = [part.strip() for part in value.split(",")]
    if any("Nicht-EU / Staatenlos" in part for part in parts):
        return "Nicht-EU / Staatenlos"
    else:
        return "Deutsch / EU"
    
df["tn_staang_simplified"] = df["tn_staang"].apply(simplify_nation_category)

# Teilung auf Bildungsausländer*innen und Bildungsinländer*innen, AT
def simplify_residence_permit_category(value):
    if value in [
        "16b AufenthG",
        "25 AufenthG",
        "24 AufenthG"
    ]:
        return "Bildungsausländer*innen, AT"
    else:
        return "Bildungsinländer*innen"
    
df["tn_aufenth_simplified"] = df["tn_aufenth"].apply(simplify_residence_permit_category)

# Teilung auf Hochschule und Universität
def simplify_univ_category(value):
    parts = [part.strip() for part in value.split(",")]
    if any("Hochschule" in part for part in parts):
        return "Hochschule"
    else:
        return "Universität"
    
df["tn_hochschulart_simplified"] = df["tn_hochschulart"].apply(simplify_univ_category)

# Dimensionalitätsreduktion bei Fächerkategorien
def simplify_fach_category(value):
    parts = [part.strip() for part in value.split(",")]
    if any("Erziehungswissenschaften / Sozialwesen / Bildungswissenschaften / Soziale Arbeit" in part for part in parts):
        return "Erziehungswissenschaften / Sozialwesen / Bildungswissenschaften / Soziale Arbeit"
    elif any(part in [
        "Ingenieurwissenschaften (ohne Maschinenbau und Verfahrens- / Elektro- / Informationstechnik)",
        "Maschinenbau und Verfahrens- / Elektro- / Informationstechnik",
        "Mathematik / Informatik und verwandte Fächer wie Wirtschaftsinformatik oder Data Science"
    ] for part in parts):
        return "Mathematik / Informatik und Ingenieurwissenschaften"
    elif any(part in [
        "Wirtschaftswissenschaften",
        "Sozial- / Gesellschafts- / Politik- / Regional- / Verwaltungswissenschaften"
    ] for part in parts):
        return "Sozial- und Wirtschaftswissenschaften"
    elif any(part in [
        "Geisteswissenschaften (ohne Sprachwissenschaften)",
        "Sprachwissenschaften",
        "Kunst (z. B. bildende Kunst / darstellende Kunst / Musik / Regie) / Kunstwissenschaften"
    ] for part in parts):
        return "Geisteswissenschaften"
    elif any(part in [
        "Medizin (Human- und Zahnmedizin) / Gesundheitswissenschaften",
        "Psychologie",
        "Naturwissenschaften",
        "Agrar-/ Forst- / Ernährungswissenschaften"
    ] for part in parts):
        return "Medizin, Gesundheits- und Naturwissenschaften und Psychologie"
    elif any("Lehramt" in part for part in parts):
        return "Lehramt"
    else:
        return "Sonstiges"
    
df["tn_fach_simplified"] = df["tn_fach"].apply(simplify_fach_category)
#print(df["tn_fach_simplified"].value_counts())

# Klassen
label_columns = [
    #"tn_staang",
    "tn_staang_simplified", # Vereinfacht
    "tn_migr",
    #"tn_aufenth",
    "tn_aufenth_simplified", # Vereinfacht
    "tn_dt_spr",
    "tn_hsabsch_eltern",
    "tn_fin_sit",
    #"tn_hochschulart",
    "tn_hochschulart_simplified", # Vereinfacht
    "tn_fach",
    "tn_fach_simplified" # Vereinfacht
]

text_columns = [
    "uebergang_unterschiede",
    "anpassung",
    "interaktionen_unterschiede",
    "gesp_hab_extern",
    "gesp_hab_intern",
    "alles_lemmatisiert"
]

# BERT für Deutsch
model_name = "bert-base-german-cased"
max_length = 512 # da die Daten längere Texte darstellen
epochs = 7 # zum besseren Training auf unterräpresentierten Klassen
batch_size = 4

for text_col in text_columns:
    for label_col in label_columns:
        print(f"\nModell für Textabscnitt: '{text_col}' → Klassendimension: '{label_col}'")

        # Löschen fehlender Werte
        sub_df = df.dropna(subset=[text_col, label_col]).copy()

        sub_df["text"] = sub_df[text_col].astype(str)
        sub_df["label_text"] = sub_df[label_col].astype(str)

        # Label-Encoding (aber Beibehaltung der Textlabels mit Mapping)
        label2id = {label: i for i, label in enumerate(sub_df["label_text"].unique())}
        id2label = {i: label for label, i in label2id.items()}
        sub_df["label"] = sub_df["label_text"].map(label2id)

        # Split
        train_df, test_df = train_test_split(sub_df, test_size=0.2, stratify=sub_df["label"], random_state=42)
        train_labels = train_df["label"].tolist()
        
        # Gewichtungen der Klassen
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels
        )
        weights = torch.tensor(class_weights, dtype=torch.float)

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels
        )
        weights = torch.tensor(class_weights, dtype=torch.float)

        # Tokenizer & Modell
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )

        def tokenize(example):
            return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)
        
        # Traun und Test
        train_dataset = TextDataset(
            texts=train_df["text"].tolist(),
            labels=train_df["label"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length
        )

        test_dataset = TextDataset(
            texts=test_df["text"].tolist(),
            labels=test_df["label"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length
        )

        # Training
        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            weights=weights
        )

        trainer.train()

        # Evaluation
        predictions = trainer.predict(test_dataset)
        y_pred_ids = predictions.predictions.argmax(axis=1)
        y_true_ids = predictions.label_ids

        y_pred_labels = [id2label[i] for i in y_pred_ids]
        y_true_labels = [id2label[i] for i in y_true_ids]

        print(f"\n Classification Report für Text='{text_col}' → Label='{label_col}':")
        print(classification_report(y_true_labels, y_pred_labels))

        ''' Confusion-Matrix '''
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=list(label2id.keys()))

        wrapped_labels = [textwrap.fill(label, width=20) for label in label2id.keys()]

        cmap = sns.diverging_palette(80, 267, l=50, center="light", as_cmap=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=wrapped_labels,
                    yticklabels=wrapped_labels)

        plt.xlabel("Vorhergesagte Klasse")
        plt.ylabel("Tatsächliche Klasse")
        plt.title(f"Confusion Matrix\nText='{text_col}' → Label='{label_col}'")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()