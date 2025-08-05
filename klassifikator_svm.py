import pandas as pd
import ast
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

# CSV-Datei einlesen
df = pd.read_csv("umfrage_featured.csv")

'''Features'''
# LDA
lda_columns = [
    "uebergang_unterschiede_lda_topics", # wenig aussagekräftig
    "anpassung_lda_topics",
    "interaktionen_unterschiede_lda_topics", # wenig aussagekräftig
    "gesp_hab_extern_lda_topics", # wenig aussagekräftig
    "gesp_hab_intern_lda_topics", # wenig aussagekräftig
    "alles_lemmatisiert_lda_topics"
] # LDA kombinierte Spalte
lda_feature_dfs = []

for col in lda_columns:
    lda_dicts = df[col].apply(ast.literal_eval)
    lda_df = pd.DataFrame(lda_dicts.tolist()).fillna(0)
    lda_df.columns = [f"{col}_topic{i+1}" for i in range(lda_df.shape[1])]
    lda_feature_dfs.append(lda_df)

X_lda_combined = pd.concat(lda_feature_dfs, axis=1)

# BERTopic
bertopic_columns = [
    "uebergang_unterschiede_bertopic_topics",
    "anpassung_bertopic_topics",
    "interaktionen_unterschiede_bertopic_topics",
    "gesp_hab_extern_bertopic_topics",
    "gesp_hab_intern_bertopic_topics",
    "alles_lemmatisiert_bertopic_topics"
]
bertopic_feature_dfs = []

for col in bertopic_columns:
    bertopic_dicts = df[col].apply(ast.literal_eval)
    bertopic_df = pd.DataFrame(bertopic_dicts.tolist()).fillna(0)
    bertopic_df.columns = [f"{col}_topic{i+1}" for i in range(bertopic_df.shape[1])]
    bertopic_feature_dfs.append(bertopic_df)

X_bertopic_combined = pd.concat(bertopic_feature_dfs, axis=1)

# EmoLex
emolex_columns = [
    "emolex_emotions"
]
emolex_feature_dfs = []

for col in emolex_columns:
    emolex_dicts = df[col].apply(ast.literal_eval)
    emolex_df = pd.DataFrame(emolex_dicts.tolist()).fillna(0)
    emolex_df.columns = [f"{col}_topic{i+1}" for i in range(emolex_df.shape[1])]
    emolex_feature_dfs.append(emolex_df)

X_emolex_combined = pd.concat(emolex_feature_dfs, axis=1)

# germansentiment
germansentiment_columns = [
    "germansentiment_emotions"
]
germansentiment_feature_dfs = []

for col in germansentiment_columns:
    germansentiment_dicts = df[col].apply(ast.literal_eval)
    germansentiment_df = pd.DataFrame(germansentiment_dicts.tolist()).fillna(0)
    germansentiment_df.columns = [f"{col}_topic{i+1}" for i in range(germansentiment_df.shape[1])]
    germansentiment_feature_dfs.append(germansentiment_df)

X_germansentiment_combined = pd.concat(germansentiment_feature_dfs, axis=1)

# N-Gramme
ngram_count_features = df.filter(regex="count_1gram|count_2gram|count_3gram|count_4gram") # Frequenzen
ngram_tfidf_features = df.filter(regex="tfidf_1gram|tfidf_2gram|tfidf_3gram|tfidf_4gram") # TF-IDF

# Skalierte Frequenzen der Wörter mit dem höchsten Informationsgewinn
ig_features = df.filter(regex="TopIG")

# Annonationsbasierte Fratures
# Frequenz jedes Annotationstags pro Antwort
xml_freq_all_features = df.filter(regex="FREQ")
# TF-IDF jedes Annotationstags pro Antwort
xml_tfidf_all_features = df.filter(regex="TFIDF")
# Frequenzen im Tag <nicht_passung>
xml_freq_nicht_passung_features = df.filter(regex=r'^FREQ<nicht_passung.*>$')
# TF-IDF im Tag <nicht_passung>
xml_tfidf_nicht_passung_features = df.filter(regex=r'^TFIDF<nicht_passung.*>$')
# Frequenzen im Tag <sek_habitus>
xml_freq_sek_habitus_features = df.filter(regex=r'^FREQ<sek_habitus.*>$')
# IF-IDF im Tag <sek_habitus>
xml_tfidf_sek_habitus_features = df.filter(regex=r'^TFIDF<sek_habitus.*>$')

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
y_nation = df['tn_staang_simplified'] # Vereinfacht
y_migration = df['tn_migr']
y_residence_perm = df['tn_aufenth_simplified'] # Vereinfacht
y_german_lang = df['tn_dt_spr']
y_edu_parents = df['tn_hsabsch_eltern']
y_finance_status = df['tn_fin_sit']
y_univ = df['tn_hochschulart_simplified'] # Vereinfacht
y_subj = df['tn_fach_simplified'] # Vereinfacht

''' Trainings- und Testdaten '''
# LDA
#X_train, X_test, y_train, y_test = train_test_split(X_lda_combined, y_nation, test_size=0.2, stratify=y_nation, random_state=42) # Staatsangehörigkeit
#X_train, X_test, y_train, y_test = train_test_split(X_lda_combined, y_migration, test_size=0.2, stratify=y_migration, random_state=42) # Migrationshintergrund
#X_train, X_test, y_train, y_test = train_test_split(X_lda_combined, y_residence_perm, test_size=0.2, stratify=y_residence_perm, random_state=42) # Bildungsinländer*in oder Bildungsausländer*in, AT
#X_train, X_test, y_train, y_test = train_test_split(X_lda_combined, y_german_lang, test_size=0.2, stratify=y_german_lang, random_state=42) # Erwerb der deutschen Sprache
#X_train, X_test, y_train, y_test = train_test_split(X_lda_combined, y_edu_parents, test_size=0.2, stratify=y_edu_parents, random_state=42) # Bildungsstatus der Eltern
#X_train, X_test, y_train, y_test = train_test_split(X_lda_combined, y_finance_status, test_size=0.2, stratify=y_finance_status, random_state=42) # Finanzielle Situation in der Familie
#X_train, X_test, y_train, y_test = train_test_split(X_lda_combined, y_univ, test_size=0.2, stratify=y_univ, random_state=42) # Hochschultyp
#X_train, X_test, y_train, y_test = train_test_split(X_lda_combined, y_subj, test_size=0.2, stratify=y_subj, random_state=42) # Fach

# BERTopic
#X_train, X_test, y_train, y_test = train_test_split(X_bertopic_combined, y_nation, test_size=0.2, stratify=y_nation, random_state=42) # Staatsangehörigkeit
#X_train, X_test, y_train, y_test = train_test_split(X_bertopic_combined, y_migration, test_size=0.2, stratify=y_migration, random_state=42) # Migrationshintergrund
#X_train, X_test, y_train, y_test = train_test_split(X_bertopic_combined, y_residence_perm, test_size=0.2, stratify=y_residence_perm, random_state=42) # Bildungsinländer*in oder Bildungsausländer*in, AT
#X_train, X_test, y_train, y_test = train_test_split(X_bertopic_combined, y_german_lang, test_size=0.2, stratify=y_german_lang, random_state=42) # Erwerb der deutschen Sprache
#X_train, X_test, y_train, y_test = train_test_split(X_bertopic_combined, y_edu_parents, test_size=0.2, stratify=y_edu_parents, random_state=42) # Bildungsstatus der Eltern
#X_train, X_test, y_train, y_test = train_test_split(X_bertopic_combined, y_finance_status, test_size=0.2, stratify=y_finance_status, random_state=42) # Finanzielle Situation in der Familie
#X_train, X_test, y_train, y_test = train_test_split(X_bertopic_combined, y_univ, test_size=0.2, stratify=y_univ, random_state=42) # Hochschultyp
#X_train, X_test, y_train, y_test = train_test_split(X_bertopic_combined, y_subj, test_size=0.2, stratify=y_subj, random_state=42) # Fach

# EmoLex
#X_train, X_test, y_train, y_test = train_test_split(X_emolex_combined, y_nation, test_size=0.2, stratify=y_nation, random_state=42) # Staatsangehörigkeit
#X_train, X_test, y_train, y_test = train_test_split(X_emolex_combined, y_migration, test_size=0.2, stratify=y_migration, random_state=42) # Migrationshintergrund
#X_train, X_test, y_train, y_test = train_test_split(X_emolex_combined, y_residence_perm, test_size=0.2, stratify=y_residence_perm, random_state=42) # Bildungsinländer*in oder Bildungsausländer*in, AT
#X_train, X_test, y_train, y_test = train_test_split(X_emolex_combined, y_german_lang, test_size=0.2, stratify=y_german_lang, random_state=42) # Erwerb der deutschen Sprache
#X_train, X_test, y_train, y_test = train_test_split(X_emolex_combined, y_edu_parents, test_size=0.2, stratify=y_edu_parents, random_state=42) # Bildungsstatus der Eltern
#X_train, X_test, y_train, y_test = train_test_split(X_emolex_combined, y_finance_status, test_size=0.2, stratify=y_finance_status, random_state=42) # Finanzielle Situation in der Familie
#X_train, X_test, y_train, y_test = train_test_split(X_emolex_combined, y_univ, test_size=0.2, stratify=y_univ, random_state=42) # Hochschultyp
#X_train, X_test, y_train, y_test = train_test_split(X_emolex_combined, y_subj, test_size=0.2, stratify=y_subj, random_state=42) # Fach

# germansentiment
#X_train, X_test, y_train, y_test = train_test_split(X_germansentiment_combined, y_nation, test_size=0.2, stratify=y_nation, random_state=42) # Staatsangehörigkeit
#X_train, X_test, y_train, y_test = train_test_split(X_germansentiment_combined, y_migration, test_size=0.2, stratify=y_migration, random_state=42) # Migrationshintergrund
#X_train, X_test, y_train, y_test = train_test_split(X_germansentiment_combined, y_residence_perm, test_size=0.2, stratify=y_residence_perm, random_state=42) # Bildungsinländer*in oder Bildungsausländer*in, AT
#X_train, X_test, y_train, y_test = train_test_split(X_germansentiment_combined, y_german_lang, test_size=0.2, stratify=y_german_lang, random_state=42) # Erwerb der deutschen Sprache
#X_train, X_test, y_train, y_test = train_test_split(X_germansentiment_combined, y_edu_parents, test_size=0.2, stratify=y_edu_parents, random_state=42) # Bildungsstatus der Eltern
#X_train, X_test, y_train, y_test = train_test_split(X_germansentiment_combined, y_finance_status, test_size=0.2, stratify=y_finance_status, random_state=42) # Finanzielle Situation in der Familie
#X_train, X_test, y_train, y_test = train_test_split(X_germansentiment_combined, y_univ, test_size=0.2, stratify=y_univ, random_state=42) # Hochschultyp
#X_train, X_test, y_train, y_test = train_test_split(X_germansentiment_combined, y_subj, test_size=0.2, stratify=y_subj, random_state=42) # Fach

# N-Gramme. Frequenzen
#X_train, X_test, y_train, y_test = train_test_split(ngram_count_features, y_nation, test_size=0.2, stratify=y_nation, random_state=42) # Staatsangehörigkeit
#X_train, X_test, y_train, y_test = train_test_split(ngram_count_features, y_migration, test_size=0.2, stratify=y_migration, random_state=42) # Migrationshintergrund
#X_train, X_test, y_train, y_test = train_test_split(ngram_count_features, y_residence_perm, test_size=0.2, stratify=y_residence_perm, random_state=42) # Bildungsinländer*in oder Bildungsausländer*in, AT
#X_train, X_test, y_train, y_test = train_test_split(ngram_count_features, y_german_lang, test_size=0.2, stratify=y_german_lang, random_state=42) # Erwerb der deutschen Sprache
#X_train, X_test, y_train, y_test = train_test_split(ngram_count_features, y_edu_parents, test_size=0.2, stratify=y_edu_parents, random_state=42) # Bildungsstatus der Eltern
#X_train, X_test, y_train, y_test = train_test_split(ngram_count_features, y_finance_status, test_size=0.2, stratify=y_finance_status, random_state=42) # Finanzielle Situation in der Familie
#X_train, X_test, y_train, y_test = train_test_split(ngram_count_features, y_univ, test_size=0.2, stratify=y_univ, random_state=42) # Hochschultyp
#X_train, X_test, y_train, y_test = train_test_split(ngram_count_features, y_subj, test_size=0.2, stratify=y_subj, random_state=42) # Fach

# N-Gramme. TF-IDF
#X_train, X_test, y_train, y_test = train_test_split(ngram_tfidf_features, y_nation, test_size=0.2, stratify=y_nation, random_state=42) # Staatsangehörigkeit
#X_train, X_test, y_train, y_test = train_test_split(ngram_tfidf_features, y_migration, test_size=0.2, stratify=y_migration, random_state=42) # Migrationshintergrund
#X_train, X_test, y_train, y_test = train_test_split(ngram_tfidf_features, y_residence_perm, test_size=0.2, stratify=y_residence_perm, random_state=42) # Bildungsinländer*in oder Bildungsausländer*in, AT
#X_train, X_test, y_train, y_test = train_test_split(ngram_tfidf_features, y_german_lang, test_size=0.2, stratify=y_german_lang, random_state=42) # Erwerb der deutschen Sprache
#X_train, X_test, y_train, y_test = train_test_split(ngram_tfidf_features, y_edu_parents, test_size=0.2, stratify=y_edu_parents, random_state=42) # Bildungsstatus der Eltern
#X_train, X_test, y_train, y_test = train_test_split(ngram_tfidf_features, y_finance_status, test_size=0.2, stratify=y_finance_status, random_state=42) # Finanzielle Situation in der Familie
#X_train, X_test, y_train, y_test = train_test_split(ngram_tfidf_features, y_univ, test_size=0.2, stratify=y_univ, random_state=42) # Hochschultyp
#X_train, X_test, y_train, y_test = train_test_split(ngram_tfidf_features, y_subj, test_size=0.2, stratify=y_subj, random_state=42) # Fach

# Top-Wörter mit dem höchsten IG
#X_train, X_test, y_train, y_test = train_test_split(ig_features, y_nation, test_size=0.2, stratify=y_nation, random_state=42) # Staatsangehörigkeit
#X_train, X_test, y_train, y_test = train_test_split(ig_features, y_migration, test_size=0.2, stratify=y_migration, random_state=42) # Migrationshintergrund
#X_train, X_test, y_train, y_test = train_test_split(ig_features, y_residence_perm, test_size=0.2, stratify=y_residence_perm, random_state=42) # Bildungsinländer*in oder Bildungsausländer*in, AT
#X_train, X_test, y_train, y_test = train_test_split(ig_features, y_german_lang, test_size=0.2, stratify=y_german_lang, random_state=42) # Erwerb der deutschen Sprache
#X_train, X_test, y_train, y_test = train_test_split(ig_features, y_edu_parents, test_size=0.2, stratify=y_edu_parents, random_state=42) # Bildungsstatus der Eltern
#X_train, X_test, y_train, y_test = train_test_split(ig_features, y_finance_status, test_size=0.2, stratify=y_finance_status, random_state=42) # Finanzielle Situation in der Familie
#X_train, X_test, y_train, y_test = train_test_split(ig_features, y_univ, test_size=0.2, stratify=y_univ, random_state=42) # Hochschultyp
#X_train, X_test, y_train, y_test = train_test_split(ig_features, y_subj, test_size=0.2, stratify=y_subj, random_state=42) # Fach

# Frequenz jedes Annotationstags pro Antwort
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_all_features, y_nation, test_size=0.2, stratify=y_nation, random_state=42) # Staatsangehörigkeit
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_all_features, y_migration, test_size=0.2, stratify=y_migration, random_state=42) # Migrationshintergrund
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_all_features, y_residence_perm, test_size=0.2, stratify=y_residence_perm, random_state=42) # Bildungsinländer*in oder Bildungsausländer*in, AT
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_all_features, y_german_lang, test_size=0.2, stratify=y_german_lang, random_state=42) # Erwerb der deutschen Sprache
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_all_features, y_edu_parents, test_size=0.2, stratify=y_edu_parents, random_state=42) # Bildungsstatus der Eltern
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_all_features, y_finance_status, test_size=0.2, stratify=y_finance_status, random_state=42) # Finanzielle Situation in der Familie
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_all_features, y_univ, test_size=0.2, stratify=y_univ, random_state=42) # Hochschultyp
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_all_features, y_subj, test_size=0.2, stratify=y_subj, random_state=42) # Fach

# TF-IDF jedes Annotationstags pro Antwort 
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_all_features, y_nation, test_size=0.2, stratify=y_nation, random_state=42) # Staatsangehörigkeit
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_all_features, y_migration, test_size=0.2, stratify=y_migration, random_state=42) # Migrationshintergrund
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_all_features, y_residence_perm, test_size=0.2, stratify=y_residence_perm, random_state=42) # Bildungsinländer*in oder Bildungsausländer*in, AT
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_all_features, y_german_lang, test_size=0.2, stratify=y_german_lang, random_state=42) # Erwerb der deutschen Sprache
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_all_features, y_edu_parents, test_size=0.2, stratify=y_edu_parents, random_state=42) # Bildungsstatus der Eltern
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_all_features, y_finance_status, test_size=0.2, stratify=y_finance_status, random_state=42) # Finanzielle Situation in der Familie
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_all_features, y_univ, test_size=0.2, stratify=y_univ, random_state=42) # Hochschultyp
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_all_features, y_subj, test_size=0.2, stratify=y_subj, random_state=42) # Fach

# Frequenzen im Tag <nicht_passung>  
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_nicht_passung_features, y_nation, test_size=0.2, stratify=y_nation, random_state=42) # Staatsangehörigkeit
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_nicht_passung_features, y_migration, test_size=0.2, stratify=y_migration, random_state=42) # Migrationshintergrund
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_nicht_passung_features, y_residence_perm, test_size=0.2, stratify=y_residence_perm, random_state=42) # Bildungsinländer*in oder Bildungsausländer*in, AT
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_nicht_passung_features, y_german_lang, test_size=0.2, stratify=y_german_lang, random_state=42) # Erwerb der deutschen Sprache
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_nicht_passung_features, y_edu_parents, test_size=0.2, stratify=y_edu_parents, random_state=42) # Bildungsstatus der Eltern
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_nicht_passung_features, y_finance_status, test_size=0.2, stratify=y_finance_status, random_state=42) # Finanzielle Situation in der Familie
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_nicht_passung_features, y_univ, test_size=0.2, stratify=y_univ, random_state=42) # Hochschultyp
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_nicht_passung_features, y_subj, test_size=0.2, stratify=y_subj, random_state=42) # Fach

# TF-IDF im Tag <nicht_passung> 
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_nicht_passung_features, y_nation, test_size=0.2, stratify=y_nation, random_state=42) # Staatsangehörigkeit
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_nicht_passung_features, y_migration, test_size=0.2, stratify=y_migration, random_state=42) # Migrationshintergrund
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_nicht_passung_features, y_residence_perm, test_size=0.2, stratify=y_residence_perm, random_state=42) # Bildungsinländer*in oder Bildungsausländer*in, AT
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_nicht_passung_features, y_german_lang, test_size=0.2, stratify=y_german_lang, random_state=42) # Erwerb der deutschen Sprache
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_nicht_passung_features, y_edu_parents, test_size=0.2, stratify=y_edu_parents, random_state=42) # Bildungsstatus der Eltern
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_nicht_passung_features, y_finance_status, test_size=0.2, stratify=y_finance_status, random_state=42) # Finanzielle Situation in der Familie
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_nicht_passung_features, y_univ, test_size=0.2, stratify=y_univ, random_state=42) # Hochschultyp
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_nicht_passung_features, y_subj, test_size=0.2, stratify=y_subj, random_state=42) # Fach 

# Frequenzen im Tag <sek_habitus>
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_sek_habitus_features, y_nation, test_size=0.2, stratify=y_nation, random_state=42) # Staatsangehörigkeit
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_sek_habitus_features, y_migration, test_size=0.2, stratify=y_migration, random_state=42) # Migrationshintergrund
X_train, X_test, y_train, y_test = train_test_split(xml_freq_sek_habitus_features, y_residence_perm, test_size=0.2, stratify=y_residence_perm, random_state=42) # Bildungsinländer*in oder Bildungsausländer*in, AT
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_sek_habitus_features, y_german_lang, test_size=0.2, stratify=y_german_lang, random_state=42) # Erwerb der deutschen Sprache
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_sek_habitus_features, y_edu_parents, test_size=0.2, stratify=y_edu_parents, random_state=42) # Bildungsstatus der Eltern
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_sek_habitus_features, y_finance_status, test_size=0.2, stratify=y_finance_status, random_state=42) # Finanzielle Situation in der Familie
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_sek_habitus_features, y_univ, test_size=0.2, stratify=y_univ, random_state=42) # Hochschultyp
#X_train, X_test, y_train, y_test = train_test_split(xml_freq_sek_habitus_features, y_subj, test_size=0.2, stratify=y_subj, random_state=42) # Fach

# IF-IDF im Tag <sek_habitus> 
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_sek_habitus_features, y_nation, test_size=0.2, stratify=y_nation, random_state=42) # Staatsangehörigkeit
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_sek_habitus_features, y_migration, test_size=0.2, stratify=y_migration, random_state=42) # Migrationshintergrund
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_sek_habitus_features, y_residence_perm, test_size=0.2, stratify=y_residence_perm, random_state=42) # Bildungsinländer*in oder Bildungsausländer*in, AT
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_sek_habitus_features, y_german_lang, test_size=0.2, stratify=y_german_lang, random_state=42) # Erwerb der deutschen Sprache
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_sek_habitus_features, y_edu_parents, test_size=0.2, stratify=y_edu_parents, random_state=42) # Bildungsstatus der Eltern
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_sek_habitus_features, y_finance_status, test_size=0.2, stratify=y_finance_status, random_state=42) # Finanzielle Situation in der Familie
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_sek_habitus_features, y_univ, test_size=0.2, stratify=y_univ, random_state=42) # Hochschultyp
#X_train, X_test, y_train, y_test = train_test_split(xml_tfidf_sek_habitus_features, y_subj, test_size=0.2, stratify=y_subj, random_state=42) # Fach 

# Pipeline mit RandomUnderSampler + SVC
pipeline = Pipeline([
    ('under', RandomUnderSampler(random_state=42)),
    ('svc', SVC(class_weight='balanced', decision_function_shape='ovo'))
])

# Parameter-Tuning mit Grid für GridSearchCV
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'svc__kernel': ['rbf', 'poly', 'linear']
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Optimale Parameter:", grid_search.best_params_)

# Test
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))

'''Ermittlung der bedeutendsten Features bei einem F-Maß über 0.80'''
def get_top_features_for_class(df, columns, class_column, target_class, top_n=10):
    # Filtern nach Zielklasse
    df_subset = df[df[class_column] == target_class]
    
    aggregate_topics = {}
    
    for col in columns:
        # Spalte von String zu Dict 
        dict_series = df_subset[col].apply(ast.literal_eval)
        
        # Summe aller Dicts
        for d in dict_series:
            for topic, value in d.items():
                aggregate_topics[topic] = aggregate_topics.get(topic, 0) + value
    
    # Durchschnitt
    n = len(df_subset)
    if n == 0:
        print(f"Keine Daten für Klasse '{target_class}' gefunden.")
        return
    
    average_topics = {k: v / n for k, v in aggregate_topics.items()}
    
    # Sortierung (abst.)
    sorted_topics = sorted(average_topics.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top {top_n} Features für Klasse '{target_class}':")
    for topic, score in sorted_topics[:top_n]:
        print(f"{topic}: {score:.4f}")

def get_top_num_features_for_class(df, columns, class_column, target_class, top_n=10):
    # Filtern nach Zielklasse
    df_subset = df[df[class_column] == target_class]

    if df_subset.empty:
        print(f"Keine Daten für Klasse '{target_class}' gefunden.")
        return

    # Mittelwert
    mean_values = df_subset[columns].mean().sort_values(ascending=False)

    print(f"\nTop {top_n} Features für Klasse '{target_class}':")
    for ngram, score in mean_values.head(top_n).items():
        print(f"{ngram}: {score:.4f}")

# LDA bei Migrationshintergrund
#get_top_features_for_class(df, colums=lda_columns, class_column='tn_migr', target_class='nur TN eingewandert', top_n=10)
# BERTopic bei Staatsangehörigkeit
#get_top_features_for_class(df, colums=bertopic_columns, class_column='tn_staang_simplified', target_class='Deutsch / EU', top_n=10)
# N-Gramme (Freuenzen) bei Staatsangehörigkeit
#ngram_columns_c = ngram_count_features.columns.tolist()
#get_top_num_features_for_class(df=df, columns=ngram_columns_c, class_column='tn_staang_simplified', target_class='Deutsch / EU', top_n=10)
# N-Gramme (TF-IDF) bei Bildunggtatus der Eltern
#ngram_columns_t = ngram_tfidf_features.columns.tolist()
#get_top_num_features_for_class(df=df, columns=ngram_columns_t, class_column='tn_hsabsch_eltern', target_class='beide keinen', top_n=10)
# Frequenzen der Wörter mit dem höchsten IG bei Staatsangehörigkeit, Aufenthaltssatus und dem Hochschulabschluss der Eltern
ig_features = ig_features.columns.tolist()
#get_top_num_features_for_class(df=df, columns=ig_features, class_column='tn_aufenth_simplified', target_class='Bildungsausländer*innen, AT', top_n=10)
#get_top_num_features_for_class(df=df, columns=ig_features, class_column='tn_aufenth_simplified', target_class='Bildungsinländer*innen', top_n=10)
get_top_num_features_for_class(df=df, columns=ig_features, class_column='tn_staang_simplified', target_class='Deutsch / EU', top_n=10)
get_top_num_features_for_class(df=df, columns=ig_features, class_column='tn_staang_simplified', target_class='Nicht-EU / Staatenlos', top_n=10)
#get_top_num_features_for_class(df=df, columns=ig_features, class_column='tn_hsabsch_eltern', target_class='beide Elternteile', top_n=10)
#get_top_num_features_for_class(df=df, columns=ig_features, class_column='tn_hsabsch_eltern', target_class='beide keinen', top_n=10)
# Frequenzen der Attribute der XML-Tags bei Staatsangehörigkeit, Aufenthaltssatus und dem Hochschulabschluss der Eltern
# xml_freq_all_features = xml_freq_all_features.columns.tolist()
#get_top_num_features_for_class(df=df, columns=xml_freq_all_features, class_column='tn_staang_simplified', target_class='Deutsch / EU', top_n=10)
#get_top_num_features_for_class(df=df, columns=xml_freq_all_features, class_column='tn_aufenth_simplified', target_class='Bildungsausländer*innen, AT', top_n=10)
#get_top_num_features_for_class(df=df, columns=xml_freq_all_features, class_column='tn_aufenth_simplified', target_class='Bildungsinländer*innen', top_n=10)
#get_top_num_features_for_class(df=df, columns=xml_freq_all_features, class_column='tn_hsabsch_eltern', target_class='beide Elternteile', top_n=10)
#get_top_num_features_for_class(df=df, columns=xml_freq_all_features, class_column='tn_hsabsch_eltern', target_class='beide keinen', top_n=10)
# ZF-IDF der Attribute der XML-Tags bei Staatsangehörigkeit
# xml_tfidf_all_features = xml_tfidf_all_features.columns.tolist()
#get_top_num_features_for_class(df=df, columns=xml_tfidf_all_features, class_column='tn_staang_simplified', target_class='Deutsch / EU', top_n=10)
#get_top_num_features_for_class(df=df, columns=xml_tfidf_all_features, class_column='tn_staang_simplified', target_class='Nicht-EU / Staatenlos', top_n=10)
# Frequenzen der Attribute des XML-Tags <sek_habitus> beim Aufenthaltssatus
xml_freq_sek_habitus_features = xml_freq_sek_habitus_features.columns.tolist()
# get_top_num_features_for_class(df=df, columns=xml_freq_sek_habitus_features, class_column='tn_aufenth_simplified', target_class='Bildungsausländer*innen, AT', top_n=10)
# get_top_num_features_for_class(df=df, columns=xml_freq_sek_habitus_features, class_column='tn_aufenth_simplified', target_class='Bildungsinländer*innen', top_n=10)
get_top_num_features_for_class(df=df, columns=xml_freq_sek_habitus_features, class_column='tn_staang_simplified', target_class='Deutsch / EU', top_n=10)
get_top_num_features_for_class(df=df, columns=xml_freq_sek_habitus_features, class_column='tn_staang_simplified', target_class='Nicht-EU / Staatenlos', top_n=10)

''' Confusion-Matrix '''

# Optimale Parameter für Staatsangehörigkeit, Wörter mit Informationsgewinn
# pipeline = Pipeline([
#     ('under', RandomUnderSampler(random_state=42)),
#     ('randfor', SVC(
#         kernel='rbf',
#         gamma=0.01,
#         C=10
#     ))
# ])

# Optimale Parameter für Aufenthaltsstatus, <sek_habitus>
pipeline = Pipeline([
    ('under', RandomUnderSampler(random_state=42)),
    ('randfor', SVC(
        kernel='linear',
        gamma='scale',
        C=10
    ))
])

# Optimale Parameter für Bildungsabschluss der Eltern, N-Gramme+TF-IDF
# pipeline = Pipeline([
#     ('under', RandomUnderSampler(random_state=42)),
#     ('randfor', SVC(
#         kernel='poly',
#         gamma='scale',
#         C=10
#     ))
# ])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
labels = pipeline.named_steps['randfor'].classes_

wrapped_labels = [textwrap.fill(label, width=10) for label in labels]

cm = confusion_matrix(y_test, y_pred, labels=labels)

cmap = sns.diverging_palette(80, 267, l=50, center="light", as_cmap=True)
sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=wrapped_labels, yticklabels=wrapped_labels)
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Tatsächliche Klasse")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()