
"""Dieser Code zum Extrahieren und Abspeichern der Features für die Klassifizierung wurde auf Google Colab auf T4 GPU ausgeführt"""

import torch.serialization
import numpy
import spacy_stanza
# Umgang der Warnung vom "trusted" Unpicking von PyTorch
def load_pipeline_trusted(*args, **kwargs):
    import torch
    orig_load = torch.load

    def load_override(f, *args, **kwargs):
        kwargs['weights_only'] = False
        return orig_load(f, *args, **kwargs)

    torch.load = load_override
    try:
        pipe = spacy_stanza.load_pipeline(*args, **kwargs)
    finally:
        torch.load = orig_load
    return pipe

with torch.serialization.safe_globals([numpy.core.multiarray._reconstruct]):
    nlp_deu = load_pipeline_trusted("de", use_gpu=True)

if "ner" in nlp_deu.pipe_names:
    nlp_deu.remove_pipe("ner")
import stanza
stanza.download("de")
from google.colab import files
import xml.etree.ElementTree as ET
import pandas as pd
import re
import requests

# Datei hochladen
uploaded = files.upload()
xml_filename = list(uploaded.keys())[0]
# Einlesen der XML-Datei als DataFrame
def parse_xml_to_dataframe(xml_path: str, stopwords: list) -> pd.DataFrame:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    daten = []

    for tn in root.findall('tn'):
        eintrag = {}

        # ID
        eintrag['tn_id'] = tn.findtext('tn_id', default='')

        # Labels (Features)
        label = tn.find('label')
        if label is not None:
            for child in label:
                eintrag[child.tag] = child.text.strip() if child.text else ''

        # Habitus-Untertags (Texte bereinigen und lemmatisieren)
        habitus = tn.find('habitus')
        if habitus is not None:
            for elem in habitus:
                raw_text = elem.text.strip() if elem.text else ''
                cleaned = clean_text(raw_text, stopwords)
                lemmatized = lemmatize_text(cleaned)
                eintrag[elem.tag] = lemmatized

        daten.append(eintrag)

    return pd.DataFrame(daten)

# ISO-Stopwordliste von Diaz, Gene: https://github.com/stopwords-iso/stopwords-iso
def load_stopwords(languages=["de"]):
    url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-iso/master/stopwords-iso.json"
    response = requests.get(url)
    if response.status_code == 200:
        stopwords_json = response.json()
        stopwords = []
        for lang in languages:
            stopwords += stopwords_json.get(lang, [])
        return stopwords
    else:
        print(f"Fehler beim Laden der Stopwörter. Statuscode: {response.status_code}")
        return []

# Entfernen der Stopwörter, Ziffern und Zeichen
def clean_text(text, stopwords):
    text = re.sub(r'[^\w\s]', '', text)  # Sonderzeichen
    text = re.sub(r'\d+', '', text)      # Ziffern
    words = text.split()
    cleaned_text = " ".join([word for word in words if word.lower() not in stopwords])
    return cleaned_text

# Lemmatisierung des Textes
def lemmatize_text(text):
    text = str(text)
    doc = nlp_deu(text)
    return ' '.join([token.lemma_ for token in doc])

stopwords = load_stopwords(["de"])
df = parse_xml_to_dataframe(xml_filename, stopwords)
df.to_csv("umfrage_lemm.csv", index=False)
print("Verarbeitung abgeschlossen. Datei gespeichert: umfrage_lemm.csv")

# Herunterladen der CSV-Datei mit lemmatisierten Freitextantworten
from google.colab import files
files.download('umfrage_lemm.csv')