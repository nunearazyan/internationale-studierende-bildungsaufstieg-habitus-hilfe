import pandas as pd
import xml.etree.ElementTree as ET
import gensim
from gensim import corpora
import bertopic
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from collections import Counter, defaultdict
from sklearn.feature_selection import mutual_info_classif

def read_and_prepare_csv(filepath):
    # Datei laden
    df = pd.read_csv(filepath)

    # Textspalten aus Freitextantwortenkombinieren
    text_columns = [
        "uebergang_unterschiede",
        "anpassung",
        "interaktionen_unterschiede",
        "gesp_hab_extern",
        "gesp_hab_intern"
    ]

    # Kombinierte Textspalte mit Antworten aus allen Spalten
    df["alles_lemmatisiert"] = df[text_columns].fillna("").agg(" ".join, axis=1)
    return df

'''Topic-Modelling'''
# LDA
def lda_feature_extraction(df, text_columns, num_topics=5):
    lda_results = {}

    for text_column in text_columns:
        texts = []
        valid_indices = []

        for idx, row in enumerate(df.itertuples(index=False)):
            text = getattr(row, text_column)
            if isinstance(text, str) and text.strip():
                tokens = text.lower().split()
                tokens = [w for w in tokens if len(w) > 4]
                texts.append(tokens)
                valid_indices.append(idx)

        if not texts:
            continue  # Wenn keine gültigen Texte vorhanden sind
            
        dictionary = corpora.Dictionary(texts)  # Einstellung der Parameter
        dictionary.filter_extremes(no_above=0.2, no_below=5)

        corpus = [dictionary.doc2bow(text) for text in texts]

        lda_model = gensim.models.LdaModel( # LDA aus Gensim
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42
        )

        topic_titles = {}
        for i in range(lda_model.num_topics):
            words = lda_model.show_topic(i, topn=5)
            topic_titles[i] = " | ".join([word for word, _ in words]) if words else f"{i}: (leer)"

        topic_distributions = []
        for i in range(len(texts)):
            dist = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
            topic_score_dict = {
                topic_titles[topic_id]: round(prob, 4)
                for topic_id, prob in dist
            }
            topic_distributions.append(topic_score_dict)

        df_valid = df.iloc[valid_indices].copy()
        result_column = f"{text_column}_lda_topics"
        df[result_column] = None
        df.loc[df_valid.index, result_column] = topic_distributions

        lda_results[text_column] = topic_titles

    return df, lda_results

# BERTopic 
def bert_feature_extraction(df, text_columns, min_topic_size=5, min_cluster_size=5):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True, alpha=0.15)

    topic_models = {}

    for col in text_columns:
        texts = df[col].fillna("").astype(str).tolist()

        # BERTopic-Modell samt Parameter und HDBSCAN
        topic_model = BERTopic(
            embedding_model=embedding_model,
            language='german',
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True,
            verbose=True,
            min_topic_size=min_topic_size
        )

        topics, probs = topic_model.fit_transform(texts)

        def topic_probs_dict(probs_row):
            result = {}
            if probs_row is None or not hasattr(probs_row, '__iter__'):
                return result

            for i, prob in enumerate(probs_row):
                if prob > 0.01:
                    try:
                        label_words = topic_model.get_topic(i)
                        label = " | ".join([word for word, _ in label_words[:5]])
                        result[label] = round(prob, 4)
                    except KeyError:
                        continue
            return result

        df[f"{col}_bertopic_topics"] = [topic_probs_dict(p) for p in probs]
        topic_models[col] = topic_model

    return df, topic_models

'''Sentiment-Analyse'''
# EmoLex
def emolex_feature_extraction(df, text_column="alles_lemmatisiert"):
    emolex_path = "German-NRC-EmoLex.txt"
    emolex_df = pd.read_csv(emolex_path, sep="\t")

    # Emotionen
    emotion_labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
                      'negative', 'positive', 'sadness', 'surprise', 'trust']

    # Wörterbuch, bestehend aus emotionatragenden Wörtern und entsprechenden Emotionen
    emolex_dict = {
        row['German Word'].lower(): {emo: int(row[emo]) for emo in emotion_labels}
        for _, row in emolex_df.iterrows()
        if any(int(row[emo]) for emo in emotion_labels)
    }

    # Berechung der Verteilung von Emotionen in Texten
    def compute_emotions(text):
        tokens = text.lower().split()
        emotion_sum = {emo: 0 for emo in emotion_labels}
        count = 0

        for token in tokens:
            if token in emolex_dict:
                count += 1
                for emo in emotion_labels:
                    emotion_sum[emo] += emolex_dict[token][emo]

        # Berechnung des Durchschnitts
        if count > 0:
            return {emo: round(val / count, 4) for emo, val in emotion_sum.items()}
        else:
            return {emo: 0.0 for emo in emotion_labels}

    df["emolex_emotions"] = df[text_column].astype(str).apply(compute_emotions)

    return df

# germansentiment
def german_sentiment_bert_feature_extraction(df, text_column="alles_lemmatisiert"):
    # Sentiment-Analyse Pipeline
    classifier = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

    def analyze_sentiment(text):
        if not isinstance(text, str) or not text.strip():
            return {"neutral": 0.0}  # leere oder ungültige Texte

        result = classifier(text[:512])[0]  # max 512 Tokens
        label = result['label'].lower() 
        score = round(result['score'], 4)

        return {label: score}

    # Anwendung auf Antworten
    df["germansentiment_emotions"] = df[text_column].apply(analyze_sentiment)

    return df

'''N-Gramme (N=1;4)'''
# Frequnzen und TF-IDF der Top-20 Uni-, Bi-, Tri- und Quadrigrammen
def extract_top_ngrams_by_each_n(df, text_column="alles_lemmatisiert", ngram_range=(1, 4), top_k=20):
    texts = df[text_column].astype(str).tolist()
    all_df_counts = []
    all_df_tfidf = []

    for n in range(ngram_range[0], ngram_range[1] + 1):
        # Count
        count_vect = CountVectorizer(ngram_range=(n, n), analyzer='word', min_df=2)
        count_matrix = count_vect.fit_transform(texts)
        count_vocab = count_vect.get_feature_names_out()
        count_sums = count_matrix.sum(axis=0).A1
        count_top_indices = count_sums.argsort()[::-1][:top_k]
        count_top_ngrams = count_vocab[count_top_indices]
        count_top_matrix = count_matrix[:, count_top_indices]
        df_counts = pd.DataFrame(
            count_top_matrix.toarray(),
            columns=[f"count_{n}gram_{ng}" for ng in count_top_ngrams]
        )
        all_df_counts.append(df_counts)

        # TF-IDF
        tfidf_vect = TfidfVectorizer(ngram_range=(n, n), analyzer='word', min_df=2)
        tfidf_matrix = tfidf_vect.fit_transform(texts)
        tfidf_vocab = tfidf_vect.get_feature_names_out()
        tfidf_sums = tfidf_matrix.sum(axis=0).A1
        tfidf_top_indices = tfidf_sums.argsort()[::-1][:top_k]
        tfidf_top_ngrams = tfidf_vocab[tfidf_top_indices]
        tfidf_top_matrix = tfidf_matrix[:, tfidf_top_indices]
        df_tfidf = pd.DataFrame(
            tfidf_top_matrix.toarray(),
            columns=[f"tfidf_{n}gram_{ng}" for ng in tfidf_top_ngrams]
        )
        all_df_tfidf.append(df_tfidf)

    # Zusammenführen
    df_counts_all = pd.concat(all_df_counts, axis=1)
    df_tfidf_all = pd.concat(all_df_tfidf, axis=1)

    scaler = MinMaxScaler()

    df_counts_scaled = pd.DataFrame(scaler.fit_transform(df_counts_all), columns=df_counts_all.columns, index=df_counts_all.index)
    df_tfidf_scaled = pd.DataFrame(scaler.fit_transform(df_tfidf_all), columns=df_tfidf_all.columns, index=df_tfidf_all.index)

    return df_counts_scaled, df_tfidf_scaled

'''skalierte Frequenz von Top-Wörtern'''
# skalierte Frequenz von Top-50 Wörter mit dem höchsten Informationsgewinn
def info_gain_count_combined(df, text_column, target_columns, top_k=50):
    texts = df[text_column].fillna("").astype(str).tolist()

    # Basis-Vektorisierung
    base_vectorizer = CountVectorizer()
    X_all = base_vectorizer.fit_transform(texts)
    feature_names = base_vectorizer.get_feature_names_out()

    # Wörter sammeln
    unique_words = set()

    # Ermittlung der Top-50 Wörter mit dem höchsten Informationsgewinn je Antwortsequenz mithilfe von mutual_info_classif
    for target_col in target_columns:
        if target_col not in df.columns:
            print(f"Spalte {target_col} nicht gefunden")
            continue
        
        y = df[target_col]
        valid_idx = y.notna()
        X_valid = X_all[valid_idx]
        y_valid = y[valid_idx]

        mi = mutual_info_classif(X_valid, y_valid, discrete_features=True)
        top_indices = mi.argsort()[::-1][:top_k]
        top_words = feature_names[top_indices]
        unique_words.update(top_words)

    # Wortzählungen und Skalierung für eindeutige Wörter
    feature_columns = []
    for word in sorted(unique_words):
        vec = CountVectorizer(vocabulary=[word])
        X_word = vec.fit_transform(texts)
        counts = X_word.toarray().ravel()

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(counts.reshape(-1, 1)).ravel()
        feature_columns.append(pd.Series(scaled, name=word))

    # Integration der Werte ins DataFrame
    result_df = pd.concat(feature_columns, axis=1)
    result_df.columns = [f"TopIG_{col}" for col in result_df.columns]
    df = pd.concat([df, result_df], axis=1)

    return df

'''Features auf der Basis der XML-Annotation'''
# skalierte Frequenzen von XML-Tags
def extract_scaled_tag_counts_from_xml(xml_path, df_main):

    # Einlesen der XML-Datei
    tree = ET.parse(xml_path)
    root = tree.getroot()

    all_tag_keys = set()
    tn_data = []

    # Auslesen und Zählen der Tags in jeder Antwortsequenz
    for tn in root.findall("tn"):
        tn_id = tn.findtext("tn_id", default="")
        tag_counter = Counter()

        habitus = tn.find("habitus")
        container_tags = {
            "uebergang_unterschiede",
            "anpassung",
            "interaktionen_unterschiede",
            "gesp_hab_extern",
            "gesp_hab_intern"
        }

        if habitus is not None:
            for elem in habitus.iter():
                if elem is habitus or elem.tag in container_tags:
                    continue

                tag_name = elem.tag
                attrib_str = " ".join([f'{k}="{v}"' for k, v in elem.attrib.items()])
                full_tag = f"<{tag_name} {attrib_str}>".strip()
                tag_counter[full_tag] += 1
                all_tag_keys.add(full_tag)

        entry = {"tn_id": tn_id}
        entry.update(tag_counter)
        tn_data.append(entry)

    # DataFrame mit allen Features
    df_tags = pd.DataFrame(tn_data).fillna(0)

    for tag in sorted(all_tag_keys):
        if tag not in df_tags.columns:
            df_tags[tag] = 0

    df_tags.set_index("tn_id", inplace=True)

    # Skalieren mit MinMaxScaler
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df_tags)
    freq_columns = [f"FREQ{col}" for col in df_tags.columns]
    df_scaled = pd.DataFrame(scaled_array, columns=freq_columns, index=df_tags.index).reset_index()

    # Integration ins Original-DataFrame
    df_combined = pd.merge(df_main, df_scaled, on="tn_id", how="left")
    return df_combined

# TF-IDF von XML-Tags
def extract_tfidf_tag_features_from_xml(xml_path, df_main):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    container_tags = {
        "uebergang_unterschiede",
        "anpassung",
        "interaktionen_unterschiede",
        "gesp_hab_extern",
        "gesp_hab_intern"
    }

    tn_ids = []
    tag_docs = []

    for tn in root.findall("tn"):
        tn_id = tn.findtext("tn_id", default="")
        tn_ids.append(tn_id)

        habitus = tn.find("habitus")
        tags = []

        if habitus is not None:
            for elem in habitus.iter():
                if elem is habitus or elem.tag in container_tags:
                    continue

                tag_name = elem.tag
                attrib_str = " ".join([f'{k}="{v}"' for k, v in elem.attrib.items()])
                full_tag = f"<{tag_name} {attrib_str}>".strip()
                tags.append(full_tag)

        tag_docs.append(" ".join(tags))

    # TF-IDF-Vektorisierung
    vectorizer = TfidfVectorizer(token_pattern=r"<[^>]+>")
    tfidf_matrix = vectorizer.fit_transform(tag_docs)
    feature_names = vectorizer.get_feature_names_out()
    prefixed_names = [f'TFIDF{tag}' for tag in feature_names]

    # Skalierung mit MinMaxScaler
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(tfidf_matrix.toarray())

    tfidf_df = pd.DataFrame(scaled_array, columns=prefixed_names)
    tfidf_df.insert(0, "tn_id", tn_ids)

    # Integration ins Original-DataFrame
    df_combined = pd.merge(df_main, tfidf_df, on="tn_id", how="left")
    return df_combined

'''Anwendung'''
def main():
    filepath = "umfrage_lemm.csv"
    xml_path = "umfrage_gelabelt_annotiert.xml"

    # Daten einlesen und vorbereiten
    df = read_and_prepare_csv(filepath)

    text_columns = [
        "uebergang_unterschiede",
        "anpassung",
        "interaktionen_unterschiede",
        "gesp_hab_extern",
        "gesp_hab_intern",
        "alles_lemmatisiert"
    ]

    # LDA
    df, lda_topic_maps = lda_feature_extraction(df, text_columns=text_columns, num_topics=5)

    for col in text_columns:
        topic_col = f"{col}_lda_topics"
        df[topic_col] = df[topic_col].apply(str)

    # BERTopic-Modellierung
    df, topic_models = bert_feature_extraction(df, text_columns=text_columns)

    # EmoLex
    df = emolex_feature_extraction(df)

    # germansentiment
    df = german_sentiment_bert_feature_extraction(df)

    # N-Gramme
    df_counts, df_tfidf = extract_top_ngrams_by_each_n(
        df,
        text_column="alles_lemmatisiert",
        ngram_range=(1, 4),
        top_k=20
        )
    df = pd.concat([df, df_counts, df_tfidf], axis=1)

    # Top-IG-Wörter
    df = info_gain_count_combined(
        df,
        text_column="alles_lemmatisiert",
        target_columns=[
        "tn_staang", "tn_migr", "tn_aufenth", "tn_dt_spr",
        "tn_hsabsch_eltern", "tn_fin_sit", "tn_hochschulart", "tn_fach"
        ],
        top_k=50
        )

    # skalierte Frequenzen der XML-Tags
    df = extract_scaled_tag_counts_from_xml(xml_path, df)

    # TF-IDF Frequenzen der XML-Tags
    df = extract_tfidf_tag_features_from_xml(xml_path, df)

    # Gemeinsame Datei mit Features
    df.to_csv("umfrage_featured.csv", index=False)
    print("Datei umfrage_featured.csv erfolgreich gespeichert.")
    
    #print(df.columns)
    #print(text_columns)

if __name__ == "__main__":
    main()
