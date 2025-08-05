import pandas as pd
import xml.etree.ElementTree as ET

# Vereinfachung der Kategorien
def simplify_staang(value):
    if "Nicht-EU / Staatenlos" in value:
        return "Nicht-EU / Staatenlos"
    else:
        return "Deutsch und EU"
    
def simplify_residence_permit_category(value):
    if value in [
        "16b AufenthG",
        "25 AufenthG",
        "24 AufenthG"
    ]:
        return "Bildungsausländer*innen mit einem Aufenthaltstitel"
    else:
        return "Bildungsinländer*innen"
    
def simplify(text):
    return text.strip().replace("\n", " ") if text else "unbekannt"

# XML-Datei einlesen
xml_file = "umfrage_gelabelt_annotiert.xml"
tree = ET.parse(xml_file)
root = tree.getroot()

# Labels
label_fields = ["tn_staang", "tn_migr", "tn_aufenth", "tn_dt_spr", "tn_hsabsch_eltern"]

# Tags für Enfremdung und Habitus nach dem Aufstieg
tag_definitions = {
    "entfremdung": ["typ"],
    "habitus_nach_aufstieg": ["typ", "zugehoerigkeit"]
}

data = []

# Zugrff auf die Klassen
for tn in root.findall("tn"):
    label = tn.find("label")
    if label is None:
        continue

    # Labels
    labels = {}
    for field in label_fields:
        raw_value = simplify(label.findtext(field))
        if field == "tn_staang":
            labels[field] = simplify_staang(raw_value)
        elif field == "tn_aufenth":
            labels[field] = simplify_residence_permit_category(raw_value)
        else:
            labels[field] = raw_value

    # Oteration über entsprechende Tag und Sammeln der Ergebnisse
    for tag, attrs in tag_definitions.items():
        for elem in tn.iter(tag):
            entry = labels.copy()
            entry["tag"] = tag

            for attr in attrs:
                entry[attr] = elem.attrib.get(attr, "unbekannt")

            data.append(entry)

# In DataFrame umwandeln
df = pd.DataFrame(data)

# Tabelle
for label in label_fields:
    print(f"\n--- Auswertung nach '{label}' ---")

    for attr in ["typ", "zugehoerigkeit"]: 
        if attr in df.columns:
            print(f"\nTabelle: Gruppierung nach 'tag' und '{attr}'\n")
            table = pd.pivot_table(
                df,
                index=[label],
                columns=["tag", attr],
                aggfunc="size",
                fill_value=0
            )
            print(table)

# TXT-Datei mit Ergebnissen
with open("habitus_entfremdung_erebnisse.txt", "w", encoding="utf-8") as f:
    for label in label_fields:
        f.write(f"\n--- Auswertung nach '{label}' ---\n")

        for attr in ["typ", "zugehoerigkeit"]:
            if attr in df.columns:
                f.write(f"\nTabelle: Gruppierung nach 'tag' und '{attr}'\n\n")
                table = pd.pivot_table(
                    df,
                    index=[label],
                    columns=["tag", attr],
                    aggfunc="size",
                    fill_value=0
                )
                f.write(table.to_string())
                f.write("\n\n") 

print("Ergebnisse wurden in 'habitus_entfremdung_erebnisse.txt' gespeichert.")