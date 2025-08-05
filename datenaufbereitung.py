import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Pfad zur CSV-Datei
csv_datei = "umfragedaten_gefiltert.csv"
# Pfad zur XML-Ausgabedatei
xml_datei = "umfrage_gelabelt.xml"

root = ET.Element("umfrage_ergebnisse")  # Wurzel-Element

with open(csv_datei, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';') # Einlesen
    for row in reader:
        # TN-Bereich
        tn = ET.SubElement(root, "tn")

        # TN-ID
        tn_id = ET.SubElement(tn, "tn_id")
        tn_id.text = row.get("v_1", "").strip()

        # Kathegoriesierungsbereich
        label = ET.SubElement(tn, "label")

        # Staatsangehörigkeit(en) von TN
        sta_mapping = {
            "v_2": "Deutsch",           
            "v_3": "EU",               
            "v_4": "Nicht-EU / Staatenlos",         
            "v_5": "Deutsch",          
            "v_6": "EU",               
            "v_7": "Nicht-EU / Staatenlos",       
            "v_8": "Deutsch",        
            "v_9": "EU",               
            "v_10": "Nicht-EU / Staatenlos"         
        }

        sta = []

        for feld, bezeichnung in sta_mapping.items():
            if row.get(feld, "").strip() == "1":
                sta.append(bezeichnung)

        if sta:
            tn_staang = ET.SubElement(label, "tn_staang")
            tn_staang.text = ", ".join(sta)

        # Migrationshintergrund von TN
        v_11_mapping = {
            "1": "nicht vorhanden",
            "2": "ein Elternteil eingewandert",
            "3": "beide Elternteile eingewandert",
            "4": "nur TN eingewandert",
            "5": "Einwanderungs- geschichte tiefer"
        }

        v_11_wert = row.get("v_11", "").strip()
        if v_11_wert in v_11_mapping:
            antwort = ET.SubElement(label, "tn_migr")
            antwort.text = v_11_mapping[v_11_wert]
        
        # Aufenthaltstitel von TN
        v_12_wert = row.get("v_12", "").strip()
        if v_12_wert == "1":
            aufenth = ET.SubElement(label, "tn_aufenth")
            aufenth.text = "nicht nötig"
        elif v_12_wert == "6":
            # Dann nimm den tatsächlichen Freitext aus einer weiteren Spalte – z. B. "v_13"?
            freitext = row.get("v_13", "").strip()
            if freitext:
                aufenth = ET.SubElement(label, "tn_aufenth")
                aufenth.text = f"{freitext} AufenthG"

        # Erwerb der deutschen Sprache von TN
        v_14_mapping = {
            "1": "Muttersprache nur Deutsch",
            "2": "Muttersprache neben anderer Sprache",
            "3": "Spracherwerb im jungen Alter in DE",
            "4": "Spracherwerb im jungen Alter außerhalb DE",
            "5": "Spracherwerb im erwachsenen Alter"
        }

        v_14_wert = row.get("v_14", "").strip()
        if v_14_wert in v_14_mapping:
            antwort = ET.SubElement(label, "tn_dt_spr")
            antwort.text = v_14_mapping[v_14_wert]

        # Hochschulabschluss im Elternhaus von TN
        v_15_mapping = {
            "1": "beide Elternteile",
            "2": "nur ein Elternteil",
            "3": "beide keinen",
            "4": "ein Elternteil ja und anderer unbekannt",
            "5": "ein Elternteil nein und anderer unbekannt",
            "6": "beide unbekannt"
        }

        v_15_wert = row.get("v_15", "").strip()
        if v_15_wert in v_15_mapping:
            antwort = ET.SubElement(label, "tn_hsabsch_eltern")
            antwort.text = v_15_mapping[v_15_wert]

        # Finanzielle Situation im Elternhaus von TN
        v_16_mapping = {
            "1": "Geld reichte für grundlegende Bedürfnisse und zusätzliche Ausgaben ohne Finanzplanung",
            "2": "Geld reichte für grundlegende Bedürfnisse und zusätzliche Ausgaben mit Finanzplanung",
            "3": "Geld reichte für grundlegende Bedürfnisse vollständig",
            "4": "Geld reichte für grundlegende Bedürfnisse knapp",
            "5": "Geld reichte für ein Teil der grundlegenden Bedürfnisse mit Möglichkeit zu Sonderersparnissen",
            "6": "Geld reichte für ein Teil der grundlegenden Bedürfnisse ohne Möglichkeit zu Sonderersparnissen"
        }

        v_16_wert = row.get("v_16", "").strip()
        if v_16_wert in v_16_mapping:
            antwort = ET.SubElement(label, "tn_fin_sit")
            antwort.text = v_16_mapping[v_16_wert]

        # Hochschulart(en) von TN
        hs_mapping = {
            "v_40": "staatliche Universität",
            "v_41": "private Universität",
            "v_42": "kirchliche Universität",
            "v_43": "staatliche Hochschule",
            "v_44": "private Hochschule",
            "v_45": "kirchliche Hochschule"
        }

        hochschularten = []

        for var, bezeichnung in hs_mapping.items():
            if row.get(var, "").strip() == "1":
                hochschularten.append(bezeichnung)

        if hochschularten:
            tn_hochschule = ET.SubElement(label, "tn_hochschulart")
            tn_hochschule.text = ", ".join(hochschularten)

        # Fachrichtung(en) von TN
        hs_mapping = {
            "v_46": "Erziehungswissenschaften / Sozialwesen / Bildungswissenschaften / Soziale Arbeit",
            "v_47": "Wirtschaftswissenschaften",
            "v_48": "Maschinenbau und Verfahrens- / Elektro- / Informationstechnik",
            "v_49": "Mathematik / Informatik und verwandte Fächer wie Wirtschaftsinformatik oder Data Science",
            "v_50": "Geisteswissenschaften (ohne Sprachwissenschaften)",
            "v_51": "Sprachwissenschaften",
            "v_52": "Agrar-/ Forst- / Ernährungswissenschaften",
            "v_53": "Veterinärmedizin",
            "v_54": "Ingenieurwissenschaften (ohne Maschinenbau und Verfahrens- / Elektro- / Informationstechnik)",
            "v_55": "Sozial- / Gesellschafts- / Politik- / Regional- / Verwaltungswissenschaften",
            "v_56": "Psychologie",
            "v_57": "Naturwissenschaften",
            "v_58": "Rechtswissenschaften",
            "v_59": "Kunst (z. B. bildende Kunst / darstellende Kunst / Musik / Regie) / Kunstwissenschaften",
            "v_60": "Medizin (Human- und Zahnmedizin) / Gesundheitswissenschaften",
            "v_61": "Lehramt",
            "v_62": "Sportwissenschaften",
            "v_63": "Sonstiges"
        }

        hochschularten = []

        for var, bezeichnung in hs_mapping.items():
            if row.get(var, "").strip() == "1":
                hochschularten.append(bezeichnung)

        if hochschularten:
            tn_hochschule = ET.SubElement(label, "tn_fach")
            tn_hochschule.text = ", ".join(hochschularten)

        # Bereich für Habitus
        habitus = ET.SubElement(tn, "habitus")

        # Übergang in das neue Milieu und Unterschiede zum alten
        uebergang_unterschiede = ET.SubElement(habitus, "uebergang_unterschiede")
        uebergang_unterschiede.text = row.get("v_17", "").strip()

        # Anpassung an das neue Milieu
        anpassung = ET.SubElement(habitus, "anpassung")
        anpassung.text = row.get("v_18", "").strip()

        # Interaktionen in beiden Milieus und Unterschiede
        interaktionen_unterschiede = ET.SubElement(habitus, "interaktionen_unterschiede")
        interaktionen_unterschiede.text = row.get("v_19", "").strip()

        # Gespaltener Habitus: externe Ausprägungen (Verhalten, Mimik, Gestig, Sprache)
        gesp_hab_extern = ET.SubElement(habitus, "gesp_hab_extern")
        gesp_hab_extern.text = row.get("v_20", "").strip()

        # Gespaltener Habitus: interne Ausprägungen (Gedanken, Gefühle, innere Erlebnisse)
        gesp_hab_intern = ET.SubElement(habitus, "gesp_hab_intern")
        gesp_hab_intern.text = row.get("v_21", "").strip()

        # Bereich für Hilfestellungen
        hilfe = ET.SubElement(tn, "hilfe")

        # konkrete Hilfsangebote und Begründungen
        for i in range(28, 39, 2):  # Start bei v_28 bis v_39, Schritt 2, da Werte gepaart sind
            angebot_value = row.get(f"v_{i}", "").strip() 
            begruendung_value = row.get(f"v_{i+1}", "").strip()

            if angebot_value != "-99" and begruendung_value != "-99":  # Falls beide Werte nicht "-99", also nicht leer sind
                angebot_tag = ET.SubElement(hilfe, "angebot")
                angebot_tag.text = angebot_value

                # Begründung innerhalb des Hilfsangebots
                begruendung_tag = ET.SubElement(angebot_tag, "begruendung")
                begruendung_tag.text = begruendung_value

# Formatierung der XML-Ausgabedatei
raw_string = ET.tostring(root, encoding='utf-8')
parsed = minidom.parseString(raw_string)
final_xml = parsed.toprettyxml(indent="  ")

# Speichern der XML-Ausgabedatei
with open(xml_datei, "w", encoding="utf-8") as f:
    f.write(final_xml)

print(f"Formatierte und strukturierte XML-Datei mit allen Antworten wurde in '{xml_datei}' gespeichert.")