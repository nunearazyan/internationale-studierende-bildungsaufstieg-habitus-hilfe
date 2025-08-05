import xml.etree.ElementTree as ET
from collections import defaultdict, Counter

def parse_tags_and_attributes(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    tag_attribute_combinations = defaultdict(Counter)

    def recurse(element):
        tag = element.tag
        attr_combo = frozenset(element.attrib.items())
        tag_attribute_combinations[tag][attr_combo] += 1
        
        for child in element:
            recurse(child)

    recurse(root)
    return tag_attribute_combinations

xml_datei = "umfrage_gelabelt_annotiert.xml"
result = parse_tags_and_attributes(xml_datei)

for tag, attr_counter in sorted(result.items()):
    total_count = sum(attr_counter.values())
    print(f"Tag: <{tag}> – Gesamtanzahl mit Attributen: {total_count}")
    
    if not attr_counter:
        print("  (Keine Attribute)")
    else:
        for attr_combo, count in attr_counter.items():
            attrs = ", ".join([f'{k}="{v}"' for k, v in sorted(attr_combo)])
            print(f"  Attribut-Kombination: {attrs if attrs else '(keine)'}")
            print(f"    Anzahl: {count}")
    print()