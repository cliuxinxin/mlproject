"""Convert entity annotation from spaCy v2 TRAIN_DATA format to spaCy v3
.spacy format."""
import srsly
import typer
import warnings
from pathlib import Path

import spacy
from spacy.tokens import DocBin,Doc

def get_all_relations(path):
    label_type = {}
    for entry in srsly.read_jsonl(path):
        relations = entry["relations"]
        for relation in relations:
            label_type[relation["type"]] = 1
    return list(label_type.keys())

def convert(lang: str, input_path: Path, output_path: Path):
    Doc.set_extension("rel", default={})
    nlp = spacy.blank(lang)
    db = DocBin(store_user_data=True)
    label_types = get_all_relations(input_path)
    for entry in srsly.read_jsonl(input_path):
        text = entry['text']
        doc = nlp.make_doc(text)
        ents = []
        span_starts = set()
        span_end_to_start = {}
        for one_label in entry['entities']:
            span = doc.char_span(one_label['start_offset'], one_label['end_offset'], label=one_label['label'],alignment_mode="contract")
            span_end_to_start[one_label['id']] = span.start
            span_starts.add(span.start)
            if span is None:
                msg = f"Skipping entity [{one_label['start_offset']}, {one_label['end_offset']}, {one_label['label']}] in the following text because the character span '{doc.text[one_label['start_offset']:one_label['end_offset']]}' does not align with token boundaries:\n\n{repr(text)}\n"
                warnings.warn(msg)
            else:
                ents.append(span)
        # Parse the relations
        rels = {}
        for x1 in span_starts:
            for x2 in span_starts:
                rels[(x1, x2)] = {}
        relations = entry["relations"]
        for relation in relations:
            start = span_end_to_start[relation["from_id"]]
            end = span_end_to_start[relation["to_id"]]
            label = relation["type"]
            rels[(start, end)][label] = 1.0
        for x1 in span_starts:
            for x2 in span_starts:
                    for label in label_types:
                        if label not in rels[(x1, x2)]:
                            rels[(x1, x2)][label] = 0.0
        doc.ents = ents
        doc._.rel = rels
        db.add(doc)
    db.to_disk(output_path)


if __name__ == "__main__":
    typer.run(convert)
    # convert('zh', Path('assets/train_dev.json'), Path('corpus/train_dev.spacy'))






