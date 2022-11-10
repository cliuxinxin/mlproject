import spacy
from spacy import displacy

import concise_concepts

data = {
    "fruit": ["apple", "pear", "orange"],
    "vegetable": ["broccoli", "spinach", "tomato"],
    "meat": ["beef", "pork", "fish", "lamb"],
}

text = """
    Heat the oil in a large pan and add the Onion, celery and carrots.
    Then, cook over a medium–low heat for 10 minutes, or until softened.
    Add the courgette, garlic, red peppers and oregano and cook for 2–3 minutes.
    Later, add some oranges and chickens. """

nlp = spacy.load("en_core_web_lg", disable=["ner"])

nlp.add_pipe(
    "concise_concepts",
    config={
        "data": data,
        "ent_score": True, # Entity Scoring section
        "verbose": True,
        "exclude_pos": ["VERB", "AUX"],
        "exclude_dep": ["DOBJ", "PCOMP"],
        "include_compound_words": False,
        "json_path": "./fruitful_patterns.json",
    },
)
doc = nlp(text)
