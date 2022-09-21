import spacy
# make the factory work
from rel_pipe import make_relation_extractor

# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors
nlp = spacy.load('training/model-best')