import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# Extracción de nombres propios
text = "Barack Obama fue el presidente de los Estados Unidos durante dos mandatos consecutivos, desde 2009 hasta 2017."

# Tokenización y etiquetado POS
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

# Extracción de nombres propios
named_entities = ne_chunk(pos_tags)

# Imprimir los nombres propios
for entity in named_entities:
    if hasattr(entity, 'label'):
        print(entity.label(), ' '.join(c[0] for c in entity))
