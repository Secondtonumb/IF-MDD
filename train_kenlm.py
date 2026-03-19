
from lm.builder import KenLMBuilder
builder = KenLMBuilder()
builder.train(text_file='corpus.txt', 
                output='language_model.arpa',
                order=3)