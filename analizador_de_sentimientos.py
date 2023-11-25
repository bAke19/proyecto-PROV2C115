import tensorflow.keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

oraciones = [ "Worst Headphones ever. Bluetooth is also not working.", "You can try for better camera options as well, but for this amount, it's great to buy."]

tokenizador =  Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizador.fit_on_texts(oraciones)

indices_dicctionario = tokenizador.word_index

secuencias = tokenizador.texts_to_sequences(oraciones)

secuencias_rellenadas = pad_sequences(secuencias, maxlen=150, padding='post', truncating='post')

modelo = tensorflow.keras.models.load_model('Reviews_Cellphones.h5')

resultado = modelo.predict(secuencias_rellenadas)

clases_predichas = np.argmax(resultado, axis=1)

print(clases_predichas)