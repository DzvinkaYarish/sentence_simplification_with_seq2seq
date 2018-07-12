import seq2seq
from seq2seq.models import AttentionSeq2Seq
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import json
import numpy as np


DATA = '../QG/wiki_sentence_pairs/data.json'
MAX_NUM_WORDS = 4000
MAXLEN = 15
REVERSE = True
MAX_DATA = 2500
EPOCHS = 1

with open(DATA) as f:
    f.seek(0)
    data_json = json.load(f)
tok = Tokenizer(MAX_NUM_WORDS)
tok.fit_on_texts(data_json['normal'][:MAX_DATA] + data_json['simple'][:MAX_DATA])


indx2w = {v: k for k, v in tok.word_index.items()}


source = tok.texts_to_sequences(data_json['normal'][:MAX_DATA])
target = tok.texts_to_sequences(data_json['simple'][:MAX_DATA])
padded_source = pad_sequences(source, maxlen=MAXLEN)
padded_target = pad_sequences(target, maxlen=MAXLEN - 5)



x = np.zeros((len(padded_source), MAXLEN, MAX_NUM_WORDS), dtype=np.bool)
y = np.zeros((len(padded_target), MAXLEN - 5, MAX_NUM_WORDS), dtype=np.bool)

X = to_categorical(padded_source)
Y = to_categorical(padded_target)

for i, k in enumerate(X):
    x[i] = k
for i, k in enumerate(Y):
    y[i] = k

# X = np.expand_dims(padded_source, axis=2)
# y = np.expand_dims(padded_target, axis=2)





if REVERSE:
    X = np.flip(X, axis=1)



# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

# Try replacing GRU, or SimpleRNN.

HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 3



model = AttentionSeq2Seq(input_dim=MAX_NUM_WORDS, input_length=MAXLEN, hidden_dim=128, output_length=MAXLEN-5, output_dim=MAX_NUM_WORDS, depth=4)
model.compile(loss='mse', optimizer='rmsprop')
print(model.summary())
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(x_val, y_val))


model.save('models/seq2seq.h5')

model_l = load_model('models/seq2seq.h5')


#test model
if REVERSE:
    x_val = np.flip(x_val, axis=1)
preds = model_l.predict_classes(x_val, verbose=0)
for i in range(len(preds)):
    print('Source: ', end='')
    for w in y_val[i]:
        try:
            print(indx2w[np.argmax(w)] + ' ', end='')
        except KeyError:
            continue
    print('Target: ', end='')
    for w in preds[i]:
        try:
            # print(np.argmax(w) + '  ')
            print(indx2w[w] + ' ', end='')
        except KeyError:
            continue
    print()
