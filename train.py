from __future__ import print_function
from attention_decoder import AttentionDecoder
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.chrf_score import corpus_chrf
from nltk.translate.bleu_score import corpus_bleu
import json


DATA = '../QG/wiki_sentence_pairs/data.json'
MAX_NUM_WORDS = 4000
MAXLEN = 15
REVERSE = True
MAX_DATA = 2500
EPOCHS = 1
ATTENTION = False


with open(DATA) as f:
    f.seek(0)
    data_json = json.load(f)
tok = Tokenizer(MAX_NUM_WORDS)
tok.fit_on_texts(data_json['normal'][:MAX_DATA] + data_json['simple'][:MAX_DATA])


indx2w = {v: k for k, v in tok.word_index.items()}


source = tok.texts_to_sequences(data_json['normal'][:MAX_DATA])
target = tok.texts_to_sequences(data_json['simple'][:MAX_DATA])
padded_source = pad_sequences(source, maxlen=MAXLEN)
if ATTENTION:
    padded_target = pad_sequences(target, maxlen=MAXLEN)
    y = np.zeros((len(padded_target), MAXLEN, MAX_NUM_WORDS), dtype=np.bool)
else:
    padded_target = pad_sequences(target, maxlen=MAXLEN - 5)
    y = np.zeros((len(padded_target), MAXLEN - 5, MAX_NUM_WORDS), dtype=np.bool)


x = np.zeros((len(padded_source), MAXLEN, MAX_NUM_WORDS), dtype=np.bool)


X = to_categorical(padded_source)
Y = to_categorical(padded_target)

for i, k in enumerate(X):
    x[i] = k
for i, k in enumerate(Y):
    y[i] = k

if REVERSE:
    x = np.flip(x, axis=1)



# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 3

print('Build model...')
model = Sequential()

# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, MAX_NUM_WORDS), return_sequences=True))
# As the decoder RNN's input, repeatedly provide with the last hidden state of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
if ATTENTION:
    model.add(AttentionDecoder(HIDDEN_SIZE, MAX_NUM_WORDS))

else:
    model.add(layers.RepeatVector(MAXLEN - 5))
    # The decoder RNN could be multiple layers stacked or a single layer.
    for _ in range(LAYERS):
        # By setting return_sequences to - 5 True, return not only the last output but
        # all the outputs so far in the form of (num_samples, timesteps,
        # output_dim). This is necessary as TimeDistributed in the below expects
        # the first dimension to be the timesteps.
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))
        model.add(layers.Dropout(0.3))

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
    model.add(layers.TimeDistributed(layers.Dense(MAX_NUM_WORDS)))
    model.add(layers.Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()



try:
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(x_val, y_val))

except KeyboardInterrupt as e:
    print('Model training stopped early.')


finally:
    if ATTENTION:
        model.save('models/seq2seq_lstm_attention.h5')

    else:
        model.save('models/seq2seq_lstm.h5')


#test model
if REVERSE:
    x_val = np.flip(x_val, axis=1)
preds = model.predict(x_val, verbose=0)
print(preds)
simpl_sentences_h = []
simpl_sentences_r = []
for i in range(len(preds)):
    print('Source: ', end='')
    try:
        s = ''.join([indx2w[np.argmax(w)] for w in y_val[i]])
        simpl_sentences_h.append(s)
        print(s)
    except KeyError:
        continue
    print('Target: ', end='')
    try:
        if ATTENTION:
            s = ''.join([indx2w[np.argmax(w)] for w in preds[i]])
        else:
            s = ''.join([indx2w[w] for w in preds[i]])
        simpl_sentences_r.append(s)
        print(s)
        print()
    except KeyError:
        continue


print('Chrf score: %s' % corpus_chrf(simpl_sentences_h, simpl_sentences_r))
print('Bleu score: %s' % corpus_bleu(simpl_sentences_h, simpl_sentences_r))