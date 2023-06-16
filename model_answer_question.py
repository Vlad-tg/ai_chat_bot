import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from nltk.translate.bleu_score import corpus_bleu

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the training data
data = pd.read_csv('output.csv')
questions = data['Question'].tolist()
answers = data['Answer'].tolist()

# Add start and end tokens to the answers
answers = ['<start> ' + answer + ' <end>' for answer in answers]

# Tokenize the questions and answers
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions + answers)

# Add the start and end tokens to the tokenizer
tokenizer.word_index['<start>'] = len(tokenizer.word_index) + 1
tokenizer.word_index['<end>'] = len(tokenizer.word_index) + 1

# Convert the questions and answers to sequences of integers
question_seqs = tokenizer.texts_to_sequences(questions)
answer_seqs = tokenizer.texts_to_sequences(answers)

# Pad the sequences to a fixed length
max_seq_length = max(len(seq) for seq in question_seqs + answer_seqs)
question_seqs_padded = tf.keras.preprocessing.sequence.pad_sequences(question_seqs, maxlen=max_seq_length, padding='post')
answer_seqs_padded = tf.keras.preprocessing.sequence.pad_sequences(answer_seqs, maxlen=max_seq_length, padding='post')

encoder_input_train = question_seqs_padded
decoder_input_train = answer_seqs_padded[:, :-1]  #
decoder_target_train = answer_seqs_padded[:, 1:]

hidden_units = 256
dropout_rate = 0.2

encoder_inputs = Input(shape=(max_seq_length,))
encoder_embedding = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=hidden_units, input_length=max_seq_length)
encoder_lstm = LSTM(units=hidden_units, dropout=dropout_rate, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding(encoder_inputs))
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_seq_length-1,))
decoder_embedding = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=hidden_units, input_length=max_seq_length-1)
decoder_lstm = LSTM(units=hidden_units, dropout=dropout_rate, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding(decoder_inputs), initial_state=encoder_states)
decoder_dense = Dense(units=len(tokenizer.word_index)+1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

optimizer = Adam(learning_rate=0.001)
loss_function = SparseCategoricalCrossentropy()
accuracy_metric = SparseCategoricalAccuracy()
model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy_metric])

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

epochs = 50
batch_size = 128

history = model.fit(
    [encoder_input_train, decoder_input_train],
    decoder_target_train,
    validation_data=0.2,
    batch_size=batch_size,
    epochs=epochs
)

def generate_answer(question):
    question_seq = tokenizer.texts_to_sequences([question])
    question_seq_padded = tf.keras.preprocessing.sequence.pad_sequences(question_seq, maxlen=max_seq_length,
                                                                        padding='post')
    answer_seq_padded = np.zeros((1, max_seq_length - 1))
    answer_seq_padded[0, -1] = tokenizer.word_index['<start>']

    for i in range(max_seq_length - 2):
        predictions = model.predict([question_seq_padded, answer_seq_padded]).argmax(axis=-1)
        answer_seq_padded[0, i] = predictions[0, i]

        if predictions[0, i] == tokenizer.word_index['<end>']:
            break

    answer_seq = answer_seq_padded.flatten().tolist()
    answer = tokenizer.sequences_to_texts([answer_seq])[0]
    return answer


question = 'What is your name?'
answer = generate_answer(question)
print('Question:', question)
print('Answer:', answer)





