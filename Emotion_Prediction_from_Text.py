# 2. Emocijų atpažinimas iš teksto (NLP)
#    Šis projektas susijęs su teksto emocijų atpažinimu, kuris gali būti naudojamas socialinės žiniasklaidos įrašų, klientų atsiliepimų arpokalbių analizėje.
#    Tikslas yra sukurti modelį, kuris nustatytų ir klasifikuotų įvairias emocijas tekste, pvz., džiaugsmą, liūdesį, pyktį, baimę ir kt.
# Technologijos: Python, TensorFlow/Keras, NLTK, scikit-learn
# Darbo etapai:
# Duomenų rinkimas: Naudojant duomenų rinkinius, tokius kaip "Emotion Dataset" iš Kaggle.
# Duomenų apdorojimas: Teksto valymas, tokenizavimas..
# Modelių kūrimas ir mokymas: CNN, RNN ar LSTM naudojimas emocijų atpažinimui.
# Vertinimas ir tobulinimas: Modelio efektyvumo įvertinimas naudojant matavimo rodiklius, pavyzdžiui, tikslumą, ir modelio tobulinimas.

import tensorflow as tf
import numpy as np
from keras.src.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GRU
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasClassifier
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import EarlyStopping

def load_sentences():
    # Duomenų surinkimas ir apdorojimas

    # Nuskaitymas
    df = pd.read_csv('Emotion_classify_Data.csv')

    # Emocijos ir jų perkodavimas
    label_encoder = LabelEncoder()
    df['Emotion_ENCOD'] = label_encoder.fit_transform(df['Emotion'])
    labels = np.array(to_categorical(df['Emotion_ENCOD']))

    emociniu_kategorijos_ir_statistika = df.groupby(['Emotion_ENCOD', 'Emotion'])[
        ['Emotion', 'Emotion_ENCOD']].size().reset_index().rename(columns={0: 'Count'})
    print('Categories of emotional sentences and their statistics:')
    print(emociniu_kategorijos_ir_statistika)

    unique_emotion_names = emociniu_kategorijos_ir_statistika['Emotion'].to_list()

    # Žodynas
    tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['Comment'])

    # Sakinių perkodavimas
    sequences = tokenizer.texts_to_sequences(df['Comment'])
    padded_sequences = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')

    return tokenizer, padded_sequences, labels, unique_emotion_names


def create_model(optimizer='adam'):
    model = Sequential([
        # būtent dėl Embedding sluoksnio neveikia kvietimas per RandomizedSearchCV,
        # nors RandomizedSearchCV veikia išmetus Embedding ir LSTM
        # Embedding veikia modelį tiesiogiai paleidus
        Embedding(input_dim=1000, output_dim=10, input_length=20),
        GRU(60),
        Dense(3, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def decode_emotion_name(unique_emotions, prediction):
    # Skaitinių rezultatų dekodavimas į emocijų pavadinimus
    max_idx = np.argmax(prediction)
    emotion_name = unique_emotions[max_idx]
    return emotion_name


def test_manually(model, unique_emotion_names):
    print(f'\nIšbandykite patys!')
    while True:
        test_tekstas = input(f'Iveskite teksta anglu kalba '
                             f'(noredami baigti, iveskite "exit" arba nieko nevede spauskite Enter):\n> ')
        if test_tekstas in ['exit', '']:
            break
        test_sequence = tokenizer.texts_to_sequences([test_tekstas])
        test_padded = pad_sequences(test_sequence, maxlen=20, padding='post', truncating='post')

        predictions = model.predict(test_padded)
        decoded_predictions = decode_emotion_name(unique_emotion_names, predictions)
        print(f'Priskirtas emocijos pavadinimas: {decoded_predictions}')


def train_single_model(padded_seq_train, labels_train, epochs=100, batch_size=20):
    model = create_model()
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=1,
        mode="min",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=5,
    )
    log = model.fit(padded_seq_train, labels_train,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[early_stopping],
              validation_split=0.2)
    loss_df = pd.DataFrame(model.history.history)
    loss_df['accuracy'].plot(label='accuracy')
    loss_df['val_accuracy'].plot(label='val_accuracy')
    plt.title('Modelio tikslumo kaita')
    plt.xlabel('Ciklas (epoch)')
    plt.ylabel('Tikslumas (accuracy)')
    plt.legend()
    plt.show()
    loss_df['loss'].plot(label='loss')
    loss_df['val_loss'].plot(label='val_loss')
    plt.title('Modelio nuostolių kaita')
    plt.xlabel('Ciklas (epoch)')
    plt.ylabel('Nuostoliai (loss)')
    plt.legend()
    plt.show()
    return model, log


def run_grid_search(padded_seq_train, labels_train):
    model = KerasClassifier(model=create_model, verbose=0)

    param_grid = {
        'batch_size': sp_randint(10, 50),
        'epochs': sp_randint(10, 50),
        'optimizer': ['RMSprop', 'Adam']
    }

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_jobs=-1,
        cv=3,
        n_iter=10,
        random_state=42)

    # # ValueError: Sequential model 'sequential_10' has no defined outputs yet.
    random_result = random_search.fit(padded_seq_train, labels_train)

    print(f"Best: {random_result.best_score_} using {random_result}")
    print('Geriausias rezultatas %.3f naudojant %s' % (random_result.best_score_, random_result.best_params_))

    return random_result.best_params_


# Įkeliame duomenų rinkinį, užkoduojame emocijas ir paruošiame duomenis modeliui
tokenizer, padded_sequences, labels, unique_emotion_names = load_sentences()

# Papildomai padalinti į apmokymo ir testavimo imtis
padded_seq_train, padded_seq_test, labels_train, labels_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42)

# Hiperparametrų optimizavimas - neveikia:
# ValueError: Sequential model 'sequential_10' has no defined outputs yet.
# # https://github.com/mrdbourke/tensorflow-deep-learning/discussions/256
# # https://github.com/tensorflow/tensorflow/releases/tag/v2.7.0 Breaking Changes
# # The methods Model.fit(), Model.predict(), and Model.evaluate() will no longer uprank input data of shape (batch_size,) to become (batch_size, 1).
# best_params = run_grid_search(padded_seq_train, labels_train)


# Treniruotas modelis
model, log = train_single_model(padded_seq_train, labels_train, epochs=100, batch_size=20)

# Modelio įvertinimas
#model_predict = model.predict(padded_seq_test)
test_loss, test_accuracy = model.evaluate(padded_seq_test, labels_test)
print(f'Modelio ivertinimas su apmokyme nenaudotais duomenimis:\n'
      f' - nuostoliai: {test_loss:.3f}, \n '
      f'- tikslumas: {test_accuracy:.3f} ')


# Naudotojo prasyti ivesti
test_manually(model, unique_emotion_names)
