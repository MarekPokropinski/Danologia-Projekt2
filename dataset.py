import tensorflow as tf
import csv


def load():
    with open('train.csv') as f:
        texts = []
        sentiments = []
        reader = csv.reader(f)
        # for line in f.readlines():
        #     values = line.split(',')
        #     text = ','.join(values[:-1])
        #     sentiment = values[-1]
        #     texts.append(text)
        #     sentiments.append(sentiment[:-1])
        for line in reader:
            texts.append(line[1])
            sentiments.append(line[-1])

    return texts, sentiments


def load_test():
    with open('test.csv') as f:
        texts = []
        sentiments = []
        reader = csv.reader(f)

        for line in reader:
            texts.append(line[1])
            sentiments.append(line[-1])

    return texts, sentiments


def getDataset(tokenizer, batch_size=64, test=False):
    def to_number(x):
        if x == 'neutral':
            return 0
        if x == 'positive':
            return 1
        if x == 'negative':
            return 2

    if test:
        texts, sentiments = load_test()
    else:
        texts, sentiments = load()

    sentiments = [to_number(x) for x in sentiments]

    # tokenizer.fit_on_texts(texts)
    vocab_size = tokenizer.num_words

    tensor = tokenizer.texts_to_sequences(texts)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor, padding='post')
    y_tensor = tf.keras.utils.to_categorical(sentiments, 3)

    BUFFER_SIZE = len(tensor)
    steps_per_epoch = len(tensor)//batch_size

    # train_size = int(BUFFER_SIZE *(1-validation_split))

    dataset = tf.data.Dataset.from_tensor_slices(
        (tensor, y_tensor)).shuffle(BUFFER_SIZE)
    # train_dataset = dataset.take(train_size)
    # test_dataset = dataset.skip(train_size)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    # test_dataset= test_dataset.batch(batch_size, drop_remainder=True)

    return dataset
