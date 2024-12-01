import re
import numpy as np
import pandas as pd
import time
import seaborn as sns

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import torch.nn as nn
from models import UniLSTM, QRNN, UniGRU, BiLSTM


def clean_data(text):
    text = str(text).lower()
    text = text.split(', ')
    text = [x.replace(' ', '') for x in text]
    text = ' '.join(text)
    text = re.sub(r'[^\w\s]', '', text)

    return text


def make_wordcloud(words):
    words = ' '.join(word for word in words)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(words)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()


def predict(model, input_sequence):
    # Set the model to evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        prediction = model(input_sequence, torch.from_numpy(np.zeros(1)))  # No target during prediction
    pred = np.mean(prediction.squeeze().cpu().numpy())
    return pred


def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length.
    '''

    # getting the correct rows x cols shape
    print(len(reviews_ints))
    if not isinstance(reviews_ints[0], int):
        features = np.zeros((len(reviews_ints), seq_length), dtype=int)
        # for each review, I grab that review and
        for i, row in enumerate(reviews_ints):
            features[i, -len(row):] = np.array(row)[:seq_length]

    else:
        features = np.array(reviews_ints)[:seq_length]
        if len(features) < seq_length:
            features = np.pad(features, (0, seq_length - len(features)), mode='constant', constant_values=0)
    return features


# Загрузка и очистка данных
data = pd.read_excel("games_data.xlsx")
data = data[data['Reviews Total'] >= 100][:14500]

# Распределение оценок
sns.set_style('whitegrid')
sns.histplot(data['Reviews Score'], kde=True, color='green', bins=30)
plt.show()

# Пример игры с высокой оценкой
# example_tags = [
#     'JRPG, RPG, Anime, Party-Based RPG, Turn-Based Strategy, Adventure, Strategy, Dating Sim, Story Rich, Dark, Strategy RPG, Emotional, Visual Novel, Turn-Based Combat, Colorful, Great Soundtrack, Supernatural, Mystery, Detective, Multiple Endings']
# example_title = ['Persona 3 Reload']
# example_score = np.float64(0.9235)

# Пример игры с невысокой оценкой
example_tags = ['Colorful, FPS, Shooter, Hero Shooter, Action, Funny, Multiplayer, First-Person, PvP, Team-Based, Sci-fi, Parody, Perma Death, Crime, Gambling, Dark Fantasy, Psychological, Post-apocalyptic, Heist, Zombies']
example_title = ['Concord']
example_score = np.float64(0.6645)

data['Tags'] = data['Tags'].apply(lambda x: clean_data(x))
data['Title'] = data['Title'].apply(lambda x: clean_data(x))
example_tags = [clean_data(example_tags)]

# Полный список тегов
tags = list()
for game in data['Tags']:
    for tag in game.split(' '):
        tags.append(tag)

# Подсчет уникальных тегов
counts = Counter(tags)
unique_tags = sorted(counts, key=counts.get, reverse=True)
make_wordcloud(unique_tags)

tag_data = list(counts.items())
tag_df = pd.DataFrame(tag_data, columns=['Tag', 'Count'])
tag_df = tag_df.sort_values(by='Count', ascending=False)

# Распределение тегов
sns.set_style('whitegrid')
sns.barplot(y='Tag', x='Count', data=tag_df.head(20), palette='viridis')
plt.tight_layout()
plt.show()

# Преобразование тегов в числовые признаки
tags_to_int = {word: ii for ii, word in enumerate(unique_tags, 1)}
X_tags = []
for game_tags in data['Tags']:
    X_tags.append([tags_to_int[tag] for tag in game_tags.split()])

test_features = [tags_to_int[tag] for tag in example_tags[0].split()]

X_tags = pad_features(X_tags, seq_length=20)
test_features = pad_features(test_features, 20)

# Нормализация
# scaler = MinMaxScaler(feature_range=(0, 1))
# y_scores = scaler.fit_transform(data['Reviews Score'].values.reshape(-1, 1))
y_scores = data['Reviews Score'].values.reshape(-1, 1)
example_score = example_score.reshape(-1, 1)

# Разделение на обучающую, валидационную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_tags, y_scores, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Преобразование в тензоры
X_tags_tensor = torch.tensor(X_tags, dtype=torch.int32)
X_train_tensor = torch.tensor(X_train, dtype=torch.int32)
X_val_tensor = torch.tensor(X_val, dtype=torch.int32)
X_test_tensor = torch.tensor(X_test, dtype=torch.int32)
y_scores_tensor = torch.tensor(y_scores, dtype=torch.int32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# DataLoader для батчевой обработки данных
train_data = TensorDataset(X_train_tensor, y_train_tensor)
valid_data = TensorDataset(X_val_tensor, y_val_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
overall_data = TensorDataset(X_tags_tensor, y_scores_tensor)
test_sample = torch.from_numpy(test_features)

batch_size = 50
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
overall_loader = DataLoader(test_data, batch_size=batch_size)

# Параметры модели
vocab_size = len(unique_tags) + 1
embed_dims = 300
num_units = 256
dropout = 0.5
seq_len = 20
epochs = 10
kernel_size = 2
layers = 2
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Инициализация моделей
uniGRU = UniGRU(vocab_size, embed_dims, num_units, 1, dropout, seq_len, batch_size, layers).to(device)
_QRNN = QRNN(vocab_size, embed_dims, num_units, kernel_size, batch_size, seq_len, layers, device, dropout=0.3).to(device)
uniLSTM = UniLSTM(vocab_size, embed_dims, num_units, 1, dropout, seq_len, batch_size, 1).to(device)
biLSTM = BiLSTM(vocab_size, embed_dims, num_units, 1, dropout, seq_len, batch_size, layers).to(device)

# Определяем функцию потерь и оптимизатор
criterion = torch.nn.MSELoss().to(device)
counter = 0
acc = []
valacc = []

# Загрузка обученных моделей
# uniGRU = torch.load('models/uniGRU-model.pt')
# #_QRNN = torch.load('steam-model-10.pt')
# _QRNN = torch.load('models/qrnn-model.pt')
# uniLSTM = torch.load('models/uniLSTM-model.pt')
# biLSTM = torch.load('models/biLSTM-model.pt')
model_names = ['uniGRU', 'QRNN', 'uniLSTM', 'biLSTM']
model_list = [uniGRU, _QRNN, uniLSTM, biLSTM]
# model_list = [uniLSTM]
mse_train_loss = [[], [], [], []]
mse_val_loss = [[], [], [], []]
mse_test_loss = [[], [], [], []]
predicted_score = [[], [], [], []]

i = 0

print('uniGRU: ', predicted_score[0])
print('QRNN: ', predicted_score[1])
print('uniLSTM: ', predicted_score[2])
print('biLSTM: ', predicted_score[3])
print('Exact score: ', float(example_score))

total_time = 0

for model in model_list:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    train_loss_acc = []
    val_loss_acc = []
    for e in range(epochs):
        start_time = time.time()
        for inputs, labels in train_loader:
            # inputs, labels = inputs.cuda(), labels.cuda()
            model.zero_grad()
            logits = model(inputs, labels)
            loss = criterion(logits, labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if counter % 10 == 0:
                print("Epoch: {}/{}".format(e, epochs),
                      "\tIteration: {}".format(counter),
                      "\tTrain MSE Loss: {:.3f}".format(loss.item()))
            mse_train_loss[i].append(loss.item())
            if counter % 399 == 0:
                with torch.no_grad():
                    model.eval()
                    val_loss = []
                    for inputs, labels in valid_loader:
                        inputs_val, labels_val = inputs, labels
                        logits_val = model(inputs_val, labels_val)
                        loss_val = criterion(logits_val, labels_val.float())
                        val_loss.append(loss_val.item())
                    model.train()
            counter += 1
        mse_val_loss[i].append(np.mean(val_loss))
        time_epoch = time.time() - start_time
        total_time += time_epoch
        print("Time to train epoch: {0} s".format(time.time() - start_time))

        with torch.no_grad():
            test_loss = []
            model.eval()
            for inputs, labels in test_loader:
                input_test, labels_test = inputs, labels
                logits_test = model(input_test, labels_test)
                loss_test = criterion(logits_test, labels_test.float())
                test_loss.append(loss_test.item())

        mse_test_loss[i].append(np.mean(test_loss))
        predicted_score[i].append(predict(model, test_sample))

    print('Total time to train model: {0} s'.format(total_time))

    # if i == 0:
    #     torch.save(model, 'uniGRU-model.pt')
    # elif i == 1:
    #     torch.save(model, 'qrnn-model.pt')
    # elif i == 2:
    #     torch.save(model, 'uniLSTM-model.pt')
    # elif i == 3:
    #     torch.save(model, 'biLSTM-model.pt')

    i += 1


def plot_graphs(data, name, xLabel='Epoch', ylabel='MSE'):
    for i in range(len(data)):
        plt.plot(data[i], label=model_names[i])
    plt.xlabel(xLabel)
    plt.ylabel(ylabel)
    if name == 'Predicted Score':
        plt.axhline(y=example_score, color='r', linestyle='-')
    plt.title(name)
    plt.legend()
    plt.show()


plot_graphs(mse_train_loss, 'Train', 'Iteration')
plot_graphs(mse_val_loss, 'Validation')
plot_graphs(mse_test_loss, 'Test')
plot_graphs(predicted_score, 'Predicted Score', ylabel='Score')
