import os
import random
import spacy
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data

from TextClassifier import TextClassifier

SEED = 1234
torch.manual_seed(SEED)

TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
LABEL = data.LabelField(dtype=torch.float, batch_first=True)

fields = [('label', LABEL), ('text', TEXT)]
# fields = [(None, None), ('text', TEXT), ('label', LABEL)]

# Loading dataset
training_data = data.TabularDataset(path='output/balanced_20201008235520.csv',
                                    format='csv',
                                    fields=fields,
                                    skip_header=True)

# Split dataset into training and testing datasets
train_data, valid_data = training_data.split(split_ratio=0.7, random_state=random.seed(SEED))

# Initialize glove embeddings
TEXT.build_vocab(train_data, min_freq=3, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# No. of unique tokens in text
print("Size of TEXT vocabulary:", len(TEXT.vocab))

# No. of unique tokens in label
print("Size of LABEL vocabulary:", len(LABEL.vocab))

# Commonly used words
print(TEXT.vocab.freqs.most_common(10))

# Word dictionary
tmp = TEXT.vocab.stoi
print(TEXT.vocab.stoi)

# Cuda algorithms, and checking if whether cuda is available
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64

# Prepare batches for training the model
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device)

# define hyper-parameters
size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 1
num_layers = 2
bi_direction = True
dropout = 0.2

# instantiate the model
model = TextClassifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers,
                       bidirectional=True, dropout=dropout)

# architecture
print(model)


# No. of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

# Initialize the pretrained embedding
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

print(pretrained_embeddings.shape)

# define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()


# define metric
def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# push to cuda if available
model = model.to(device)
criterion = criterion.to(device)


def train(model, iterator, optimizer, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # set the model in training phase
    model.train()

    for batch in iterator:
        # resets the gradients after every batch
        optimizer.zero_grad()

        # retrieve text and no. of words
        text, text_lengths = batch.text

        # convert to 1D tensor
        predictions = model(text, text_lengths).squeeze()

        # compute the loss
        loss = criterion(predictions, batch.label)

        # compute the binary accuracy
        acc = binary_accuracy(predictions, batch.label)

        # backpropage the loss and compute the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # deactivating dropout layers
    model.eval()

    # deactivates autograd
    with torch.no_grad():
        for batch in iterator:
            # retrieve text and no. of words
            text, text_lengths = batch.text

            # convert to 1d tensor
            predictions = model(text, text_lengths).squeeze()

            # compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if os.path.isfile('saved_weights.pt'):
    # load weights
    path = 'saved_weights.pt'
    model.load_state_dict(torch.load(path));
    model.eval()
else:
    N_EPOCHS = 500
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        # train the model
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

        # evaluate the model
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

# inference
nlp = spacy.load('en')


def predict(input_model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  # tokenize the sentence
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]  # convert to integer sequence
    length = [len(indexed)]  # compute no. of words
    tensor = torch.LongTensor(indexed).to(device)  # convert to tensor
    tensor = tensor.unsqueeze(1).T  # reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)  # convert to tensor
    prediction = input_model(tensor, length_tensor)  # prediction
    return prediction.item()


# make predictions
print(predict(model, "did a rooster stand on your head"))

# insincere question
print(predict(model, "i knew there was a double letter there it is so okay"))
