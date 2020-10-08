import torch
from torchtext import data

# Assuring reproducibility
torch.manual_seed(1234)

TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
LABEL = data.LabelField(dtype=torch.float, batch_first=True)

fields = [('Role', LABEL), ('Text', TEXT)]

# Loading dataset
training_data = data.TabularDataset(path='output/balanced_20201008010117.csv',
                                    format='csv',
                                    fields=fields,
                                    skip_header=True)

print(vars(training_data.examples[0]))
