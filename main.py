import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
f = open("4chan_dataset.txt")
text = f.read()
f.close()
print(text)
def wrap(s, w):
    return [s[i:i + w] for i in range(0, len(s), w)]

samples = wrap(text, 1024*8)

# Create a vocabulary and data loaders
char_counts = Counter(" ".join(samples))
vocab = sorted(char_counts, key=char_counts.get, reverse=True)
vocab_size = len(vocab)
char_to_int = {char: i for i, char in enumerate(vocab)}
int_to_char = {i: char for char, i in char_to_int.items()}
print(f"Char counts: {char_counts}")
print(f"Vocabulary: {vocab}")
print(f"Character to integer mapping: {char_to_int}")
print(f"Integer to character mapping: {int_to_char}")

class TextDataset(Dataset):
    def __init__(self, samples, char_to_int):
        self.samples = samples
        self.char_to_int = char_to_int
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        chars = torch.LongTensor([self.char_to_int[char] for char in sample])
        input_seq = chars[:-1]
        target_seq = chars[1:]
        return input_seq, target_seq
        
batch_size = 1
dataset = TextDataset(samples, char_to_int)
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True,
)

iterator=iter(dataloader)
inputs, targets = next(iterator)
print(inputs, targets)
# Number of samples in one batch.
num_inputs = len(inputs)
num_targets = len(targets)
print(f"Num samples in one batch: {num_inputs} inputs, {num_targets} targets")
for i in range(num_inputs):
    num_input_chars = len(inputs[i])
    num_target_chars = len(targets[i])
    print(num_input_chars, num_target_chars)
    
class CharTextGenerationLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_layers
    ):
        super(CharTextGenerationLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    def forward(self, x, hidden=None):
        if hidden == None:
            hidden = self.init_hidden(x.shape[0])
        x = self.embedding(x)
        out, (h_n, c_n) = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        return out, (h_n, c_n)
    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0, c0
        
# Training Setup
embedding_dim = 512
hidden_size = 128
num_layers = 3
learning_rate = 0.01
epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CharTextGenerationLSTM(
    vocab_size, 
    embedding_dim, 
    hidden_size, 
    num_layers
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.load_state_dict(torch.load("./my-model/model.json", weights_only=True))

def train(model, epochs, dataloader, criterion):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for input_seq, target_seq in dataloader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            outputs, _ = model(input_seq)
            loss = criterion(outputs, target_seq.view(-1))
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().numpy()
            torch.save(model.state_dict(), "my-model")
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} loss: {epoch_loss:.3f}")
train(model, epochs, dataloader, criterion)
	
def generate_text_one_char(model, start_char, length):
    model.eval()
    generated_text = start_char
    input_seq = torch.LongTensor([char_to_int[start_char]]).unsqueeze(0).to(device)
    h, c = model.init_hidden(1)
    for _ in range(length):
        with torch.no_grad():
            output, (h, c) = model(input_seq, (h, c))
        # Greedy approach.
        next_token = output.argmax(1).item()
        generated_text += int_to_char[next_token]
        input_seq = torch.LongTensor([next_token]).unsqueeze(0).to(device)
    return generated_text
    
def generate_text_multiple_char(model, start_string, length):
    model.eval()
    generated_text = start_string
    input_seq = torch.LongTensor([char_to_int[char] for char in start_string]).unsqueeze(0).to(device)
    h, c = model.init_hidden(1)
    for _ in range(length):
        output, (h, c) = model(input_seq, (h, c))
        next_token = output.argmax(1)[-1].item()  # Get the prediction for the last character
        generated_text += int_to_char[next_token]
        # Update input_seq to include the predicted character
        input_seq = torch.cat((input_seq, torch.LongTensor([[next_token]]).to(device)), 1)[:, 1:]
    return generated_text

print("Generated Text:", generate_text_multiple_char(model, start_string="He", length=100))
