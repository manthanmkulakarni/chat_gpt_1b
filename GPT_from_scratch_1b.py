# %%
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# %%
questions_df = pd.read_csv("/kaggle/input/pythonquestions/Questions.csv",encoding='latin-1')
answers_df = pd.read_csv("/kaggle/input/pythonquestions/Answers.csv",encoding='latin-1')


# %%
answers_df.head()

# %%
import re
# as per recommendation from @freylis, compile once only
CLEANR = re.compile('<.*?>') 

def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext

# %%
pid_map = {}
for i in tqdm(range(len(questions_df))):
    pid_map[questions_df.iloc[i]["Id"]] = [i,[]]
    
for i in tqdm(range(len(answers_df))):
    if answers_df.iloc[i]["ParentId"] in pid_map:
        pid_map[answers_df.iloc[i]["ParentId"]][1].append(i)
        
questions = []
answers = []

for pid in tqdm(pid_map.keys()):
    if len(pid_map[pid][1]) > 0:
        questions.append(cleanhtml(questions_df.iloc[pid_map[pid][0]]["Title"]))
        answer = []
        for i in pid_map[pid][1]:
            answer.append(cleanhtml(answers_df.iloc[i]["Body"]))
        answers.append("\n\n".join(answer))


# %%


# %%


# %%
# hyperparameters
max_length = 512
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 512 # what is the maximum context length for predictions?
max_iters = 50
eval_interval = 25
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 5
n_embd = 2048
n_head = 8
n_layer = 20
dropout = 1e-6
# ------------

# %%


torch.manual_seed(1337)

# data read from above cells

# build vocab
chars = set()
for i in tqdm(range(len(answers))):
    for c in answers[i]:
        chars.add(c)
        
for i in tqdm(range(len(questions))):
    for c in questions[i]:
        chars.add(c)
        


chars = sorted(list(chars))
vocab_size = len(chars)

def padTokens(data, max_length):
    if len(data) > max_length:
        return data[:max_length]
    else:
        diff = max_length - len(data)
        data.extend([0]*diff)
        return data

# create a mapping from characters to integers
stoi = {}
itos = {}
for i in range(len(chars)):
    stoi[chars[i]] = i+1
    itos[i+1] = chars[i]
    
stoi["<pad>"] = 0
itos[0]="<pad>"
encode_sentence = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode_sentence = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data_x = torch.tensor([padTokens(encode_sentence(answers[i]),max_length) for i in tqdm(range(len(answers)))], dtype=torch.long)
data_y = torch.tensor([padTokens(encode_sentence(questions[i]),max_length) for i in tqdm(range(len(questions)))], dtype=torch.long)

data_x.to(device)
data_y.to(device)
n = int(0.9*len(data_x)) # first 90% will be train, rest val
train_data_x = data_x[:n]
val_data_x = data_x[n:]

train_data_y = data_x[:n]
val_data_y = data_x[n:]



# %%
import pickle as pkl

with open("/kaggle/working/tokenizer_tokens.pkl","wb") as f:
    pkl.dump(chars,f)

# %%


# %%

    
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data_x = train_data_x if split == 'train' else val_data_x
    data_y = train_data_y if split == 'train' else val_data_y
    idxs = torch.randint(0,len(data_x), (batch_size,))
    x = torch.stack([data_x[i].to(device) for i in idxs])
    y = torch.stack([data_y[i].to(device) for i in idxs])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx




# %%
len(data_x)

# %%
model = BigramLanguageModel()
m.load_state_dict(torch.load('/kaggle/input/model-weights/model_500e.pt'))
m = model.half().to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')




# %%
# create a PyTorch optimizer
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

# %%
for iter in range(1000):
    print(f"Interation {iter}")
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if iter % 100 == 0 and iter != 0:
        torch.save(m.state_dict(),'/kaggle/working/model.pt')
        print("Saved model ...")
        # generate from the model
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(decode_sentence(m.generate(context, max_new_tokens=512)[0].tolist()))

# %%
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode_sentence(m.generate(context, max_new_tokens=512)[0].tolist()))

# %%
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode_sentence(m.generate(context, max_new_tokens=512)[0].tolist()))


# %%



