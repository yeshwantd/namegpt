import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class Head(nn.Module):
    def __init__(self, head_size, embedding_dim, block_size, dropout_rate):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Formula from the Attention all you need paper is:
        # Attention Weights = Softmax(Q @ K.T / (d_k ** 0.5)) @ V
        # Where Q, K, V are matrices of shape (B, T, head_size)
        B, T, C = x.shape # T = block_size, C = embedding_dim
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # we need the [:T, :T] slice because during inference, the sequence length can be less than block_size
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embedding_dim, block_size, dropout_rate):
        super().__init__()
        # each head is batch_size B x block_size T x head_size
        self.heads = nn.ModuleList([Head(head_size, embedding_dim, block_size, dropout_rate) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, head_size * num_heads)
        out = self.proj(out) # (B, T, C) - project back to embedding dim
        out = self.dropout(out) # (B, T, C)
        return out

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embedding_dim, head_size, num_heads, block_size, dropout_rate):
        super().__init__()
        self.sa_heads = MultHeadAttention(num_heads, head_size//num_heads, embedding_dim, block_size, dropout_rate) # (B, T, head_size)
        self.ffwd = FeedForward(embedding_dim, dropout_rate) # (B, T, C)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Apply layernorm 1 to the input of the heads (before attention)  and layernorm 2 to input of FF layer (output of attention)
        x = x + self.sa_heads(self.ln1(x)) # first add a residual connection to the output of the attention. Size is (B, T, C)
        x = x + self.ffwd(self.ln2(x)) # then add another residual connection from output of head to the output of the FF layer. Size is (B, T, C)
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size, head_size, num_heads, num_layers, embedding_dim, dropout_rate, device):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        # self.sa_heads = MultHeadAttention(num_heads, head_size//num_heads, embedding_dim, block_size) # (B, T, head_size)
        # self.ffwd = FeedForward(embedding_dim) # (B, T, C)
        # Sequential doesn't accept a list, so we need to use * to decompose the list
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, head_size, num_heads, block_size, dropout_rate) for _ in range(num_layers)]
        )
        # self.blocks = nn.Sequential(
        #     Block(embedding_dim, head_size, num_heads, block_size),
        #     Block(embedding_dim, head_size, num_heads, block_size),
        #     Block(embedding_dim, head_size, num_heads, block_size),
        #     nn.LayerNorm(embedding_dim),
        # )
        self.ln_final = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size) # (B, T, C)
        
    def forward(self, idx, targets=None):
        # idx is (batch_size, block_size) tensor of integers
        # targets is (batch_size, block_size) tensor of integers
        # batch_size, block_size, vocab_size = B (batch), T (time), C (channel)
        B, T = idx.shape # batch_size, block_size
        tok_emb = self.token_embedding_table(idx) # (batch_size, block_size, num_embeddings)
        pos_emb = self.position_embedding_table(torch.arange(T).to(self.device)) # (block_size, num_embeddings)
        x = tok_emb + pos_emb # (batch_size, block_size, num_embeddings)
        # x = self.sa_heads(x) # (batch_size, block_size, head_size)
        # x = self.ffwd(x) # (batch_size, block_size, C)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x) # (batch_size, block_size, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (batch_size, block_size) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cropped = idx[:, -self.block_size:] # B, T (cropping ensures that we are only looking at the last block_size tokens)
            # get the predictions
            logits, _ = self(idx_cropped) # gives (B, T, C) since we don't specify targets
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C) taking the last time step T
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def get_data():
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for i, c in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        
        data = torch.tensor(encode(text), dtype=torch.long)

        n = int(len(data) * 0.9)
        train_data = data[:n]
        val_data = data[n:]
    return train_data, val_data, vocab_size, encode, decode

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def train():
    train_data, val_data, vocab_size, encode, decode = get_data()
    
    block_size = 256
    batch_size = 64
    n_embeddings = 64*4
    head_size = 12*6
    num_heads = 6
    num_layers = 6
    dropout_rate = 0.2
    learning_rate = 3e-4

    max_iters = 5000
    eval_iters = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    m = BigramLanguageModel(vocab_size, block_size, head_size, num_heads, num_layers, n_embeddings, dropout_rate, device)
    m.to('device')
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # sample a batch of data
        xb, yb = get_batch(train_data, block_size, batch_size)
        xb = xb.to(device)
        yb = yb.to(device)
        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 1000 == 0 or iter == max_iters - 1:
            # calculate evaluation loss on validation set
            m.eval()
            val_loss = 0
            with torch.no_grad():
                for _ in range(eval_iters):
                    xb, yb = get_batch(val_data, block_size, batch_size)
                    logits, loss = m(xb, yb)
                    val_loss += loss.item()
            m.train()
            print(f"iter {iter} | train loss {loss.item():.4f} | val loss {val_loss/1000:.4f}")
            

    return m, decode

if __name__ == '__main__':

    m, decode = train()
    print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
        
        
        