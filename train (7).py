import numpy as np
import os
import requests
from minigrad.tensor import Tensor
from minigrad.gpt import GPT
from minigrad.loss import cross_entropy
from minigrad.optim import Adam

# -----------------------------------------------------------------------------
# 1. Preparação dos Dados (Tiny Shakespeare)
# -----------------------------------------------------------------------------
data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
file_path = 'input.txt'

if not os.path.exists(file_path):
    print("Baixando dataset Tiny Shakespeare...")
    with open(file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(file_path, 'r') as f:
    text = f.read()

# Tokenizador simples (nível de caractere)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train/Val split
data = np.array(encode(text), dtype=np.uint16)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Vocab size: {vocab_size}")
print(f"Dataset size: {len(text)} chars")

# -----------------------------------------------------------------------------
# 2. Configuração do Modelo e Hiperparâmetros
# -----------------------------------------------------------------------------
# Pequeno para rodar rápido na CPU
block_size = 64      # Context length (T)
batch_size = 32      # Batch dimensions (B)
n_embd = 64          # Embedding dimension
n_head = 4           # Cabeças de atenção
n_layer = 3          # Profundidade
learning_rate = 1e-3
max_iters = 500      # Poucas iterações só para testar "vida"
eval_interval = 100

model = GPT(vocab_size, n_embd, n_head, n_layer, max_len=block_size)
optimizer = Adam(model.parameters(), lr=learning_rate)

print(f"Número de parâmetros: {sum(p.data.size for p in model.parameters())}")

# -----------------------------------------------------------------------------
# 3. Funções Auxiliares
# -----------------------------------------------------------------------------
def get_batch(split):
    data_src = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data_src) - block_size, (batch_size,))
    x = np.stack([data_src[i:i+block_size] for i in ix])
    y = np.stack([data_src[i+1:i+block_size+1] for i in ix])
    return Tensor(x), Tensor(y)

def generate(model, idx, max_new_tokens):
    # idx é (B, T) array de índices no contexto atual
    for _ in range(max_new_tokens):
        # Cortar idx para os últimos block_size tokens
        idx_cond = idx[:, -block_size:]
        idx_tensor = Tensor(idx_cond)
        
        # Forward
        logits = model(idx_tensor)
        
        # Pegar logits do último passo apenas: (B, 1, vocab_size)
        logits = logits.data[:, -1, :] 
        
        # Softmax simples para obter probabilidades
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        
        # Amostrar do output (para 1 batch)
        # Nota: Numpy choice não suporta batches nativamente de jeito fácil, 
        # então vamos fazer loop ou pegar só o primeiro se B=1
        idx_next = []
        for i in range(idx.shape[0]):
            next_token = np.random.choice(len(probs[i]), p=probs[i])
            idx_next.append(next_token)
            
        idx_next = np.array(idx_next).reshape(-1, 1)
        idx = np.concatenate((idx, idx_next), axis=1)
        
    return idx

# -----------------------------------------------------------------------------
# 4. Loop de Treinamento
# -----------------------------------------------------------------------------
print("\nIniciando treinamento... (Paciência, é Python puro na CPU!)")

for iter in range(max_iters):
    # 1. Pegar batch
    xb, yb = get_batch('train')
    
    # 2. Forward
    logits = model(xb)
    
    # 3. Loss
    loss = cross_entropy(logits, yb)
    
    # 4. Zero Grad
    model.zero_grad() # Importante!
    
    # 5. Backward
    loss.backward()
    
    # 6. Update
    optimizer.step()
    
    if iter % 10 == 0:
        print(f"Iter {iter}: loss {loss.data:.4f}")

# -----------------------------------------------------------------------------
# 5. Teste de Geração
# -----------------------------------------------------------------------------
print("\nTreinamento finalizado! Gerando texto...\n")

context = np.zeros((1, 1), dtype=int) # Começa com token 0
generated_indices = generate(model, context, max_new_tokens=200)
print(decode(generated_indices[0]))