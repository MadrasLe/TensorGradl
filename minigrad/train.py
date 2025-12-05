import os
import requests
import numpy as np
import time
import gc
import pickle

# Importando seu framework poderoso
from minigrad.tensor import Tensor
from minigrad.gpt import GPT
from minigrad.loss import cross_entropy
from minigrad.optim import Adam
import minigrad.backend as backend

# -----------------------------------------------------------------------------
# 0. Configuração do Dispositivo (A Mágica do Backend)
# -----------------------------------------------------------------------------
print(f"{'='*40}")
print(f"🚂 INICIANDO TREINAMENTO NO DISPOSITIVO: {backend.backend_name.upper()}")
if backend.backend_name == 'cuda':
    print(f"   GPU Detectada! Prepare-se para voar.")
else:
    print(f"   Rodando na CPU. (Para velocidade, use MINIGRAD_DEVICE=cuda)")
print(f"{'='*40}\n")

# -----------------------------------------------------------------------------
# 1. Preparação dos Dados (Tiny Shakespeare)
# -----------------------------------------------------------------------------
data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
file_path = 'input.txt'

if not os.path.exists(file_path):
    print("📚 Baixando dataset Tiny Shakespeare...")
    with open(file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(file_path, 'r') as f:
    text = f.read()

# Tokenizador (nível de caractere)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Dados de treino/validação
data = np.array(encode(text), dtype=np.uint16)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Vocabulário: {vocab_size} caracteres")
print(f"Tamanho do Dataset: {len(text)} caracteres")

# -----------------------------------------------------------------------------
# 2. Hiperparâmetros e Modelo
# -----------------------------------------------------------------------------
# Se estiver na GPU, podemos ser mais ousados!
if backend.backend_name == 'cuda':
    block_size = 128    # Contexto maior
    batch_size = 64     # Batch maior
    max_iters = 500     # Mais iterações
else:
    block_size = 64     # Conservador na CPU
    batch_size = 32
    max_iters = 200

n_embd = 64
n_head = 4
n_layer = 4
learning_rate = 1e-3

print(f"\nConfiguração: Batch={batch_size}, Bloco={block_size}, Iterações={max_iters}")

# Inicializa o Modelo
# Graças ao seu tensor.py e gpt.py, ele já nasce na GPU se o backend for cuda!
model = GPT(vocab_size, n_embd, n_head, n_layer, max_len=block_size)
optimizer = Adam(model.parameters(), lr=learning_rate)

print(f"Número de parâmetros: {sum(p.data.size for p in model.parameters())}")

# -----------------------------------------------------------------------------
# 3. Funções Auxiliares
# -----------------------------------------------------------------------------
def get_batch(split):
    data_src = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data_src) - block_size, (batch_size,))
    x_np = np.stack([data_src[i:i+block_size] for i in ix])
    y_np = np.stack([data_src[i+1:i+block_size+1] for i in ix])
    
    # O Tensor() vai mover isso para a GPU automaticamente se necessário
    return Tensor(x_np), Tensor(y_np)

def generate(model, idx, max_new_tokens):
    # idx é (B, T)
    for _ in range(max_new_tokens):
        # Corta o contexto para caber no block_size
        idx_cond = idx[:, -block_size:]
        
        # Importante: Se idx for Tensor, usamos ele, se for numpy, convertemos
        if not isinstance(idx_cond, Tensor):
            idx_tensor = Tensor(idx_cond)
        else:
            idx_tensor = idx_cond

        # Forward (apenas inference, pegamos .data)
        # Precisamos do logits da última posição
        logits = model(idx_tensor)
        
        # Como estamos acessando .data, precisamos ver se é cupy ou numpy
        # O backend.to_cpu garante que trazemos de volta para processar com numpy aqui
        last_logits = backend.to_cpu(logits.data[:, -1, :])
        
        # Softmax numérico (Numpy)
        probs = np.exp(last_logits) / np.sum(np.exp(last_logits), axis=-1, keepdims=True)
        
        # Amostragem
        idx_next_list = []
        for i in range(idx.shape[0]):
            next_token = np.random.choice(len(probs[i]), p=probs[i])
            idx_next_list.append(next_token)
            
        idx_next = np.array(idx_next_list).reshape(-1, 1)
        
        # Concatenar (mantendo em numpy para facilitar o loop, convertemos pra Tensor só na entrada do modelo)
        if isinstance(idx, Tensor):
            idx_np = backend.to_cpu(idx.data)
        else:
            idx_np = idx
            
        idx = np.concatenate((idx_np, idx_next), axis=1)
        
    return idx

# -----------------------------------------------------------------------------
# 4. Loop de Treinamento
# -----------------------------------------------------------------------------
print("\nIniciando treinamento...")
t0 = time.time()

for iter in range(max_iters):
    # 1. Pegar Batch
    xb, yb = get_batch('train')
    
    # 2. Forward
    logits = model(xb)
    loss = cross_entropy(logits, yb)
    
    # 3. Backward & Update
    model.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 4. Log e Limpeza
    if iter % 10 == 0 or iter == max_iters - 1:
        # Trazemos o loss para CPU só para printar
        loss_val = backend.to_cpu(loss.data)
        print(f"Iter {iter}: loss {loss_val:.4f}")
        
        # O Faxineiro do Python (importante para não estourar RAM/VRAM)
        gc.collect()

dt = time.time() - t0
print(f"\nTreino finalizado em {dt:.2f} segundos.")

# -----------------------------------------------------------------------------
# 5. Geração e Salvamento
# -----------------------------------------------------------------------------
print("\nGerando texto...")
# Contexto inicial (token 0)
context = np.zeros((1, 1), dtype=int) 
output = generate(model, context, max_new_tokens=200)
print(f"\n---\n{decode(output[0])}\n---")

# Salvar Modelo (apenas os pesos, como lista de arrays numpy)
print("\nSalvando modelo...")
model_params = [backend.to_cpu(p.data) for p in model.parameters()]
with open('minigrad_gpt.pkl', 'wb') as f:
    pickle.dump(model_params, f)
print("Modelo salvo como 'minigrad_gpt.pkl'")
