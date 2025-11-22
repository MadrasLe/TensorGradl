# 🧠 minigrad

> *A tiny, handcrafted deep learning library that thinks it's PyTorch.*

**minigrad** é uma biblioteca de Deep Learning construída do zero em Python e NumPy, criada para fins educacionais. Ela implementa um motor de **Autograd** (diferenciação automática) completo e uma arquitetura **GPT-2** capaz de aprender padrões de texto.

Se você quer entender o que acontece "por baixo do capô" de frameworks como PyTorch ou TensorFlow, você está no lugar certo.

## ✨ O que tem dentro?

Apesar de pequeno, o `minigrad` é poderoso:

*   **Autograd Real**: Cálculo automático de gradientes via *backpropagation* (grafo dinâmico).
*   **Tensores**: Suporte a broadcasting, transposição, reshape, slicing avançado e operações matriciais.
*   **Camadas (Layers)**: `Linear`, `Embedding` (otimizado), `LayerNorm`, `ReLU`, `GELU`, `MultiHeadAttention`.
*   **Otimizadores**: `SGD` e `Adam` (com correção de viés).
*   **GPT-2**: Implementação completa do Transformer Decoder com **Weight Tying** (pesos compartilhados entre embedding e saída).
*   **Loss**: `CrossEntropy` com estabilidade numérica básica.

## 📦 Instalação

Você só precisa do NumPy. Sério.

```bash
pip install numpy
```

## 🚀 Como usar

### 1. O Tensor Mágico
O coração da lib é a classe `Tensor`. Ela lembra o que você faz e sabe calcular a "volta" (backward).

```python
import numpy as np
from minigrad.tensor import Tensor

# Crie tensores (autograd ativado por padrão)
x = Tensor([2.0, 3.0])
w = Tensor([0.5, -1.0])
b = Tensor([1.0, 1.0])

# O grafo computacional é construído dinamicamente
y = x * w + b
z = y.sum()

# Backpropagation
z.backward()

print(f"Resultado: {z.data}")
print(f"Gradiente de x: {x.grad}") # dz/dx
```

### 2. Treinando uma Rede Neural (MLP)

```python
import minigrad.nn as nn
import minigrad.optim as optim

# Defina o modelo
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Otimizador
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Loop de treino (pseudocódigo)
# out = model(x)
# loss = (out - y).mean() # ou outra loss
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
```

### 3. 🤖 Treinando o GPT-2

O projeto inclui um script para treinar um modelo GPT (Transformer) do zero para aprender sequências.

```bash
# Certifique-se de estar na raiz do projeto
export PYTHONPATH=$PYTHONPATH:.
python3 train_gpt.py
```

O script `train_gpt.py` cria um modelo GPT pequeno, gera um dataset sintético (padrão repetitivo `0, 1, 2, 3...`) e treina o modelo até ele "grokkar" (entender) o padrão, alcançando Loss próxima de zero.

## 📂 Estrutura do Código

*   `minigrad/tensor.py`: O cérebro. Implementação do Tensor e Autograd.
*   `minigrad/nn.py`: As camadas (Linear, Embedding, etc.).
*   `minigrad/optim.py`: Os otimizadores (SGD, Adam).
*   `minigrad/gpt.py`: A arquitetura do modelo GPT-2.
*   `minigrad/loss.py`: Funções de perda.

## 🎓 Por que "minigrad"?

Inspirado no lendário [micrograd](https://github.com/karpathy/micrograd) do Andrej Karpathy, mas com esteróides: suporta Tensores N-dimensionais (não apenas escalares) e operações necessárias para rodar Transformers modernos.

---
*Feito com 💜 e NumPy.*
