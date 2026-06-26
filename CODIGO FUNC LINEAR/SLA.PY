import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
x_np = np.linspace(0, 1, 20)
y_np = (x_np * 5) + 4

X = torch.tensor(x_np, dtype=torch.float32).view(-1, 1)
y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)


model = nn.Sequential(nn.Linear(1, 1))
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

epocas = 4000
for e in range(epocas):
    optimizer.zero_grad()
    
    previsao = model(X)
    loss = criterion(previsao, y)
    
    loss.backward()
    optimizer.step()

    if (e+1) % 10 == 0: # Imprime só a cada 10 épocas para não poluir o terminal
        print(f"Época [{e+1}/{epocas}] | Erro (Loss): {loss.item():.6f}")

# Teste final
print("\nTeste: Quanto é a rede para x=0? (Deve ser perto de 4)")
print(model(torch.tensor([[0.0]])))


plt.style.use('dark_background')
x_test = np.linspace(0, 1, 500)
with torch.no_grad():
    y_pred = model(torch.tensor(x_test, dtype=torch.float32).view(-1, 1)).numpy()

plt.scatter(x_np, y_np, alpha=1, color='yellow', label='Dados Originais (com ruído)')
plt.plot(x_test, y_pred, color='red', linewidth=3, label='Aproximação da Rede Neural')

plt.title(f"Aproximação da Linear")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, alpha=0.2)

print("\nAbrindo o gráfico...")
plt.show()