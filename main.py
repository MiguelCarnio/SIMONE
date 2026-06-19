import torch
import numpy as np
import matplotlib.pyplot as plt
from rede import configurar_plot, menu_configuracao, gerar_dados_dinamicos, criar_modelo

# ==========================================
# MAIN
# ==========================================
def main():

    configurar_plot()

    juncao, amostras, ruido, tamanho_rede, epochs, lr = menu_configuracao()

    min_x = min(t[0] for t in juncao)
    max_x = max(t[1] for t in juncao)

    idade_np = np.linspace(min_x, max_x, amostras)
    valor_np = gerar_dados_dinamicos(juncao, idade_np, ruido)

    X = torch.tensor(idade_np, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(valor_np, dtype=torch.float32).view(-1, 1)

    model, criterion, optimizer = criar_modelo(tamanho_rede, lr)

    for e in range(epochs):
        optimizer.zero_grad()

        loss = criterion(model(X), y)

        loss.backward()
        optimizer.step()

        if e % 500 == 0:  # Ajuste para não imprimir excessivamente
            print(f"Época {e} | Erro: {loss.item():.4f}")

    x_test = np.linspace(min_x, max_x, 500)

    with torch.no_grad():
        y_pred = model(
            torch.tensor(x_test, dtype=torch.float32).view(-1, 1)
        ).numpy()

    plt.scatter(idade_np, valor_np, alpha=0.3, color='gray', label='Dados')
    plt.plot(x_test, y_pred, color='red', linewidth=3, label='Rede Neural')

    plt.title("Aproximação de Funções com Rede Neural")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.show()

if __name__ == "__main__":
    main()