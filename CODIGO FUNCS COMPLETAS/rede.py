import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAÇÃO VISUAL
# ==========================================
def configurar_plot():
    plt.style.use('dark_background')
    plt.rcParams['figure.figsize'] = (12, 8)


def menu_configuracao():
    print("="*70)
    print("REDE NEURAL BASIQUINHA KK")
    print("="*70)

    juncao = []
    print("\n[1] Usar predefinição | [2] Montar própria")
    modo = input("Escolha (1 ou 2): ").strip()

    if modo == '1':
        print("\n==================================")
        print("   TIPOS DE FUNÇÃO DISPONÍVEIS:   ")
        print("==================================")
        print("A:sla, muito doido")
        print("B:trigonometria e geometria kk")
        print("C: sem ideia")
        print("D:here go again")
        print("E: n sei")
        escolha_pre = input("Escolha: ").strip().upper()
        pre = {
            'A': [(0.0, 1.0, 'seno'), (1.0, 2.0, 'serrote'), (2.0, 3.0, 'absoluta'), (3.0, 4.0, 'exponencial')],
            'B': [(0.0, 1.0, 'constante'), (1.0, 2.0, 'tangente'), (2.0, 3.0, 'raiz'), (3.0, 4.0, 'cubica')],
            'C': [(0.0, 1.5, 'cosseno'), (1.5, 2.5, 'linear'), (2.5, 3.5, 'quadratica'), (3.5, 5.0, 'seno')],
            'D': [(0.0, 1.0, 'absoluta'), (1.0, 2.0, 'raiz'), (2.0, 3.0, 'raiz'), (3.0, 4.0, 'absoluta')],
            'E': [(0.0, 0.5, 'seno'), (0.5, 1.0, 'tangente'), (1.0, 1.8, 'cubica'), (1.8, 2.4, 'cosseno'), (2.4, 3.0, 'serrote')]
        }
        juncao = pre.get(escolha_pre, pre['A'])
    else:
        opcoes = ['linear', 'quadratica', 'cubica', 'constante', 'seno', 'cosseno',
                  'tangente', 'sigmoid', 'exponencial', 'raiz', 'absoluta', 'serrote']

        while True:
            print("\n==================================")
            print("   TIPOS DE FUNÇÃO DISPONÍVEIS:   ")
            print("==================================")
            print("  1. Linear      2. Quadratica    ")
            print("  3. Cubica      4. Constante     ")
            print("  5. Seno        6. Cosseno       ")
            print("  7. Tangente    8. Sigmoid       ")
            print("  9. Exponencial 10. Raiz         ")
            print(" 11. Absoluta    12. Serrote      ")
            print("==================================")
            
            tipo = input("Nome da função (ou 'sair'): ").strip().lower()

            if tipo == 'sair':
                if len(juncao) == 0:
                    print("\n[!] Calma lá! Você precisa adicionar pelo menos UMA função antes de sair.")
                    continue
                break

            if tipo in opcoes:
                try:
                    inicio = float(input("Início: "))
                    fim = float(input("Fim: "))
                    juncao.append((inicio, fim, tipo))
                    print(f"=> Função '{tipo}' adicionada com sucesso! (De {inicio} até {fim})")
                except ValueError:
                    print("\n[!] Por favor, digite apenas números válidos para Início e Fim!")
            else:
                print("\n[!] Ops! Nome de função não reconhecido. Verifique a ortografia e tente novamente.")

    # Convertendo os inputs garantindo que se ficarem em branco ele assuma o padrão
    amostras_input = input("\nQtd amostras (500): ").strip()
    amostras = int(amostras_input) if amostras_input else 500
    
    ruido_input = input("Ruído (0.05): ").strip()
    ruido = float(ruido_input) if ruido_input else 0.05
    
    tamanho_rede = input("Rede 1, 2 ou 3 (2): ").strip()
    tamanho_rede = tamanho_rede if tamanho_rede else '2'
    
    epochs_input = input("Épocas (5000): ").strip()
    epochs = int(epochs_input) if epochs_input else 5000
    
    lr_input = input("Learning Rate (0.005): ").strip()
    lr = float(lr_input) if lr_input else 0.005

    return juncao, amostras, ruido, tamanho_rede, epochs, lr


# GERAÇÃO DOS DADOS
def gerar_dados_dinamicos(configuracao, x, nivel_ruido):
    y = np.zeros_like(x)

    for inicio, fim, tipo in configuracao:
        mascara = (x >= inicio) & (x < fim)
        x_trecho = x[mascara]

        if len(x_trecho) == 0:
            continue

        centro = inicio + (fim - inicio) / 2
        largura = fim - inicio

        if tipo == 'linear':
            y[mascara] = 2.0 * (x_trecho - inicio) 

        elif tipo == 'quadratica':
            y[mascara] = 4.0 * ((x_trecho - centro) / largura) ** 2 - 0.5

        elif tipo == 'cubica':
            y[mascara] = 10.0 * ((x_trecho - centro) / largura) ** 3

        elif tipo == 'constante':
            y[mascara] = 1.0

        elif tipo == 'seno':
            y[mascara] = np.sin(4 * np.pi * (x_trecho - inicio) / largura)

        elif tipo == 'cosseno':
            y[mascara] = np.cos(4 * np.pi * (x_trecho - inicio) / largura)

        elif tipo == 'tangente':
            y[mascara] = np.clip(
                np.tan(3 * np.pi * (x_trecho - inicio) / largura),
                -2.0,
                2.0
            )

        elif tipo == 'sigmoid':
            z = ((x_trecho - centro) / largura) * 10
            y[mascara] = 2 / (1 + np.exp(-z)) - 1

        elif tipo == 'exponencial':
            y[mascara] = np.exp(2 * (x_trecho - inicio) / largura) - 1.0

        elif tipo == 'raiz':
            y[mascara] = np.sqrt((x_trecho - inicio) / largura) * 2.0 - 1.0

        elif tipo == 'absoluta':
            y[mascara] = np.abs((x_trecho - centro) / largura) * -4.0 + 1.0

        elif tipo == 'serrote':
            y[mascara] = ((x_trecho - inicio) * 5) % 2.0 - 1.0

    return y + np.random.randn(*x.shape) * nivel_ruido

# ==========================================
# 4. CRIAÇÃO DA REDE
# ==========================================
def criar_modelo(tamanho, lr):
    if tamanho == '1':
        m = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    elif tamanho == '3':
        m = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    else:
        m = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    return m, nn.MSELoss(), torch.optim.Adam(m.parameters(), lr=lr)