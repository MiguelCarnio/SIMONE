import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from rede import configurar_plot, menu_configuracao, gerar_dados_dinamicos, criar_modelo
import time
def main():
    configurar_plot()
    
    #SOCORRO POR FAVOR, NAO SEI OQ ESTOU FAZENDO DIREITO
    arquivos = glob.glob("*.pth")
    print("\n" + "="*35)
    print("      GERENCIADOR DE MODELOS       ")
    print("="*35)
    print("[0] Criar Novo Treino (do zero)")
    for i, arq in enumerate(arquivos, 1):#CARREGA O ARQUIVO BRUH
        print(f"[{i}] Carregar e Continuar: {arq}")
    
    escolha = input("\nEscolha uma opção: ").strip()

    if escolha.isdigit() and 1 <= int(escolha) <= len(arquivos):#VAI VER SE O NUMEOR DIGITADO É UM ARQUIVO DISPONIVEL

        nome_arquivo = arquivos[int(escolha)-1]
        checkpoint = torch.load(nome_arquivo)#LOAD NO ARQUIVO DE MERDA QUE É USADO EM REDE NEURAL ;-;
        #JUNÇÃO DEFINIÇÃO E TRATA OS ERROS DE UMA VEZ
        juncao = checkpoint.get('juncao')
        
        if juncao is None:#SE NAO TIVER NADA NO JUNCAO
            print("\n[!] Modelo antigo detectado (sem dados salvos). Você precisará reconfigurar as funções.")
            juncao, amostras, ruido, tamanho_rede, _, _ = menu_configuracao()
        else:#CASO TENHA ALGO ELE VAI PEGAR OS DADOS
            amostras = checkpoint['amostras']
            ruido = checkpoint['ruido']
            tamanho_rede = checkpoint['tamanho_rede']
            print(f"\n=> Configurações recuperadas de {nome_arquivo}!")
        #PARAMETROS INICIAIS DE X && ESCOLHAS DO USUARIO A RESPEITO DOS PARAMETROS DE TREINAMENTO
        min_x = min(t[0] for t in juncao)
        max_x = max(t[1] for t in juncao)

        print("\n--- Parâmetros de Treinamento ---")
        epocas_input = input("Quantas épocas extras deseja treinar? (5000): ").strip()
        epocas_treino = int(epocas_input) if epocas_input else 5000

        model, criterion, optimizer = criar_modelo(tamanho_rede, 0.005) 
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        lr_salvo = optimizer.param_groups[0]['lr']
        lr_input = input(f"Novo Learning Rate (deixe em branco para manter o salvo ({lr_salvo:.6f})): ").strip()

        if lr_input:
            novo_lr = float(lr_input)
            for param_group in optimizer.param_groups:
                param_group['lr'] = novo_lr
            print(f"=> Learning Rate atualizado para: {novo_lr:.6f}")
        else:
            print(f"=> Mantendo o Learning Rate salvo: {lr_salvo:.6f}")

    else:#SE NAO FOR UM ARQUIVO EXISTENTE ELE RODA OUTRA COISA(NAO CONSEGUI FAZER QUE SE ESTIVER VAI APARECER OUTRA COISA, QUEBROU O CODIGO (:) 

        nome_novo = input("\nNome para salvar este novo modelo (ex: rede1): ").strip()
        nome_arquivo = nome_novo + ".pth" if not nome_novo.endswith('.pth') else nome_novo
        print(f"\n=> Configurando NOVO modelo. Será salvo como: {nome_arquivo}")
        
        juncao, amostras, ruido, tamanho_rede, epocas_treino, lr = menu_configuracao()#VAI PARA AS ONFIG DO OUTRO ARQUIVO PY
        min_x = min(t[0] for t in juncao)
        max_x = max(t[1] for t in juncao)
        
        model, criterion, optimizer = criar_modelo(tamanho_rede, lr)

        

    x_plot = np.linspace(min_x, max_x, amostras)
    y_plot = gerar_dados_dinamicos(juncao, x_plot, ruido)
    
    X = torch.tensor(x_plot, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(y_plot, dtype=torch.float32).view(-1, 1)

    # LOOP DE TREINAMENTO
    print(f"\nIniciando {epocas_treino} épocas de treinamento...")
    for e in range(epocas_treino):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()


        print(f"Época [{e+1}/{epocas_treino}] | Erro (Loss): {loss.item():.8f}")

    # SALVAMENTO
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'juncao': juncao,               
        'amostras': amostras,           
        'ruido': ruido,                 
        'tamanho_rede': tamanho_rede    
    }, nome_arquivo)
    print(f"\n[!] Progresso e configurações salvos com sucesso em: {nome_arquivo}")

    # VISUALIZAÇÃO
    x_test = np.linspace(min_x, max_x, 500)
    with torch.no_grad():
        y_pred = model(torch.tensor(x_test, dtype=torch.float32).view(-1, 1)).numpy()

    plt.scatter(x_plot, y_plot, alpha=0.3, color='gray', label='Dados Originais (com ruído)')
    plt.plot(x_test, y_pred, color='red', linewidth=3, label='Aproximação da Rede Neural')
    
    plt.title(f"Aproximação de Funções - Modelo: {nome_arquivo}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    print("\nAbrindo o gráfico...")
    plt.show()

if __name__ == "__main__":
    main()