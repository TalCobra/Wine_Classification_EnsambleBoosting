# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:03:50 2024

@author: João Mello
"""

#%%  funções de ajuda

import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, balanced_accuracy_score, roc_auc_score, roc_curve
    
import seaborn as sns

import matplotlib.pyplot as plt

def descritiva(df_, var, vresp='survived', max_classes=5):
    """
    Gera um gráfico descritivo da taxa de sobreviventes por categoria da variável especificada.
    
    Parâmetros:
    df : DataFrame - Base de dados a ser analisada.
    var : str - Nome da variável categórica a ser analisada.
    """
    
    df = df_.copy()
    
    if df[var].nunique()>max_classes:
        df[var] = pd.qcut(df[var], max_classes, duplicates='drop')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.pointplot(data=df, y=vresp, x=var, ax=ax1)
    
    # Criar o segundo eixo y para a taxa de sobreviventes
    ax2 = ax1.twinx()
    sns.countplot(data=df, x=var, hue=var, legend=False, palette='viridis', alpha=0.5, ax=ax2)
    ax2.set_ylabel('Frequência', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)  # Tornar o fundo do eixo 1 transparente
    
    # Exibir o gráfico
    plt.show()
    
    
def avalia_clf(clf, y, X, rótulos_y=['Não Sobreviveu', 'Sobreviveu'], base = 'treino'):
    
    # Calcular as classificações preditas
    pred = clf.predict(X)
    
    # Calcular a probabilidade de evento
    y_prob = clf.predict_proba(X)[:, -1]
    
    # Calculando acurácia e matriz de confusão
    cm = confusion_matrix(y, pred)
    ac = accuracy_score(y, pred)
    bac = balanced_accuracy_score(y, pred)

    print(f'\nBase de {base}:')
    print(f'A acurácia da árvore é: {ac:.1%}')
    print(f'A acurácia balanceada da árvore é: {bac:.1%}')
    
    # Calculando AUC
    auc_score = roc_auc_score(y, y_prob)
    print(f"AUC-ROC: {auc_score:.2%}")
    print(f"GINI: {(2*auc_score-1):.2%}")
    
    # Visualização gráfica
    sns.heatmap(cm, 
                annot=True, fmt='d', cmap='viridis', 
                xticklabels=rótulos_y, 
                yticklabels=rótulos_y)
    
    # Relatório de classificação do Scikit
    print('\n', classification_report(y, pred))
    
    # Gerar a Curva ROC
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    
    # Plotar a Curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Linha de referência (modelo aleatório)
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title(f"Curva ROC - base de {base}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def relatorio_missing(df):
    print(f'Número de linhas: {df.shape[0]} | Número de colunas: {df.shape[1]}')
    return pd.DataFrame({'Pct_missing': df.isna().mean().apply(lambda x: f"{x:.1%}"),
                          'Freq_missing': df.isna().sum().apply(lambda x: f"{x:,.0f}").replace(',','.')})


def diagnóstico(df_, var, vresp='survived', pred = 'pred', max_classes=5):
    """
    Gera um gráfico descritivo da taxa de sobreviventes por categoria da variável especificada.
    
    Parâmetros:
    df : DataFrame - Base de dados a ser analisada.
    var : str - Nome da variável categórica a ser analisada.
    """
    
    df = df_.copy()
    
    df[var+'_'] = df[var]
    if df[var+'_'].nunique()>max_classes:
        df[var+'_'] = pd.qcut(df[var], max_classes, duplicates='drop')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.pointplot(data=df, y=vresp, x=var+'_', ax=ax1)
    sns.pointplot(data=df, y=pred, x=var+'_', ax=ax1, color='red', linestyles='--', errorbar=None)
    
    # Criar o segundo eixo y para a taxa de sobreviventes
    ax2 = ax1.twinx()
    sns.countplot(data=df, x=var+'_', hue=var+'_', palette='viridis', alpha=0.5, ax=ax2)
    ax2.set_ylabel('Frequência', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)  # Tornar o fundo do eixo 1 transparente
    
    # Exibir o gráfico
    plt.show()