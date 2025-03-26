import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import lightgbm as lgbm
import matplotlib.pyplot as plt

# Carregando os dados
dados = pd.read_csv('winequality-white new.csv', sep=",")

# Limpeza da coluna 'alcohol'
dados['alcohol'] = (
    dados['alcohol']
    .str.replace('R$', '', regex=False)  # Remove "R$"
    .str.replace('.', '', regex=False)   # Remove pontos (separadores de milhares)
    .str.replace(',', '.', regex=False)  # Substitui vírgula por ponto (separador decimal)
    .str.strip()                         # Remove espaços em branco no início e no final
)
dados['alcohol'] = dados['alcohol'].astype(float)

# Definindo X e y
X = dados.drop(columns='quality')
y = dados['quality']

# Dividindo os dados em treino, validação e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Criando o modelo LightGBM
lgbm_clf = lgbm.LGBMClassifier(
    boosting_type='gbdt',
    objective='multiclass',  # Objetivo de classificação multiclasse
    num_class=y.nunique(),  # Número de classes
    random_state=42,
    min_data_in_leaf=50,
    min_split_gain=0.3,
    reg_alpha=0.1,
    reg_lambda=0.1,
    #class_weight='balanced',  # Lidar com classes desbalanceadas
    n_estimators=2000
)

# Espaço de busca reduzido para otimização de hiperparâmetros
param_grid = {
    'num_leaves': [30, 50, 70, 100],
    'max_depth': [5, 10, 15, 20],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# Configurando a busca com RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=lgbm_clf,
    param_distributions=param_grid,
    n_iter=50,  # Número de iterações
    scoring='accuracy',
    cv=5,  # Aumentar o número de folds
    n_jobs=-1,
    verbose=10,
    random_state=42
)

# Treinando o modelo com RandomizedSearchCV
random_search.fit(X_train, y_train)

# Melhores parâmetros encontrados
melhor_parametro = random_search.best_params_
melhor_estimador = random_search.best_estimator_

# Previsões na base de treino
grid_pred_train_class = melhor_estimador.predict(X_train)
grid_pred_train_prob = melhor_estimador.predict_proba(X_train)

# Previsões na base de teste
grid_pred_test_class = melhor_estimador.predict(X_test)
grid_pred_test_prob = melhor_estimador.predict_proba(X_test)

# Métricas de avaliação
confusion_matrix_train = confusion_matrix(y_train, grid_pred_train_class)
confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix_train)

acc_tree_train = accuracy_score(y_train, grid_pred_train_class)
prec_tree_train = precision_score(y_train, grid_pred_train_class, average='weighted')

confusion_matrix_test = confusion_matrix(y_test, grid_pred_test_class)
confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix_test)

acc_tree_test = accuracy_score(y_test, grid_pred_test_class)
prec_tree_test = precision_score(y_test, grid_pred_test_class, average='weighted')

print("Avaliação da Árvore (Base de Treino)")
print(f"Acurácia: {acc_tree_train:.1%}")
print(f"Precision: {prec_tree_train:.1%}")

print("Avaliação da Árvore (Base de Teste)")
print(f"Acurácia: {acc_tree_test:.1%}")
print(f"Precision: {prec_tree_test:.1%}")

# Plotando a matriz de confusão
plt.rcParams['figure.dpi'] = 600
confusion_matrix_display.plot(colorbar=False, cmap='Oranges')
plt.title('LightGBM: Teste (Após Grid Search)')
plt.xlabel('Observado (Real)')
plt.ylabel('Classificado (Modelo)')
plt.show()

# Resultados
auc = random_search.best_score_
gini = auc * 2 - 1
print(f"Melhores hiperparâmetros: {random_search.best_params_}")
print(f"\nGini médio na validação cruzada: {gini:.2%}")