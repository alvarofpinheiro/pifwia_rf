#RF
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""#### Carregando a base de dados."""

df_edu = pd.read_csv('xAPI-Edu-Data.csv')

df_edu.head()

"""#### Verificando as distribuições de classes."""

df_edu['Class'].value_counts()

"""#### Verificando os registros nulos"""

df_edu.isnull().sum()

"""#### Codificando os atributos numéricos."""

Features = df_edu
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    label = LabelEncoder()
    Features[col] = label.fit_transform(Features[col])

Features.head()

"""#### Dividindo os dados em treino e teste"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_edu.drop('Class',axis=1),df_edu['Class'],test_size=0.3,random_state=0)

"""#### Verificando a forma dos dados"""

X_train.shape,X_test.shape

y_train.shape,y_test.shape

"""#### Instânciando o objeto classificador"""

random_clf = RandomForestClassifier()

"""#### Parâmetros do  objeto RandomForestClassifier
* <span style="color:red">n_estimators: número de árvores que serão criadas na floresta.</span>
* <span style="color:red"> bootstrap: se será considerado o bootstrap dataset durante a criação das árvores.</span>
* <span style="color:red"> max_features: número total de features que as árvores serão criadas.</span>
* criterion: medida de qualidade da divisão.
* splitter: estratégia utilizada para dividir o nó de decisão.
* max_depth: profundidade máxima da árvore.
* min_samples_split: número de amostras mínimas para considerar um nó para divisão.
* min_samples_leaf: número de amostras mínimas no nível folha.

#### Treinando o modelo Random Forest
"""

random_clf.fit(X_train,y_train)

"""#### Predizendo as classes a partir do modelo treinado utilizando o conjunto de teste"""

resultado = random_clf.predict(X_test)

resultado

"""#### Métricas de Validação"""

from sklearn import metrics
print(metrics.classification_report(y_test,resultado))

"""#### Verificando as features mais importantes para o modelo treinado"""

random_clf.feature_importances_

feature_imp = pd.Series(random_clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)

feature_imp

"""#### Features mais importantes de forma gráfica"""

# Commented out IPython magic to ensure Python compatibility.
def visualiza_features_importantes(features_lista):
#     %matplotlib inline

    plt.figure(figsize=(16,8))
    sns.barplot(x=features_lista, y=features_lista.index)

    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.show()

visualiza_features_importantes(feature_imp)

"""#### Selecionando apenas as features com importancia acima de um determinado score."""

features_selecionadas = []
for feature,importancia in feature_imp.iteritems():
    if importancia > 0.03:
        print("{}:\t{}".format(feature, importancia))
        features_selecionadas.append(feature)

"""#### Separando os dados em treino e teste utilizando apenas as features selecionadas"""

X_train, X_test, y_train, y_test = train_test_split(
    df_edu[features_selecionadas],
    df_edu['Class'],
    test_size=0.3,
    random_state=0
)

"""#### Verificando a nova forma dos dados"""

X_train.shape,X_test.shape

y_train.shape,y_test.shape

"""#### Instânciando o objeto classificador"""

random_clf = RandomForestClassifier(random_state=0)

"""#### Treinando novamente o modelo Random Forest"""

random_clf.fit(X_train,y_train)

"""#### Executando o algoritmo de arvore de decisão com o conjunto de teste"""

resultado = random_clf.predict(X_test)
resultado

"""#### Métricas de Validação"""

from sklearn import metrics
print(metrics.classification_report(y_test,resultado))

"""## Explorando as árvores da Floresta gerada"""

print("Número de árvores da floresta: {}".format(len(random_clf.estimators_)))
print("Árvores floresta gerada:")
for tree in random_clf.estimators_:
    print("\nNumero de nós: {}".format(tree.tree_.node_count))
    print("Profundidade da árvore: {}".format(tree.tree_.max_depth))
    print("Features importantes: {}".format(tree.feature_importances_))
    print("\nObjeto: {}".format(tree))

"""#### Selecionando uma árvore da floresta"""

tree0 = random_clf.estimators_[0]

"""#### Visualizando de forma gráfica"""

from sklearn.tree import export_graphviz
import graphviz 

dot_data = export_graphviz(
         tree0,
         max_depth=2,
         out_file=None,
         feature_names=X_train.columns,
         class_names=['0','1','2'], 
         filled=True, rounded=True,
         proportion=True,
         node_ids=True,
         rotate=False,
         label='all',
         special_characters=True
        )  
graph = graphviz.Source(dot_data)  
graph

"""#### Selecionando outra árvore da floresta"""

tree1 = random_clf.estimators_[1]

"""#### Visualizando de forma gráfica"""

dot_data = export_graphviz(
         tree1,
         max_depth=2,
         out_file=None,
         feature_names=X_train.columns,
         class_names=['0','1','2'], 
         filled=True, rounded=True,
         proportion=True,
         node_ids=True,
         rotate=False,
         label='all',
         special_characters=True
        )  
graph = graphviz.Source(dot_data)  
graph