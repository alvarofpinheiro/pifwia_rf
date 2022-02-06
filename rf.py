#instalar biblioteca Orange Canvas
!pip install Orange3

#importar biblioteca Orange Canvas
import Orange

#importar dados do disco local
from google.colab import files  
files.upload()

#instanciar objeto de dados com base no caminho gerado com a importação do arquivo
dados = Orange.data.Table("/content/Simulacao-1-dados.csv")

#imprimir os primeiros 5 registros
for d in dados[:5]:
  print(d)

#explorar os metadados
qtde_campos = len(dados.domain.attributes)
qtde_cont = sum(1 for a in dados.domain.attributes if a.is_continuous)
qtde_disc = sum(1 for a in dados.domain.attributes if a.is_discrete)
print("%d metadados: %d continuos, %d discretos" % (qtde_campos, qtde_cont, qtde_disc))
print("Nome dos metadados:", ", ".join(dados.domain.attributes[i].name for i in range(qtde_campos)),)

#explorar domínios
dados.domain.attributes

#explorar classe
dados.domain.class_var

#explorar dados
print("Campos:", ", ".join(c.name for c in dados.domain.attributes))
print("Registros:", len(dados))
print("Valor do registro 1 da coluna 1:", dados[0][0])
print("Valor do registro 2 da coluna 2:", dados[1][1])

#criar amostra
qtde100 = len(dados)
qtde70 = len(dados) * 70 / 100
qtde30 = len(dados) * 30 / 100
print("Qtde 100%:", qtde100)
print("Qtde  70%:", qtde70)
print("Qtde  30%:", qtde30)
amostra = Orange.data.Table(dados.domain, [d for d in dados if d.row_index < qtde70])
print("Registros:", len(dados))
print("Registros:", len(amostra))

#Técnica Random Forest (RF)
rf = Orange.classification.RandomForestLearner(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, class_weight=None, preprocessors=None)

#ligar técnica RF aos dados
dados_rf = rf(dados)

#treinar e testar técnica RF com os dados
avalia_rf = Orange.evaluation.CrossValidation(dados, [rf], k=5)

#avaliar os indicadores para o RF
print("Acurácia:  %.3f" % Orange.evaluation.scoring.CA(avalia_rf)[0])
print("Precisão:  %.3f" % Orange.evaluation.scoring.Precision(avalia_rf)[0])
print("Revocação: %.3f" % Orange.evaluation.scoring.Recall(avalia_rf)[0])
print("F1:        %.3f" % Orange.evaluation.scoring.F1(avalia_rf)[0])
print("ROC:       %.3f" % Orange.evaluation.scoring.AUC(avalia_rf)[0])

#Técnica Support Vector Machine (SVM)
svm = Orange.classification.SVMLearner(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, max_iter=-1, preprocessors=None)

#ligar técnica SVM aos dados
dados_svm = svm(dados)

#treinar e testar técnica SVM com os dados
avalia_svm = Orange.evaluation.CrossValidation(dados, [svm], k=5)

#avaliar os indicadores para o SVM
print("Acurácia:  %.3f" % Orange.evaluation.scoring.CA(avalia_svm)[0])
print("Precisão:  %.3f" % Orange.evaluation.scoring.Precision(avalia_svm)[0])
print("Revocação: %.3f" % Orange.evaluation.scoring.Recall(avalia_svm)[0])
print("F1:        %.3f" % Orange.evaluation.scoring.F1(avalia_svm)[0])
print("ROC:       %.3f" % Orange.evaluation.scoring.AUC(avalia_svm)[0])

#Ténica k-Nearest Neighbors
knn = Orange.classification.KNNLearner(n_neighbors=5, metric='euclidean', weights='distance', algorithm='auto', metric_params=None, preprocessors=None)

#ligar técnica KNN aos dados
dados_knn = knn(dados)

#treinar e testar técnica KNN com os dados
avalia_knn = Orange.evaluation.CrossValidation(dados, [knn], k=10)

#avaliar os indicadores para o KNN
print("Acurácia:  %.3f" % Orange.evaluation.scoring.CA(avalia_knn)[0])
print("Precisão:  %.3f" % Orange.evaluation.scoring.Precision(avalia_knn)[0])
print("Revocação: %.3f" % Orange.evaluation.scoring.Recall(avalia_knn)[0])
print("F1:        %.3f" % Orange.evaluation.scoring.F1(avalia_knn)[0])
print("ROC:       %.3f" % Orange.evaluation.scoring.AUC(avalia_knn)[0])

#validar probabilidade para o alvo para as 3 técnicas
import random
random.seed(42)
testa = Orange.data.Table(dados.domain, random.sample(dados, 5))
treina = Orange.data.Table(dados.domain, [d for d in dados if d not in testa])
aprende = [rf, svm, knn]
classifica = [learner(treina) for learner in aprende]

#imprimir a probabilidade para primeiro domínio da classe

alvo = 0
print("Probabilidade para %s:" % dados.domain.class_var.values[alvo])
print("Classe Alvo |", " | ".join("%-5s" % l.name for l in classifica))

c_valores = dados.domain.class_var.values
for d in testa:
    print(
        ("{:<15} " + " | {:.2f}" * len(classifica)).format(
            c_valores[int(d.get_class())], *(c(d, 1)[alvo] for c in classifica)
        )
    )

#imprimir a probabilidade para segundo domínio da classe

alvo = 1
print("Probabilidade para %s:" % dados.domain.class_var.values[alvo])
print("Classe Alvo |", " | ".join("%-5s" % l.name for l in classifica))

c_valores = dados.domain.class_var.values
for d in testa:
    print(
        ("{:<15} " + " | {:.2f}" * len(classifica)).format(
            c_valores[int(d.get_class())], *(c(d, 1)[alvo] for c in classifica)
        )
    )

#validar o aprendizado para as 3 técnicas
aprendizado = [rf, svm, knn]
avaliacao = Orange.evaluation.CrossValidation(dados, aprendizado, k=5)

#imprimir os indicadores para as 3 técnicas
print(" " * 10 + " | ".join("%-4s" % learner.name for learner in aprendizado))
print("Acurácia  %s" % " | ".join("%.2f" % s for s in Orange.evaluation.CA(avaliacao)))
print("Precisão  %s" % " | ".join("%.2f" % s for s in Orange.evaluation.Precision(avaliacao)))
print("Revocação %s" % " | ".join("%.2f" % s for s in Orange.evaluation.Recall(avaliacao)))
print("F1        %s" % " | ".join("%.2f" % s for s in Orange.evaluation.F1(avaliacao)))
print("ROC       %s" % " | ".join("%.2f" % s for s in Orange.evaluation.AUC(avaliacao)))
