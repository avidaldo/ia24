# Inteligencia Artificial - 2024/25

## Apuntes y ejemplos

### Introducción y tecnologías básicas

- [*Setup* y tecnologías que utilizaremos](./setup/setup.md)
- [Introducción al Machine Learning](./intro/intro-ml.md)

- NumPy
  - [NumPy](./numpy/numpy1.ipynb)
  - [Álgebra con NumPy](./numpy/numpy2_algebra.ipynb)

- Pandas
  - [Introducción, Series y estructura de DataFrame](./pandas/pandas1.ipynb)
  - [Operaciones con DataFrames](./pandas/pandas_dataframe_op.ipynb)

### Algoritmos básicos de *machine learning*

- [K-Nearest Neighbors (KNN)](./algoritmos/knn.md)
- [Regresión lineal simple](./algoritmos/regresion_lineal_simple.ipynb)
- [Clasificación con regresión logística](./algoritmos/regresion_logistica.ipynb)
- [Árboles de decisión](./algoritmos/decision_tree.ipynb)
- [Clustering con K-means](./algoritmos/kmeans.ipynb)
  

### Modelos simples con Scikit-learn

- [Conjuntos de entrenamiento y test / *Overfitting*](./algoritmos/regresion_overfitting.ipynb)
- Clasificación
  - [Regresión logística sobre Iris *Toy Dataset* de Scikit-learn](./sklearn/iris_logistic.ipynb)
  - [Comparación de algoritmos de clasificación](./sklearn/iris_comparison_cv.py)
- Regresión
  - [Regresión lineal sobre diabetes *Toy Dataset* de Scikit-learn](./sklearn/diabetes_regression.ipynb)
- Clustering
  - [Clustering de Iris con K-means y comparación con etiquetas](./sklearn/iris_clustering.ipynb)
      
### Precio medio de viviendas por distrito de California (regresión)

- [Enmarcando un proyecto de *machine learning*](./end2end/e2e010_framing.ipynb)
- [Conjuntos de entrenamiento y de prueba](./end2end/e2e020_train_test.ipynb)
- [Exploración de datos y *feature engineering* manual](./end2end/e2e030_eda.ipynb)
- Preprocesamiento de datos
  - [Tratamiento de valores no disponibles](./end2end/e2e041_missing.ipynb)
  - [Tratamiento de *features* categóricas](./end2end/e2e042_categorical.ipynb)
  - [Escalamiento de *features*](./end2end/e2e043_scaling.ipynb)
  - [*Pipeline* de preprocesamiento](./end2end/e2e050_pipelines.ipynb)
    - [Tipos de estimadores en scikit-learn](./sklearn/tipos_estimadores.md)
    - [Clarificando: ¿qué es un *pipeline*?](./sklearn/pipeline_definition.md)
  - [Transformadores personalizados](./end2end/e2e051_custom_transformers.ipynb)
  - [Tratamiento de latitud y longitud](./end2end/e2e060_spatial_clustering.ipynb)
- [Entrenamiento y evaluación del modelo](./end2end/e2e070_model_evaluation.ipynb)
- [Optimización de hiperparámetros](./end2end/e2e080_hyperparameters.ipynb)
  - [Probando más parámetros con RandomizedSearchCV](./end2end/e2e081_hyperparameters_tarea.ipynb)
  

### Otro problema de regresión: esperanza de vida
- [Análisis de datos](./life_expectancy/training/1_framing_eda.ipynb)
- [Análisis de preprocesamiento de datos (en particular valores perdidos)](./life_expectancy/training/2_missing_values.ipynb)
- [*Pipeline* de entrenamiento y evaluación de modelos](./life_expectancy/training/3_pipeline.ipynb)
- [Aplicación simple para poner en producción el modelo](./life_expectancy/production/life_expectancy_app.py)


### Clasificación de MNIST con SVM y evaluación de modelos de clasificación

- [Clasificación de MNIST con SVM y evaluación de modelos de clasificación](./sklearn/mnist_svm_eval.ipynb)

### Clasificación: enfermedades de tiroides

- [Análisis de datos](./thyroid/eda.ipynb)
- [Modelo de clasificación](./thyroid/thyroid.ipynb)
- [Variante binaria](./thyroid/thyroid_binary.ipynb)
- [Comprobando otros modelos](./thyroid/thyroid2.ipynb)

### Redes neuronales
- [Introducción a redes neuronales](./pytorch/neural_networks.md)
- [Introducción a PyTorch y tensores](./pytorch/01_pytorch.ipynb)
- [Ejemplo completo de clasificación de Fashion MNIST](./pytorch/02_FashionMNIST.ipynb)
- [Explicación de modelos y capas lineales](./pytorch/pytorch_models.ipynb)
- [Ejemplo simple de clasificación de Iris con NN](./pytorch/pytorch_iris.ipynb)
- [Clasificación de MNIST con FNN](./pytorch/FNN_MNIST.ipynb)
- [Introducción a redes convolucionales](./pytorch/convoluciones.ipynb)
- [Clasificación de MNIST con CNN](./pytorch/CNN_MNIST.ipynb)
- [Clasificación de CIFAR-10 en escala de grises con redes neuronales](./pytorch/CIFAR10_gray.ipynb)
- [Clasificación de CIFAR-10](./pytorch/CIFAR-10/CIFAR-10.ipynb)
  - [Depliegue del modelo en una API Rest con FastAPI](https://github.com/avidaldo/cifar-10-fastapi)
    - [URL con la API desplegada para testeo](https://cifar-10-fastapi.onrender.com/)
- [Regresión en dataset "California housing"](./end2end/e2e090_neural_network/e2e090_neural_network.ipynb)

## Recursos

### [Repositorios de tareas](https://github.com/orgs/avidaldo-ia24/repositories)

### [*Playlist* con explicaciones de clases](https://www.youtube.com/playlist?list=PLb-SkCRlWLK2B-rrVZ_QOp_27lF6MGcsG)

## Recursos adicionales externos

- [Data Science cheatsheet](data-science-cheatsheet.pdf)
- [Apuntes sobre probabilidad bayesiana](https://github.com/avidaldo/mates_ml)
- [Basic Mathemathics for Machine Learning](https://github.com/hrnbot/Basic-Mathematics-for-Machine-Learning)
