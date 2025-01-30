# Tipos de estimadores en scikit-learn

Es importante aclarar ciertos términos que se utilizan en scikit-learn:

- **Estimadores**: cualquier objeto que pueda estimar algunos parámetros basados en un conjunto de datos se llama estimador. La estimación en sí misma se realiza mediante el método **`fit()`**.

    - **Transformadores**: estimadores que además pueden transformar datos usando el método **`transform()`**. Por ejemplo, `SimpleImputer` es un transformador: estima valores con `fit()` y los imputa con `transform()`.
        - Scalers: son transformadores que escalan los datos.
        - Imputers: son transformadores que imputan valores faltantes.
        - Encoders: son transformadores que codifican variables categóricas.
        - Reductores de dimensionalidad: son transformadores que reducen la cantidad de variables.
        - ...

    - **Predictores**: aquellos estimadores que son capaces de hacer predicciones basadas en un conjunto de datos. Por ejemplo, el modelo de regresión lineal es un predictor: estima los hiperparámetros `fit()` y hace predicciones con **`predict()`**. 
        - Clasificadores: son predictores que predicen etiquetas categóricas.
        - Regresores: son predictores que predicen valores continuos.
        - Clusterizadores: son predictores que agrupan datos en clusters.
        - ...

El término "predictor" puede prestarse a confusión ya que también se utiliza, en general, para referirse a las *features* o variables independientes de un modelo, y a veces solo para aquellas variables que son de hecho **predictoras**, dejando fuera aquellas característcas que no tienen capacidad predictiva.

Tampoco de debe confundir el termino "transformador" con la popular arquitectura de redes neuronales "transformer", que es la base sobre la que se construyen modelos de lenguaje como GPT.

https://scikit-learn.org/stable/developers/develop.html

<!-- TODO: Explicar por separado los principios de diseño de scikit-learn en detalle con ejemplos (es un buen modo de trabajar conceptos de ingeniería del software en python)-->