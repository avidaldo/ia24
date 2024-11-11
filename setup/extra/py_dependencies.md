# Herramientas de gestión de dependencias y entornos virtuales en Python

## Evolución de la gestión de dependencias en Python

Cuando Python comenzó a ganar popularidad en los años 2000, surgió la necesidad de gestionar de manera efectiva las librerías. Sin un sistema eficiente para instalar y organizar estas librerías, era fácil que los proyectos se volvieran difíciles de manejar, **especialmente cuando se necesitaban diferentes versiones de las mismas librerías para distintos proyectos**.

### 1. **Pip**: El Gestor de Paquetes Estándar

`pip` apareció en 2008 y se convirtió en el estándar para instalar paquetes desde el repositorio oficial de Python, el [**Python Package Index (PyPI)** (https://pypi.org/)]fácilmente con un comando (`pip install paquete`).

Para reproducir el entorno en el que un proyecto fue desarrollado, `pip` utiliza archivos `requirements.txt`, que listan las librerías necesarias con las versiones específicas, pero al usarlos se sobreescriben las dependencias globales del sistema.
  
**Ventajas**:
- Ligero y sencillo.
- Muy flexible y compatible con cualquier proyecto Python.
- Es la opción predeterminada para casi cualquier desarrollador Python.

**Limitaciones**:
- `pip` no gestiona entornos virtuales directamente, aunque sí se usa junto con herramientas como `venv` o `virtualenv`.
- No tiene un sistema de resolución avanzada de dependencias hasta versiones más recientes (mejorado en `pip 20.3` con el nuevo "resolver").

### 2. Virtualenv y venv: aislamiento de dependencias

Ya antes de `pip`, se crea **virtualenv** para **aislar las dependencias** de un proyecto para evitar conflictos entre las versiones de las mismas librerías usadas en diferentes proyectos a través de la creación de **entornos virtuales**, que son directorios aislados donde puedes instalar dependencias sin que afecten al sistema global de Python.
 
A partir de Python 3.3, Python introdujo `venv`, una herramienta integrada y más ligera para crear entornos virtuales, eliminando la necesidad de instalar `virtualenv` por separado.

### 3. Conda: gestión de paquetes multilenguaje y entornos virtuales

`Conda` fue lanzado en 2012 como parte de la distribución **Anaconda**, que está especialmente orientada a ciencia de datos y *machine learning*. A diferencia de `pip`, es un gestor de paquetes **multilenguaje** (puede instalar paquetes de Python, R, y otros)

`Conda` ofrece paquetes **precompilados**, lo que facilita la instalación de librerías complejas como `numpy` o `pandas`, que pueden requerir compilación en ciertos sistemas si se instalan con `pip`.

**Ventajas de Conda en Machine Learning**:
- **Entornos Virtuales**: Al igual que `venv` o `virtualenv`, `conda` permite crear entornos aislados, pero lo hace de manera integrada, gestionando tanto las dependencias como los entornos virtuales.
- **Paquetes para ML y Ciencia de Datos**: `Conda` es extremadamente popular en el ámbito del machine learning y la ciencia de datos porque incluye **librerías optimizadas** para estas áreas, como `scikit-learn`, `TensorFlow` y `PyTorch`, con facilidad de instalación.
- **Compatibilidad**: Si bien puedes usar `pip` dentro de un entorno `conda`, `conda` tiene la ventaja de gestionar dependencias más complejas (como librerías con código C subyacente) que pueden ser problemáticas al instalarlas solo con `pip`.

**Limitaciones**:
- **Mayor peso**: `Conda` requiere más espacio y es más pesado que `pip` porque gestiona tanto paquetes de Python como de otros lenguajes.
- **Complejidad innecesaria en proyectos pequeños**: Para proyectos pequeños o simples, donde solo se necesitan librerías de Python, `pip` junto con `venv` o `virtualenv` es una opción más ligera.

### 4. Pipenv y Poetry: avances en la gestión de dependencias

Con el paso del tiempo, surgieron herramientas más avanzadas para hacer frente a las limitaciones de `pip` y ofrecer una experiencia más integrada en la gestión de entornos virtuales y dependencias:

- **Pipenv** (2017): Combina `pip` y `virtualenv` en una sola herramienta. Introduce un archivo `Pipfile` para gestionar dependencias de forma más organizada, separando las dependencias de producción y desarrollo. También añade un archivo de bloqueo (`Pipfile.lock`) para asegurar la reproducibilidad del entorno.
  
- **Poetry** (2018): Ofrece una solución más avanzada que `Pipenv`, incluyendo un manejo más automatizado de dependencias y publicación de proyectos. Utiliza el archivo `pyproject.toml` para la gestión.

Ambas herramientas han sido adoptadas por desarrolladores que buscan una forma más moderna de gestionar sus proyectos Python, aunque no han alcanzado el nivel de adopción masiva que tienen `pip` y `conda`.



## Conclusión: ¿Qué usar para *machine learning*?

- **Conda**: Para proyectos de *machine learning*, `conda` es la opción más recomendable, ya que permite gestionar no solo las librerías de Python, sino también las dependencias más complejas que suelen requerir compilación. Además, facilita la instalación de herramientas como `TensorFlow` y `PyTorch`, y se integra bien con entornos virtuales.
  
- **Pip**: Si estás trabajando en proyectos más ligeros o donde no necesitas paquetes de ciencia de datos complejos, `pip` sigue siendo una opción excelente, especialmente cuando se combina con `venv` para el manejo de entornos virtuales.
