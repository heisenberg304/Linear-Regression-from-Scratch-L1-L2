# INFORME TÉCNICO
# Fundamentos matemáticos e implementación desde cero de regresión lineal con regularización L1, L2 y solución analítica

**Proyecto:** Implementación y estudio matemático de modelos de regresión lineal  
**Área:** Machine Learning / Aprendizaje supervisado  
**Lenguaje:** Python  
**Bibliotecas principales:** NumPy y Scikit-Learn (carga de datos y evaluación)  
**Propósito:** Comprender los fundamentos matemáticos y los mecanismos internos que permiten construir y optimizar modelos de regresión lineal.

---

# 1. Resumen ejecutivo

El presente proyecto tiene como propósito estudiar e implementar desde cero los fundamentos matemáticos de la regresión lineal y sus principales técnicas de regularización: L1 (Lasso) y L2 (Ridge). El trabajo se desarrolla con un enfoque educativo y matemático, priorizando la comprensión de las operaciones algebraicas, las derivadas, los gradientes y los métodos de optimización que intervienen en el funcionamiento de estos algoritmos.

Se implementaron tres aproximaciones:

1. Regresión Ridge mediante Gradient Descent.
2. Regresión Lasso mediante el algoritmo ISTA y el operador Soft Thresholding.
3. Regresión Ridge mediante solución analítica, utilizando álgebra matricial y centrado de datos para incorporar el intercepto.

El desarrollo se realizó utilizando NumPy para las operaciones matriciales y la implementación de los algoritmos. Scikit-Learn se utilizó para obtener el conjunto de datos y calcular métricas de evaluación, sin emplear sus modelos de regresión.

El proyecto permitió estudiar cómo las funciones de pérdida determinan el comportamiento de los modelos, cómo las derivadas permiten encontrar direcciones de optimización y cómo las diferentes penalizaciones modifican los valores de los parámetros durante el entrenamiento.

# 2. Introducción

La regresión lineal es uno de los fundamentos del aprendizaje automático supervisado. Su objetivo consiste en establecer una relación matemática entre un conjunto de variables independientes y una variable dependiente, permitiendo realizar estimaciones a partir de los patrones presentes en los datos.

Aunque su formulación parece sencilla, detrás de su funcionamiento intervienen conceptos fundamentales de álgebra lineal, cálculo diferencial y optimización matemática.

La implementación de una regresión lineal desde cero permite comprender aspectos que normalmente quedan ocultos cuando se utilizan bibliotecas de alto nivel. Entre ellos se encuentran:

- La representación matricial de los datos.
- La construcción de una función de pérdida.
- La derivación de los gradientes.
- La actualización iterativa de los parámetros.
- La influencia de la regularización.
- La obtención de soluciones mediante álgebra matricial.
- La diferencia entre métodos iterativos y soluciones analíticas.

A partir de estos fundamentos se desarrollaron implementaciones independientes de Ridge y Lasso, además de una solución analítica para Ridge.

# 3. Objetivos

## 3.1. Objetivo general

Comprender e implementar los fundamentos matemáticos de la regresión lineal con regularización L1 y L2, estudiando los métodos de optimización iterativos y la solución analítica mediante álgebra matricial.

## 3.2. Objetivos específicos

1. Comprender la representación matricial de la regresión lineal.
2. Derivar matemáticamente la función de pérdida y sus gradientes.
3. Implementar Gradient Descent para optimizar los parámetros de Ridge.
4. Comprender el efecto de la regularización L2 sobre los pesos del modelo.
5. Estudiar la función de valor absoluto y su comportamiento en la regularización L1.
6. Implementar el operador Soft Thresholding mediante ISTA.
7. Derivar e implementar la solución analítica de Ridge.
8. Comprender el tratamiento del intercepto en la solución analítica.
9. Comparar los fundamentos y comportamientos de los diferentes métodos.

# 4. Fundamentos matemáticos de la regresión lineal

## 4.1. Representación matemática

La regresión lineal busca aproximar una variable objetivo mediante una combinación lineal de características.

Para una observación individual:

$$
\hat y_i=w_1x_{i1}+w_2x_{i2}+\cdots+w_dx_{id}+b
$$

Donde:

- $x_{ij}$: valor de la característica j para la observación i.
- $w_j$: peso asociado a la característica j.
- $b$: intercepto o término independiente.
- $\hat y_i$: predicción del modelo.

La expresión puede representarse de manera matricial:

$$
\boxed{\hat y=Xw+b\mathbf{1}}
$$

Donde:

$$
X\in\mathbb{R}^{N\times d}
$$

$$
w\in\mathbb{R}^{d\times1}
$$

$$
y,\hat y,\mathbf{1}\in\mathbb{R}^{N\times1}
$$

Aquí, N representa el número de observaciones y d la cantidad de características.

La multiplicación matricial produce:

$$
Xw\in\mathbb{R}^{N\times1}
$$

Esto permite calcular simultáneamente las predicciones de todas las observaciones.

## 4.2. Error de predicción

El error representa la diferencia entre el valor predicho y el valor real:

$$
e=\hat y-y
$$

Sustituyendo la expresión del modelo:

$$
\boxed{e=Xw+b\mathbf{1}-y}
$$

El objetivo del entrenamiento consiste en encontrar los parámetros w y b que minimicen el error.

## 4.3. Función de pérdida: Mean Squared Error

Una de las funciones de pérdida más utilizadas en regresión es el error cuadrático medio:

$$
\boxed{
MSE(w,b)=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
}
$$

Su representación matricial es:

$$
\boxed{
MSE(w,b)=\frac{1}{N}e^Te
}
$$

La transpuesta del vector de errores multiplicada por el propio vector produce la suma de los errores elevados al cuadrado.

Esta función penaliza especialmente los errores de mayor magnitud debido al término cuadrático.

## 4.4. Derivación del gradiente respecto a los pesos

Partimos de:

$$
L(w,b)=\frac{1}{N}e^Te
$$

Donde:

$$
e=Xw+b\mathbf{1}-y
$$

Aplicando la regla de la cadena:

$$
\nabla_w L=
\frac{2}{N}X^Te
$$

Sustituyendo el error:

$$
\boxed{
\nabla_w L=
\frac{2}{N}X^T(Xw+b\mathbf{1}-y)
}
$$

Esta expresión indica cómo cambia la función de pérdida cuando modificamos los pesos.

## 4.5. Derivación respecto al intercepto

La derivada respecto al intercepto es:

$$
\boxed{
\frac{\partial L}{\partial b}
=
\frac{2}{N}\mathbf{1}^Te
}
$$

Como la multiplicación por el vector de unos equivale a sumar los elementos:

$$
\boxed{
\frac{\partial L}{\partial b}
=
\frac{2}{N}\sum_{i=1}^{N}e_i
}
$$

El intercepto también se optimiza, pero no necesariamente recibe una penalización de regularización.

# 5. Optimización mediante Gradient Descent

## 5.1. Fundamento matemático

Gradient Descent es un algoritmo de optimización iterativa que busca minimizar una función mediante actualizaciones sucesivas de sus parámetros.

La idea consiste en desplazarse en dirección opuesta al gradiente, debido a que este señala la dirección de mayor crecimiento local de la función.

La actualización general es:

$$
\boxed{
w^{(t+1)}=w^{(t)}-\alpha\nabla_w L
}
$$

Para el intercepto:

$$
\boxed{
b^{(t+1)}=b^{(t)}-\alpha\frac{\partial L}{\partial b}
}
$$

Donde:

- $t$: iteración actual.
- $\alpha$: tasa de aprendizaje.
- $\nabla_w L$: gradiente respecto a los pesos.
- $\partial L/\partial b$: derivada respecto al intercepto.

## 5.2. Interpretación de la tasa de aprendizaje

El parámetro alfa determina el tamaño de cada actualización.

Si es demasiado pequeño, el entrenamiento puede ser lento.

Si es demasiado grande, las actualizaciones pueden sobrepasar el mínimo de la función de pérdida y provocar oscilaciones o divergencia.

Por tanto, la tasa de aprendizaje influye directamente en la estabilidad y velocidad de convergencia.

## 5.3. Convergencia

En una función cuadrática convexa, como la utilizada en regresión Ridge, el algoritmo puede converger hacia el mínimo global cuando se cumplen las condiciones apropiadas sobre la tasa de aprendizaje.

La convergencia puede analizarse mediante:

- La estabilización de la función de pérdida.
- La reducción de la norma del gradiente.
- La disminución de las diferencias entre actualizaciones sucesivas.

# 6. Regularización L2: Ridge Regression

## 6.1. Fundamento matemático

Ridge introduce una penalización basada en la suma de los cuadrados de los pesos.

Su función objetivo es:

$$
\boxed{
L(w,b)=
\frac{1}{N}\|Xw+b\mathbf{1}-y\|_2^2
+\lambda\|w\|_2^2
}
$$

Donde:

$$
\|w\|_2^2=\sum_{j=1}^{d}w_j^2
$$

El parámetro lambda controla la intensidad de la regularización.

Cuando lambda aumenta, el modelo recibe una penalización mayor por utilizar pesos de gran magnitud.

La regularización busca reducir la complejidad del modelo y controlar el sobreajuste.

## 6.2. Derivación del gradiente de Ridge

Partimos de:

$$
L(w,b)=
\frac{1}{N}e^Te+\lambda w^Tw
$$

Derivando respecto a w:

$$
\nabla_w L=
\frac{2}{N}X^Te+
2\lambda w
$$

Por tanto:

$$
\boxed{
\nabla_w L=
\frac{2}{N}X^T(Xw+b\mathbf{1}-y)
+2\lambda w
}
$$

El segundo término representa la contribución de la regularización.

Respecto al intercepto:

$$
\boxed{
\frac{\partial L}{\partial b}
=
\frac{2}{N}\mathbf{1}^Te
}
$$

El intercepto no recibe penalización.

## 6.3. Comportamiento de la regularización L2

La penalización cuadrática produce una reducción gradual de los pesos.

A medida que aumenta lambda, los coeficientes tienden a acercarse a cero.

Sin embargo, Ridge generalmente no genera coeficientes exactamente iguales a cero para valores finitos de lambda. Su comportamiento consiste principalmente en reducir la magnitud de los pesos.

Esto permite mantener las características en el modelo, aunque algunas tengan una contribución reducida.

## 6.4. Implementación mediante Gradient Descent

En el proyecto se implementó la función de pérdida:

$$
L(w,b)=
\frac{1}{N}e^Te+\lambda w^Tw
$$

Y los gradientes:

$$
\nabla_w L=
\frac{2}{N}X^Te+2\lambda w
$$

$$
\nabla_b L=
\frac{2}{N}\sum e_i
$$

Los parámetros se actualizaron iterativamente mediante Gradient Descent.

Esta implementación permitió estudiar cómo interviene la regularización durante cada actualización de los pesos.

# 7. Solución analítica de Ridge mediante álgebra matricial

## 7.1. Fundamento

A diferencia de Gradient Descent, la solución analítica permite obtener directamente los parámetros que minimizan la función objetivo, sin realizar actualizaciones iterativas.

Para simplificar la derivación se parte de la función Ridge:

$$
L(w)=
\frac{1}{N}\|Xw-y\|_2^2+
\lambda\|w\|_2^2
$$

Derivamos respecto a $w$:

$$
\nabla_w L=
\frac{2}{N}X^T(Xw-y)+2\lambda w
$$

Para encontrar el mínimo, igualamos el gradiente a cero:

$$
\frac{2}{N}X^T(Xw-y)+2\lambda w=0
$$

Multiplicando por $N/2$:

$$
X^TXw-X^Ty+N\lambda w=0
$$

Reorganizando:

$$
(X^TX+N\lambda I)w=X^Ty
$$

Finalmente:

$$
\boxed{
w=(X^TX+N\lambda I)^{-1}X^Ty
}
$$

Esta es la solución analítica de Ridge sin intercepto explícito.

La matriz identidad $I$ permite sumar la penalización a la matriz de productos cruzados.

Su dimensión debe coincidir con la cantidad de características:

$$
I\in\mathbb{R}^{d\times d}
$$

## 7.2. Incorporación del intercepto

Cuando el modelo incorpora un intercepto no regularizado:

$$
\hat y=Xw+b\mathbf{1}
$$

La función objetivo pasa a ser:

$$
L(w,b)=
\frac{1}{N}\|Xw+b\mathbf{1}-y\|_2^2
+\lambda\|w\|_2^2
$$

Para obtener una solución más sencilla, se utiliza el centrado de los datos.

Se definen las medias:

$$
\bar X=\frac{1}{N}\sum_{i=1}^{N}X_i
$$

$$
\bar y=\frac{1}{N}\sum_{i=1}^{N}y_i
$$

Y se construyen las matrices centradas:

$$
X_c=X-\bar X
$$

$$
y_c=y-\bar y
$$

La ecuación original es:

$$
y=Xw+b\mathbf{1}
$$

Al restar las medias:

$$
y-\bar y\mathbf{1}
=
Xw+b\mathbf{1}
-(\bar Xw+b)\mathbf{1}
$$

El término del intercepto se cancela:

$$
\boxed{y_c=X_cw}
$$

Por consiguiente, ya no necesitamos incluir b en el cálculo de los pesos.

## 7.3. Solución analítica centrada

Aplicando la derivación de Ridge a los datos centrados:

$$
\boxed{
w=(X_c^TX_c+N\lambda I)^{-1}X_c^Ty_c
}
$$

Una vez encontrados los pesos, recuperamos el intercepto:

$$
\boxed{
b=\bar y-\bar X^Tw
}
$$

Esta formulación permite calcular los pesos y el intercepto por separado, sin penalizar el intercepto.

## 7.4. Comparación con Gradient Descent

Los dos métodos resuelven el mismo problema de optimización cuando utilizan la misma función objetivo y los mismos parámetros.

La diferencia está en el procedimiento:

Gradient Descent se aproxima progresivamente al mínimo.

La solución analítica obtiene los parámetros directamente mediante álgebra matricial.

Cuando Gradient Descent converge completamente, sus parámetros deben coincidir con los de la solución analítica, salvo pequeñas diferencias numéricas.

Esta comparación permite comprobar experimentalmente la comprensión de ambos métodos.

# 8. Regularización L1: Lasso Regression

## 8.1. Fundamento matemático

Lasso incorpora una penalización basada en el valor absoluto de los pesos.

Su función objetivo es:

$$
\boxed{
L(w,b)=
\frac{1}{N}|Xw+b\mathbf{1}-y|_2^2
+\lambda|w|_1
}
$$

Donde:

$$
|w|1=\sum{j=1}^{d}|w_j|
$$

La diferencia principal respecto a Ridge está en la naturaleza de la penalización.

Ridge utiliza una función cuadrática, mientras que Lasso utiliza valores absolutos.

## 8.2. Derivada de la penalización L1

La función valor absoluto presenta una particularidad matemática en el punto cero.

Para un peso individual:

$$
f(w_j)=|w_j|
$$

Su derivada es:

$$
\frac{d}{dw_j}|w_j|=
\begin{cases}
1 & w_j>0\\
-1 & w_j<0
\end{cases}
$$

En $w_j=0$, la derivada convencional no existe.

En este punto se utiliza el concepto de subgradiente:

$$
\partial |w_j|=
\begin{cases}
{1} & w_j>0\\
[-1,1] & w_j=0\\
{-1} & w_j<0
\end{cases}
$$

Esto permite describir matemáticamente las condiciones de optimalidad de Lasso.

## 8.3. ¿Por qué Lasso produce pesos exactamente iguales a cero?

La penalización L1 presenta una región de optimalidad en la que el peso puede permanecer exactamente en cero.

Esto se relaciona con el comportamiento del subgradiente y con el operador Soft Thresholding.

A diferencia de Ridge, Lasso puede eliminar características del modelo asignándoles coeficientes exactamente iguales a cero.

Esta propiedad se conoce como selección de características o feature selection.

## 8.4. Algoritmo ISTA

ISTA significa Iterative Shrinkage-Thresholding Algorithm.

Es un método de optimización proximal utilizado para resolver problemas que combinan una función diferenciable con otra no diferenciable.

En Lasso, la función objetivo se divide en dos partes:

$$
f(w)=\frac{1}{N}|Xw+b\mathbf{1}-y|_2^2
$$

$$
g(w)=\lambda|w|_1
$$

La primera parte es diferenciable y la segunda contiene la penalización no diferenciable.

El procedimiento consiste en realizar una actualización mediante el gradiente de la parte diferenciable y aplicar posteriormente el operador proximal de la parte L1.

Primero:

$$
z=w-\alpha\nabla f(w)
$$

Después:

$$
\boxed{
w^{(t+1)}=S_{\alpha\lambda}(z)
}
$$

## 8.5. Operador Soft Thresholding

El operador se define como:

$$
\boxed{
S_{\alpha\lambda}(z)=
\begin{cases}
z-\alpha\lambda & z>\alpha\lambda\\

z+\alpha\lambda & z<-\alpha\lambda\\

0 & |z|\leq\alpha\lambda
\end{cases}
}
$$

Donde alfa representa el tamaño del paso y lambda la intensidad de regularización.

El funcionamiento es el siguiente:

Si el valor supera el umbral positivo, se reduce.

Si está por debajo del umbral negativo, se incrementa hacia cero.

Si está dentro del intervalo del umbral, se convierte exactamente en cero.

Este mecanismo es el responsable de producir soluciones dispersas.

## 8.6. Actualización del intercepto

En la implementación desarrollada, el intercepto no recibe regularización.

Por ello, se actualiza mediante el gradiente de la función de error cuadrático:

$$
{
b^{(t)}
-\alpha\frac{2}{N}\sum_{i=1}^{N}e_i
}
$$

Los pesos se actualizan mediante ISTA y el intercepto mediante Gradient Descent.

Esta combinación permite optimizar ambos componentes respetando la estructura de la función objetivo.

# 9. Desarrollo experimental

## 9.1. Conjunto de datos

Para las pruebas se utilizó el conjunto de datos Diabetes disponible en Scikit-Learn mediante:


`texttt{load_diabetes()}`


El conjunto contiene:

* 442 observaciones.

* 10 características predictoras.

* Una variable objetivo.

La división de los datos se realizó utilizando train_test_split con:

`
test_size=0.2
`

`
random_state=42
`

Se obtuvieron las siguientes dimensiones:

| Conjunto | Dimensiones |
|---------|--------:|
| X_train | 353 × 10 |
| y_train | 353 |
| X_test | 89 × 10 |
| y_test | 89 |



El entrenamiento se realizó con el conjunto de entrenamiento y las métricas finales se calcularon sobre el conjunto de prueba.

## 9.2. Métricas de evaluación

Se utilizaron cuatro métricas para estudiar el comportamiento de las predicciones.

Mean Squared Error (MSE)

$$
MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
$$

Mide el promedio de los errores al cuadrado.

Root Mean Squared Error (RMSE)

$$
RMSE=\sqrt{MSE}
$$

Representa el error en las mismas unidades de la variable objetivo.

Mean Absolute Error (MAE)

$$
MAE=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat y_i|
$$

Mide el promedio de las diferencias absolutas entre las predicciones y los valores reales.

Coeficiente de determinación (R²)

$$
R^2=
1-\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\bar y)^2}
$$

Permite evaluar cuánto mejor se comporta el modelo respecto a una predicción basada en la media del conjunto objetivo.

Un valor más próximo a uno indica un mejor ajuste en términos de esta métrica.

# 10. Resultados experimentales

Los archivos de implementación contienen los siguientes resultados registrados.


| Modelo | MSE | RMSE | MAE | R² |
|---------|--------:|-------:|------:|------:|
| Ridge - Gradient Descent | 2870.4058 | 53.5762 | 43.1716 | 0.4582 |
| Ridge - Solución analítica | 2889.3534 | 53.7527 | 42.8180 | 0.4546 |
| Lasso - ISTA | 2858.4316 | 53.4643 | 43.3080 | 0.4605 |



Los resultados muestran que Ridge mediante Gradient Descent, solución analítica de Ridge y Lasso mediante ISTA obtuvieron métricas de evaluación similares en los registros disponibles.

## 10.1. Observación sobre la comparación experimental

Es importante señalar que los archivos no utilizan los mismos parámetros de regularización para la implementación de Ridge y Lasso:

Ridge con Gradient Descent y solución analítica utiliza $\lambda=0.00001$.

Lasso con Gradient Descent utiliza $\lambda=0.1$.

Por tanto, ambas implementaciones no están resolviendo exactamente el mismo problema de optimización y sus resultados no permiten evaluar directamente la equivalencia entre ambos métodos.

Como $e^Te$ ya es un escalar, `np.mean()` no divide entre el número de observaciones.

La función de pérdida debe expresarse como:

$$
\boxed{
L=\frac{e^Te}{N}+\lambda w^Tw
}
$$

para mantener coherencia con la función objetivo utilizada en la derivación.

Por consiguiente, los resultados experimentales deben considerarse registros de las ejecuciones realizadas, no una comparación controlada entre métodos equivalentes.

# 11. Análisis e interpretación

El desarrollo de las implementaciones permite identificar diferencias fundamentales entre las técnicas estudiadas.

## 11.1. Diferencias entre Ridge y Lasso

Ridge y Lasso utilizan regularización para controlar la magnitud de los parámetros.

Sin embargo, su comportamiento es diferente debido a la función de penalización utilizada.

Ridge utiliza la norma L2, que penaliza cuadráticamente los pesos. Su efecto consiste en reducir progresivamente la magnitud de los coeficientes.

Lasso utiliza la norma L1, cuya geometría y comportamiento en el origen permiten producir coeficientes exactamente iguales a cero.

Por tanto, Ridge se caracteriza principalmente por la reducción de los pesos, mientras que Lasso también permite realizar selección de características.

## 11.2. Diferencias entre Gradient Descent e ISTA

Gradient Descent utiliza directamente el gradiente de una función diferenciable para actualizar los parámetros.

ISTA combina dos operaciones:

Una actualización mediante el gradiente de la parte diferenciable.

Una operación proximal que aplica el umbral correspondiente a la penalización L1.

Esta diferencia es fundamental porque el valor absoluto no es diferenciable convencionalmente en cero.

## 11.3. Diferencias entre solución iterativa y analítica

La solución analítica permite obtener los parámetros de Ridge mediante una expresión matricial.

Gradient Descent necesita iteraciones para aproximarse al mínimo.

Ambos procedimientos se fundamentan en la minimización de la misma función objetivo, siempre que sus parámetros y convenciones de escala coincidan.

El estudio de ambos métodos permite comprender la relación entre optimización diferencial y álgebra lineal.

# 12. Limitaciones y consideraciones técnicas

El proyecto tiene un propósito educativo y presenta algunas limitaciones que deben considerarse:

Los hiperparámetros utilizados no fueron seleccionados mediante búsqueda sistemática.

No se realizó validación cruzada para evaluar la estabilidad de los modelos.

Los resultados disponibles corresponden a una única división de entrenamiento y prueba.

No se realizó una comparación controlada entre Ridge analítica y Ridge mediante Gradient Descent debido a la diferencia de lambda.

La convergencia de los métodos iterativos debe verificarse mediante la norma de los gradientes y la estabilidad de la función objetivo.

La tasa de aprendizaje de ISTA debe respetar las condiciones de convergencia correspondientes.

Las implementaciones están orientadas al aprendizaje de los fundamentos y no a la construcción de una biblioteca de producción.

Estas limitaciones no invalidan el propósito educativo del proyecto, pero delimitan el alcance de sus conclusiones experimentales.

# 13. Conclusiones

El desarrollo de este proyecto permitió estudiar los fundamentos matemáticos que hacen posible el funcionamiento de la regresión lineal regularizada.

En primer lugar, se comprendió que la regresión lineal puede expresarse mediante operaciones matriciales, lo que permite representar de manera compacta las predicciones y los errores de múltiples observaciones.

En segundo lugar, se estudió la relación entre la función de pérdida y sus derivadas. La obtención de los gradientes permitió comprender cómo Gradient Descent actualiza los parámetros para minimizar el error.

En tercer lugar, se identificaron las diferencias matemáticas entre las regularizaciones L1 y L2. Mientras Ridge reduce la magnitud de los coeficientes mediante una penalización cuadrática, Lasso utiliza una penalización basada en valores absolutos que permite generar pesos exactamente iguales a cero.

Asimismo, la implementación de ISTA permitió comprender el concepto de optimización proximal y la función del operador Soft Thresholding en problemas de optimización no diferenciables.

Finalmente, la derivación de la solución analítica de Ridge permitió relacionar las operaciones de álgebra matricial con las condiciones de optimalidad obtenidas mediante cálculo diferencial. También permitió comprender cómo el centrado de los datos elimina algebraicamente el intercepto durante el cálculo de los pesos y cómo este puede recuperarse posteriormente.

En conjunto, el proyecto cumplió su propósito principal: comprender los mecanismos matemáticos internos de los modelos de regresión lineal regularizada mediante su derivación e implementación independiente.

# 14. Referencias bibliográficas

Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Second Edition. Springer.

Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.

Boyd, S., & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press.

Beck, A., & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences, 2(1), 183–202.

Scikit-Learn Developers. Diabetes dataset. Documentación de Scikit-Learn.

Nota sobre el alcance del proyecto: este trabajo se desarrolló con fines de aprendizaje y comprensión matemática. Su objetivo no fue construir un modelo comercial ni maximizar el rendimiento predictivo, sino estudiar, derivar e implementar los fundamentos que permiten funcionar a los algoritmos de regresión lineal L1 y L2.