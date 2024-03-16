'''
Ejemplo de cómo implementar una red neuronal 
para clasificación con (MLPClassifier) utilizando Sklearn
'''
#Cargar librerías
from sklearn.neural_network import MLPClassifier #Multilayer perceptron classifier de scikit-learn
from sklearn.datasets import load_iris #Dataset simple y sencillo conocido como Iris
from sklearn.model_selection import train_test_split #Divide arreglos o matrices en subconjuntos aleatorios para entrenamiento y prueba
from sklearn.preprocessing import StandardScaler #Permite transformar los estadísticos a la funcion normal estándar Z = (X-u)/s
from sklearn.metrics import accuracy_score # Permite calcular el puntaje de precision de la clasificación. 

#Cargar el conjunto de datos Iris
iris = load_iris()
X,y = iris.data, iris.target

# (Opcional) exploración de los datos
print(type(iris)) #conocer el tipo de dato del dataset
print(iris.keys()) #llama las claves/llaves que tiene el dataset
print(iris['data'])
