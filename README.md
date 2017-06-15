# pyGenreClf

pyGenreClf es una herramienta que permite crear un clasificador automático de géneros musicales. Forma parte de un trabajo de final de grado para el grado en ingeniería informática de la Universidad Complutense de Madrid.
### Getting Started

A través de estás instrucciones llegaremos a una versión funcionando del proyecto. Además veremos una forma de crear un clasificador sobre cualquier conjunto de datos o utilizar uno de los creados por nosotros en este proyecto.

## Requisitos

Para poder ejecutar es necesario tener instalado Python 2.7 (https://www.python.org/downloads/).

En el caso de usuarios de Windows recomendamos instalar anaconda (con la versión de Python 2.7) que empaqueta varias librerías necesarias (https://www.continuum.io/downloads).

## Instalando dependencias en Ubuntu

Instalar los siguientes paquetes con el gestor usando el siguiente comando en consola (pedirá la contraseña de administrador):

```
sudo apt-get install python-tk python-dev graphviz bibgraphviz-dev pkg-config
```

Instalar Essentia: https://github.com/MTG/essentia/tree/master/

(Sí no puedes instalar Essentia, se utilizará otra librería alternativa en su lugar, por lo que puedes continuar sin instalarla)

## Instalando

En primer lugar clonar el repositorio:
```
git clone https://github.com/Caparrini/pyGenreClf.git
```
Para poder utilizar correctamente el programa es necesario instalar las dependencias contenidas en el archivo requirements.txt, dentro de la carpeta del proyecto clonado. Ejecutar en consola dentro de la carpeta del proyecto:
```
pip install -r requirements.txt
```

### Quick start

En la carpeta Examples/ se da un clasificador para los géneros de Beatport. Para utilizarlo y predecir el género de una canción necesitamos un archivo de audio de al menos 2 minutos de duración. A continuación  dentro de la carpeta del proyecto y con la ruta del archivo (AUDIO_FILE) ejecutamos el siguiente comando:

```
python main.py predictClass -i AUDIO_FILE -clf Examples/beats23classifier.pkl
```


## Creando un clasificador

Para poder generar un clasificador el programa funciona como un script por lo que debemos seguir una serie de pasos:

En primer lugar, es necesario extraer características de un conjunto de datos a un archivo .csv. Para ello necesitamos una carpeta que contenga a su vez carpetas con archivos de audio. El nombre de las carpetas será el de las clases:

```
python main.py featureExtraction -f CARPETA_DATOS -o midataset.csv
```

Seguidamente, hay que conseguir el mejor árbol y generar un informe sobre el conjunto de datos utilizado:
```
python main.py bestTreeClassifier -df midataset.csv -o miclasificador.pkl -f CARPETA_INFORME
```

Una vez que el clasificador miclasificador.pkl está creado se pueden hacer predicciones con él con el comando:
```
python main.py predictClass -i AUDIO_FILE -clf miclasificador.pkl
```


### Licencia

Proyecto con licencia Apache License 2.0 [LICENSE.md](LICENSE.md)

### Acknowledgments

* https://github.com/tyiannak/pyAudioAnalysis
