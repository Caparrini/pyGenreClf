# pyGenreClf

pyGenreClf es una herramienta que permite crear un clasificador automático de géneros musicales. Forma parte de un trabajo de final de grado para el grado en ingeniería informática en la Universidad Complutense de Madrid.
## Getting Started

A través de estás instrucciones llegaremos a una versión funcionando del proyecto. Además veremos una forma de crear un clasificador sobre cualquier conjunto de datos o utilizar uno de los creados por nosotros.

### Requisitos

Para poder ejecutar es necesario tener instalado Python 2.7 (https://www.python.org/downloads/).

En el caso de usuarios de windows recomendamos instalar anaconda (con la versión de Python 2.7) que empaqueta varias librerías necesarias (https://www.continuum.io/downloads).

### Instalando

En primer lugar clonar el repositorio:
```
git clone https://github.com/Caparrini/pyGenreClf.git
```
Para poder utilizar correctamente el programa es necesario instalar las dependencias contenidas en el archivo requirements.txt, dentro de la carpeta del proyecto clonado. Ejecutar en consola dentro de la carpeta del proyecto:
```
pip install -r requirements.txt
```

## Quick start

En la carpeta Examples/ se da un clasificador para los géneros de Beatport. Para utilizarlo y predecir el género de una canción necesitamos un archivo de audio de al menos 2 minutos de duración. A continuación con la ruta del archivo en una terminal en la carpeta del proyecto ejecutamos el siguiente comando:

```
python main.py predictClass -i AUDIO_FILE -clf Examples/beats23classifier.pkl
```


## Creando un clasificador

Para poder generar un clasificador el programa funcióna como un script y realizaremos los siguientes pasos.

Extraer características de un conjunto de datos en un archivo .csv. Para ello necesitamos una carpeta que contenga a su vez carpetas con archivos de audio. El nombre de las carpetas será el de las clases:

```
python main.py featureExtraction -f CARPETA_DATOS -o midataset.csv
```

Conseguir el mejor árbol y generar un informe sobre el conjunto de datos utilizado para generarlo:
```
python main.py bestTreeClassifier -df midataset.csv -o miclasificador.pkl -f CARPETA_INFORME
```

Hacer predicciónes con el clasificador con el miclasificador.pkl que hemos creado anteriormente:
```
python main.py predictClass -i AUDIO_FILE -clf miclasificador.pkl
```


## Licencia

Proyecto con licencia Apache License 2.0 [LICENSE.md](LICENSE.md)

## Acknowledgments

* https://github.com/tyiannak/pyAudioAnalysis
