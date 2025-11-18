# EEG_control_ROSNeuro
Repositorio con paquetes de ROS-Neuro para control de exo de mano usando Imaginería Motora.

## Objetivos
+ Enviar datos del casco Hero Bitbrain por red usando LSL.
+ Trabajar con los paquetes de adquisición, filtrado, visualización de señales EEG disponibles en ROS-Neuro.
+ Grabar diviendo la señal en eventos característicos de Imagenería Motora para posterior entrenamiento.
+ Crear un modelo sólido haciendo uso de las features de la señal para predecir la clase deseada.
+ Validar funcionamiento del modelo en tiempo real.
+ Sacar métricas del modelo y mostrar las señales filtradas en tiempo real para comparar ERD vs ERS.
+ Conectar vía ROS con exo de mano.

## Introducción
En primer lugar, cabe mencionar que nuestro sistema está compuesto de dos PC's, uno con Windows, donde utilizaremos el software del casco (SennsLite) para transmitir datos vía LSL y otro con Ubuntu, que lo usaremos para conectarnos con el middleware ROS. Ambos Pc's se comunican vía Ethernet con la misma red local.
  
##  Paquetes y nodos de ROS-Neuro
- Para instalar y hacer uso de ROS-Neuro:
```bash
https://github.com/rosneuro

```
### `rosneuro_acquisition`
#### Nodo: `acquisition`
Este nodo toma datos de dispositivos externos y publica los objetos detectados en el topic `/neurodata`.

- **Publicaciones**:
  - `/neurodata` (`rosneuro_msgs/NeuroFrame`)

### `rosneuro_filters`
#### Nodo: `filterchain_node`
Nodo encargado de filtrar los datos de entrada. Diferentes filtros desarrollados por ROS-Neuro pueden aplicarse.

- **Suscripciones**:
  - `/neurodata` (`rosneuro_msgs/NeuroFrame`)

- **Publicaciones**:
  - `/neurodata_filtered` (`rosneuro_msgs/NeuroFrame`)

### `rosneuro_visualizer`
#### Nodo: `neuroviz`
Nodo encargado de visualizar la señal de EEG del casco.

## Paquetes y nodos propios
### `my_hero_bci`
#### Nodo: `motor_imagery_events`
Nodo que muestra una ventana gráfica con instrucciones y va guiando con estímulos: "Reposo", "Abrir mano", "Cerrar mano". Cada vez que cambia de estímulo publica un mensaje en `\neuroevent` con un código de evento.

- **Publicaciones**:
  - `/neuroevent` (`rosneuro_msgs/NeuroEvent`)

#### Nodo: `train_model`
Nodo que lee los topics `\neurodata` y `\neuroevent` desde un .bag, extrae las señales EEG (canales 4,5 y 6), segmenta por eventos en ventanas solapadas , filtrando usando un band-pass y calculando bandpower logarítmica en bandas mu y beta. Entrena un LDA y guarda el modelo como .joblib.

- **Suscripciones**:
  - `/neurodata` (`rosneuro_msgs/NeuroFrame`)
  - `/neuroevent` (`rosneuro_msgs/NeuroEvent`)

#### Nodo: `online_classifier`
Nodo que hace predicción en tiempo real usando modelo entrenado.

- **Suscripciones**:
  - `/neurodata` (`std_msgs/NeuroFrame`)

- **Publicaciones**:
  - `/decision` (`std_msgs/String`)  
  - `/decision_proba` (`std_msgs/Float32MultiArray`)
  - `/features` (`std_msgs/Float32MultiArray`)

## Instalación

1. Clona el repositorio en tu espacio de trabajo de ROS:
   ```bash
   git clone
   ```

2. Compila el proyecto desde la carpeta principal del espacio de trabajo:
   ```bash
   catkin build
   ```

3. Haz un source del entorno:
   ```bash
   source devel/setup.bash
  
## Uso
Inicia el núcleo fundamental de ROS:
    ```bash
    roscore
    ```
### Adquisición
Ejecuta el nodo que publica los datos que adquiere el casco:
    ```bash
    rosrun rosneuro_acquisition acquisition _plugin:=rosneuro::LSLDevice _framerate:=12.5 _stream_type:=EEG _stream_name:=hero_bitbrain_eeg
    ```
### Filtrado
Ejecuta el nodo que filtra la señal de entrada a tiempo real.
Primero debes cargar los parametros en el rosparam server (se han puesto los principales filtros a utilizar):
    ```bash
    rosparam load /config/myfilterchain.yaml
    ```
    ```bash
    rosrun rosneuro_filters filterchain_node
    ```
### Visualización
Ejecuta nodo para visualizar señal en tiempo real (puedes visualizar el topic `\neurodata` o `\neurodata_filtered`):
`   ``bash
    rosrun rosneuro_visualizer neuroviz
    ```

### Paradigma visual
Ejecuta nodo para grabar estímulos:
    ```bash
    rosrun my_hero_bci motor_imagery_events.py
    ```

### Entrenamiento
Ejecuta nodo de entrenamiento:
    ```bash
    rosrun my_hero_bci train_model.py
    ```

### Clasificador
Ejecurta nodo clasificador:
    ```bash
    rosrun my_hero_bci online_classifier.py
    ```

## Mejoras
- Probar fiabilidad del modelo y reentrenar con más datos.