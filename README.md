#  Breast Cancer Classification with PyTorch

Este proyecto implementa una red neuronal simple en **PyTorch** para clasificar tumores de mama (dataset **Breast Cancer Wisconsin**).  
El objetivo es predecir si un tumor es **benigno** o **maligno** a partir de características clínicas.

---

##  Requisitos del sistema

- Python ≥ 3.8  
- pip actualizado  
- Git (opcional pero recomendado)  
- Sistema operativo: Windows / Linux / macOS  

---

## ⚙ Instalación paso a paso

### 1️ Clona este repositorio

```bash
git clone https://github.com/tu_usuario/breast-cancer-pytorch.git
cd breast-cancer-pytorch
```

*O descarga el archivo `.zip` y descomprímelo manualmente.*

---

### 2️ (Opcional pero recomendado) Crea un entorno virtual

```bash
python -m venv venv
```

- En **Windows**:

```bash
venv\Scripts\activate
```

- En **Linux/macOS**:

```bash
source venv/bin/activate
```

---

### 3️ Instala las dependencias

```bash
pip install -r requirements.txt
```

---

## 📁 Estructura del repositorio

```text
├── breast_cancer_classifier.py     # Script principal
├── requirements.txt                # Paquetes necesarios
├── README.md                       # Instrucciones detalladas
```

---

## 🚀 Ejecutar el experimento

Ejecuta el script principal con:

```bash
python breast_cancer_classifier.py
```

Verás cómo se entrena la red neuronal y cómo se imprimen las pérdidas (**loss**) de entrenamiento y validación en cada época.  
También se mostrará una gráfica final de las curvas de pérdida.

---

##  Sobre el modelo

**Entrada**: 30 características numéricas del dataset.

**Arquitectura de la red neuronal**:
- 1 capa oculta con activación `ReLU`
- Capa de salida con 2 unidades (maligno / benigno)

**Configuración de entrenamiento**:
- Función de pérdida: `CrossEntropyLoss`
- Optimizador: `Adam`
- Épocas: 10
- Tamaño del batch: 32

---

##  Dataset utilizado

El dataset **Breast Cancer Wisconsin** está disponible en `sklearn.datasets` y es de uso libre para fines educativos y de investigación.

---

##  Licencia

Este proyecto está bajo licencia **MIT**.  
Puedes usarlo, modificarlo y compartirlo libremente.

---

## 👨‍💻 Autor

**Ángel Calvar Pastoriza**  
Ingeniero en Telecomunicaciones • Apasionado de la IA  
[LinkedIn](https://www.linkedin.com/in/angelcalvar) • [GitHub](https://github.com/tu_usuario)
