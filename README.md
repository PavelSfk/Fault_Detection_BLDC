# BLDC Motor Fault Detection Using Deep Learning

## Project Description
The project implements fault detection of a brushless DC motor (BLDC) based on measurements of currents and voltages in the **Q** and **D** axes (vector space after Clarke and Park transformations). Measurements were collected from three motor phases controlled using **FOC (Field Oriented Control)** by an ESC (Electronic Speed Controller) built on a **STM32 microcontroller** (STM32F4 family). The use of Clarke and Park transformations allowed currents and voltages from the three-phase coordinate system (ABC) to be transformed into a rotating coordinate system (DQ), which simplifies motor analysis and control.

A hybrid **CNN-LSTM** model was used, which analyzes time sequences of measurements and classifies the motor state as **healthy (normal)** or **damaged (damaged)**.

Measurements were collected for different rotational speeds (5k, 6k, 7k, 9k, 10k, 15k RPM) in two operating conditions: **startup** (first 15000 samples) and **steady state** (samples 19500-25300). These data were used to train, validate, and test the model.

## Author
- **Pavel Skabeltsyn**

## Requirements and Tools

### Development Environment
The project was implemented using the following tools:

- **Python 3.12** - main programming language for ML models
- **MATLAB** - used for preliminary data analysis and trimming measurement files
- **Miniforge (Anaconda)** - conda environment manager, lightweight alternative to full Anaconda
- **Linux (Ubuntu)** - operating system running via WSL

### Miniforge Installation
Miniforge was installed using the Linux Ubuntu operating system.
The environment was downloaded using this command:
```bash
wget https://github.com/conda-forge/miniforge/releases/download/26.1.0-0/Miniforge3-26.1.0-0-Linux-x86_64.sh
```
Installation using this command:
```bash
bash Miniforge3-26.1.0-0-Linux-x86_64.sh
```

**NOTE**: during Conda installation you must agree to its initialization!  
After restarting the terminal, the text $(base)$ should appear before the username, which means that the Miniforge environment has been installed and initialized correctly.

### Installing TensorFlow with GPU Support
TensorFlow with GPU support was used for training the models, which significantly accelerated the learning process. The installation was performed according to the following procedure in the WSL/Linux terminal:

```bash
# Create a new conda environment with Python 3.12
conda create -n tf_gpu python=3.12

# Activate the environment
conda activate tf_gpu

# Install TensorFlow with GPU support
conda install tensorflow[gpu]
```

This approach automatically installs all necessary dependencies, including:
- **CUDA Toolkit** - libraries for parallel GPU computations
- **cuDNN** - deep learning library optimized for NVIDIA GPUs

After installation, GPU availability can be verified:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

Example of correct GPU detection:
```python
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

This means that TensorFlow will use the GPU to accelerate computations.

### Required Python Libraries
```text
tensorflow[gpu]>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
joblib>=1.3.0
```

## Model Architecture
The model is built using the TensorFlow/Keras framework and consists of the following layers:

1.  **Conv1D (128 filters, kernel=3)** - extraction of local features from sequences.
2.  **BatchNormalization + MaxPooling1D** - normalization and dimensionality reduction.
3.  **Dropout (20%)** - regularization, preventing overfitting.
4.  **LSTM (128 units)** - learning long-term dependencies in sequences.
5.  **BatchNormalization + Dropout (30%)**
6.  **Dense (64 neurons, ReLU activation)** - fully connected layer.
7.  **BatchNormalization + Dropout (40%)**
8.  **Dense (1 neuron, sigmoid activation)** - binary classification.

9.  <img width="1574" height="915" alt="image" src="https://github.com/user-attachments/assets/a8d0bfe8-d7ae-419f-91e0-b3e57219b744" />

## Repository Structure
```
.
├── 1CNN_1LSTM_1DENSE.py      # Script for training the model
├── test.py                    # Script for predictions on new data
├── obcinanie.m                # (MATLAB) Trimming data for startup/steady-state
├── sprawdzenie.m              # (MATLAB) Visualization of raw data
├── pomiary/                    # Raw measurement data from ESC
│   ├── stannormalny5k/
│   ├── stannormalny6k/
│   ├── stannormalny7k/
│   ├── uszkodzenia5k/
│   ├── uszkodzenia6k/
│   └── uszkodzenia7k/
├── testy_is_startup_raw/       # Training data with is_startup column (processed)
│   ├── stannormalny5k/
│   ├── stannormalny6k/
│   ├── stannormalny7k/
│   ├── uszkodzenia5k/
│   ├── uszkodzenia6k/
│   └── uszkodzenia7k/
└── test/                       # Test data with is_startup column (processed)
    ├── stannormalny10k/
    ├── stannormalny15k/
    └── uszkodzenia9k/
```

## Data Acquisition and Preprocessing

### Measurement Setup
The measurement setup consists of:
- **BLDC Motor** - the object under test
- **ESC** with **STM32G4** microcontroller - performs FOC (Field Oriented Control) and data acquisition
- **Current and Voltage Sensors** - measurements on three motor phases
- **Communication Interface** - data transfer to a computer

<img width="1616" height="825" alt="image" src="https://github.com/user-attachments/assets/9e9c45ee-21a3-4b58-b3fb-a17d34a8d8ec" />

### BLDC Motor Measurements
Measurements were taken from three phases of the BLDC motor controlled by **FOC (Field Oriented Control)** via the ESC (Electronic Speed Controller) based on **STM32** microcontroller (STM32F4 family). The following are measured in real time:

1. **Phase current measurement** - via shunt resistors  
2. **Phase voltage measurement** - via voltage dividers  
3. **Clarke Transformation** - converting three-phase (ABC) currents and voltages to stationary frame (αβ)  
4. **Park Transformation** - converting stationary frame (αβ) to rotating frame (DQ)  

These operations produce four quantities in the Q-D vector space:
- **I_Q** - current in Q-axis (torque-producing)
- **I_D** - current in D-axis (flux-producing)
- **V_Q** - voltage in Q-axis
- **V_D** - voltage in D-axis

All values are saved directly to CSV files at **1 kHz sampling frequency**. Data structure example:

```
TIMESTAMPS, I_Q_MEAS, I_D_MEAS, V_Q, V_D
1020421, 0.0202004, 0.113857, -0.00539168, -0.0293547
1020422, -0.00734559, -0.00367279, 0, -0.00898614
1020423, -0.00734559, 0.0624375, 0.000599076, -0.0257603
1020424, -0.0661103, 0.0293824, 0.016175, -0.0224653
```

### Preparing Training Data
Raw CSV files from the ESC contain long measurement recordings. For model training, equal-length samples were required, so the script `obcinanie.m`:

- Trims the first **15000 samples** (15 seconds at 1 kHz) as **startup state**  
- Trims samples from **19500 to 25300** (≈5.8 seconds) as **steady state**  

Additionally, the script adds a column **is_startup**:
- **is_startup = 1** for startup data  
- **is_startup = 0** for steady-state data

## Datasets

### Training Data
- **Healthy motor (label 0):** 30 files (5 files × 2 types × 3 speeds: 5k, 6k, 7k)  
- **Damaged motor (label 1):** 30 files (5 files × 2 types × 3 speeds: 5k, 6k, 7k)  
- **File types:** `ustalonyX.csv` (steady-state, is_startup=0), `rozruchX.csv` (startup, is_startup=1)  

### Test Data
- **Healthy motor:** 20 files (5 files × 2 types × 2 speeds: 10k, 15k)  
- **Damaged motor:** 10 files (5 files × 2 types × 1 speed: 9k)  

## Data Preprocessing

### Preliminary Analysis and Segment Extraction
Before trimming, a preliminary visual analysis of raw measurements was performed using the script `sprawdzenie.m`, which allowed assessment of signal characteristics and identification of segments representing startup and steady states.

Based on this analysis, sample ranges for extraction were determined:
- **Startup state** - first 15000 samples (from recording start)  
- **Steady state** - samples from 19500 to 25300 (≈5800 samples)  

Selecting these exact ranges ensured that all extracted segments have equal length within each category, which is crucial for training deep learning models.

The script `obcinanie.m` trims raw CSV files and adds the **is_startup** column:
- **is_startup = 1** for startup data  
- **is_startup = 0** for steady-state data

- ### Data Normalization
Each of the four continuous features (`I_Q`, `I_D`, `V_Q`, `V_D`) is normalized separately using **Z-score normalization** with `StandardScaler` from scikit-learn:

```python
# Z-score normalization of each feature (I_Q, I_D, V_Q, V_D)
for index in range(len(features)):
    current_feature = features[index]
    scaled_data[:, index] = self.scalers[current_feature].fit_transform(
        data_df[current_feature].values.reshape(-1, 1)
    ).ravel()
```

For each feature, the mean and standard deviation are calculated on the training set, and the data are transformed according to:

$$z = \frac{x - \mu}{\sigma}$$

where:
- **x** - original value  
- **μ** - feature mean on the training set  
- **σ** - feature standard deviation on the training set  

The mean and standard deviation values are saved to the `scalers.pkl` file using `joblib`, which allows applying the same normalization during prediction on new data (using `transform()` instead of `fit_transform()`).

**Note:** The `is_startup` column is not normalized, as it contains binary values (0 or 1).

<img width="1433" height="904" alt="image" src="https://github.com/user-attachments/assets/fd633608-e482-4a22-b097-ee73f104b3d4" />

### Sequence Segmentation
After normalization, the data are split into sequences using a sliding window:
- **Sequence length:** 100 time steps (100 ms at 1 kHz sampling frequency)  
- **Window shift:** 1 time step (maximal overlapping of sequences)  

Each sequence has shape **(100, 5)** and represents a single training example for the model. This approach allows the CNN-LSTM model to learn temporal dependencies in the signal, which is crucial for detecting patterns characteristic of motor faults.

## Running the Model

### Training
```bash
python 1CNN_1LSTM_1DENSE.py
```
The script will load data from the `testy_is_startup_raw` directory, preprocess it, build and train the model. The best model (based on `val_accuracy`) will be saved as `best_motor_model.keras`, and the scalers as `scalers.pkl`. Training progress plots and test metrics will be displayed.

### Prediction on New Data
```bash
python test.py
```
The script will load the trained model (`best_motor_model.keras`) and scalers (`scalers.pkl`), process all files defined in the `test_files` variable, and output for each file the classification (DAMAGED/NORMAL) along with the confidence level.

## Results
The model achieved satisfactory accuracy on the test set for steady-state data and was able to detect faults during motor startup, confirming the effectiveness of the CNN-LSTM architecture for detecting motor faults based on current-voltage signals in the vector space.

<img width="1464" height="726" alt="image" src="https://github.com/user-attachments/assets/709ca806-b7d8-4181-b58b-656df94dc2a1" />

<img width="1625" height="858" alt="image" src="https://github.com/user-attachments/assets/cfaf1587-f847-416c-9f6d-d9a1b8d14834" />
