# Heart Disease Detection with Neural Networks

A neural network for heart disease detection built **from scratch in NumPy** (forward pass, binary cross-entropy, backpropagation, gradient descent), compared with **TensorFlow/Keras** implementations. The project also includes a hyperparameter study and a reflection on fairness and ethics in clinical machine learning.

**Tools:** Python · NumPy · pandas · scikit-learn · TensorFlow/Keras · Matplotlib · Seaborn

---

## Project overview

Heart disease is a leading cause of death worldwide, and identifying at-risk patients early supports timely diagnosis and treatment. This project predicts whether a patient has heart disease from 13 clinical attributes. It uses that task to explore how neural networks learn: first by implementing every step by hand, then by reproducing the model in Keras.

The project has five parts:
1. **Data preparation and analysis:**
   - profiling and quality checks
   - visual exploration of the target and key features
   - leak-free train/validation/test splits
2. **Neural network from scratch (NumPy):** a feed-forward network with sigmoid activations and hand-written backpropagation, trained with mini-batch gradient descent.
3. **TensorFlow/Keras:**
   - a ReLU + Adam baseline
   - a tanh + SGD-with-momentum alternative
   - a side-by-side comparison with the NumPy model
4. **Hyperparameter analysis:** a grid over learning rate, hidden-layer size and optimiser.
5. **Reflection:**
   - the modelling findings
   - a bias and fairness concern, and how pipeline choices affect it
   - a mitigation and how to evaluate it
   - lessons about deep learning fundamentals

## Highlights

- **Data quality first:**
  - 723 of the 1,025 rows (70.5%) are exact duplicates, leaving 302 unique patients.
  - Duplicates are removed **before** splitting, so no patient appears in both training and test data.
  - Imputation, outlier capping, encoding and scaling are all fitted on the training split only.
- **All three models reach a test ROC-AUC of about 0.90:**

  | Model | Test accuracy | Test recall | Test ROC-AUC |
  | --- | --- | --- | --- |
  | NumPy NN (sigmoid + gradient descent) | **0.852** | **0.848** | 0.898 |
  | Keras Baseline (ReLU + Adam) | 0.803 | 0.758 | **0.908** |
  | Keras Alt (tanh + SGD with momentum) | 0.820 | 0.758 | 0.896 |

- **Overfitting is visible and explained:**
  - The NumPy network's validation loss bottoms out around epoch 10 and then rises.
  - Early stopping keeps the Keras models stable.
- **Hyperparameters:**
  - Learning rate had the biggest effect: mean validation AUC was 0.839 at 0.01 and 0.764 at 0.001.
  - Adam beat SGD on average.
  - The smaller hidden layer (8 units) generalised slightly better than 16 on this small cohort.
- **Strongest signals in the data:**
  - chest-pain type, maximum heart rate, exercise-induced angina, ST depression (`oldpeak`) and the number of major vessels (`ca`)
  - cholesterol shows almost no linear relationship with the target

---

## Dataset

The data is `data/heart.csv`: 1,025 rows, of which 302 are unique patient records. There are 13 features, a binary target and no missing values.

| Feature | Description |
| --- | --- |
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG result (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (0–2) |
| `ca` | Number of major vessels coloured by fluoroscopy (0–4) |
| `thal` | Thalassemia result (0–3) |
| `target` | **1 = heart disease, 0 = no heart disease** |

---

## Methodology

**Preprocessing**

| Step | Approach |
| --- | --- |
| Duplicates | Removed before splitting (1,025 → 302 rows) |
| Split | Stratified train/validation/test: 192 / 49 / 61 patients |
| Missing values | None present; fallback imputation uses the training-set median (numeric) or mode (categorical) |
| Outliers | IQR capping on the five continuous measurements, with bounds learned from training data; categorical codes are never capped |
| Encoding | One-hot encoding for `cp`, `restecg`, `slope`, `ca` and `thal` |
| Scaling | `StandardScaler` fitted on training data only |

**Models**

| Model | Architecture | Training |
| --- | --- | --- |
| NumPy NN | 27 → 16 (sigmoid) → 1 (sigmoid) | Mini-batch gradient descent, learning rate 0.5, 300 epochs |
| Keras Baseline | 27 → 16 → 8 (ReLU) → 1 (sigmoid) | Adam, learning rate 0.001, early stopping |
| Keras Alt | 27 → 16 → 8 (tanh) → 1 (sigmoid) | SGD with momentum 0.9, learning rate 0.01, early stopping |

All three models use binary cross-entropy loss. They are evaluated on accuracy, precision, recall, F1 and ROC-AUC, with confusion matrices and ROC curves.

**Hyperparameter grid:** learning rate {0.01, 0.001} × hidden units {8, 16} × optimiser {Adam, SGD}, with 12 epochs per configuration.

---

## Repository structure

```
Heart-Disease-Detection/
├── notebooks/
│   └── Heart_Disease_Detection_Neural_Networks.ipynb   # Full analysis with outputs and charts
├── docs/
│   └── Heart_Disease_Detection_Neural_Networks.html    # HTML version (read without running anything)
├── data/
│   └── heart.csv                                       # Heart disease dataset
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/u2jyothibhat/Heart-Disease-Detection.git
cd Heart-Disease-Detection
pip install -r requirements.txt
jupyter notebook notebooks/Heart_Disease_Detection_Neural_Networks.ipynb
```

The notebook finds `data/heart.csv` whether you start Jupyter from the repository root or from `notebooks/`. Seeds are fixed and TensorFlow runs in deterministic mode, so results are reproducible. The whole notebook runs in under a minute on a CPU.

## Limitations and next steps

- **Small sample:** with 302 unique patients, the validation (49) and test (61) sets are small, so metrics can move several points with a handful of patients. Stratified k-fold cross-validation would give more stable estimates.
- **NumPy model overfitting:** early stopping or L2 regularisation would improve its generalisation.
- **Fairness:** the cohort is about 68% male. A subgroup audit of sensitivity and false negative rate by sex and age band should come before any clinical use.
- **Decision threshold:** for screening, the threshold should be tuned for recall on the disease class rather than fixed at 0.5.

## Author

**Jyothi Kodenkiri**: [LinkedIn](https://www.linkedin.com/in/jyothi-k-89bb423/) · [GitHub](https://github.com/u2jyothibhat)
