# -*- coding: utf-8 -*-
"""strokeprediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LXVzyMM47x93Ms-00ieMVZjCLWKT2dOz

# Predictive Analytics: Penyakit *Stroke*


---

Oleh: [Sinta Siti Nuriah](https://www.linkedin.com/in/sintasitinuriah/)

![stroke](https://ners.unair.ac.id/site/images/Lihat/20_stroke.png)

## Latar Belakang Masalah
---
Menurut Organisasi Kesehatan Dunia (WHO), *stroke* merupakan penyebab kematian terbanyak ke-2 di dunia, yang bertanggung jawab atas sekitar 11% dari total kematian.

Di Indonesia, *stroke* menjadi penyebab utama kecacatan dan kematian, yakni sebesar 11,2% dari total kecacatan dan 18,5% dari total kematian. Menurut data Survei Kesehatan Indonesia tahun 2023, prevalensi stroke di Indonesia mencapai 8,3 per 1.000 penduduk. Stroke juga merupakan salah satu penyakit katastropik dengan pembiayaan tertinggi ketiga setelah penyakit jantung dan kanker, yaitu mencapai Rp5,2 triliun pada 2023.


Upaya Kementerian Kesehatan (Kemenkes) dalam meminimalisir penyakit *stroke* dengan meningkatkan deteksi dini dislipidemia pada pasien diabetes melitus dan hipertensi sebagai upaya pencegahan *stroke*, dengan target pada 2024 sebesar 90% atau sekitar 10,5 juta penduduk. Namun, saat ini capaian deteksi dini stroke baru mencapai sekitar 11,3% dari target.

## Deskripsi Proyek
---
Proyek ini bertujuan untuk memaksimalkan upaya pemerintah dalam meminimalisir penyakit *stroke* dengan capaian deteksi sejak dini. Maka model *machine learning* terkiat *predictive analysis* penyakit *stroke* diharapkan dapat membantu dalam memprediksi penyakit *stroke* sejak dini.

Dataset yang digunakan dalam proyek ini berterkaitan dengan seorang pasien yang kemungkinan terkena stroke berdasarkan parameter input seperti jenis kelamin, usia, berbagai penyakit, dan status merokok. Setiap baris dalam data memberikan informasi yang relevan tentang pasien.

# 1. Business Understanding
## Problem Statments
1. Tingginya angka kematian dan kecacatan akibat *stroke* di Indonesia, menjadikannya salah satu beban kesehatan utama secara nasional.
2. Capaian deteksi dini terhadap faktor risiko *stroke* seperti dislipidemia pada penderita diabetes melitus dan hipertensi masih sangat rendah (11,3%) dibandingkan target Kemenkes tahun 2024 (90%).
3. Kurangnya sistem prediksi atau skrining berbasis data untuk mengidentifikasi individu dengan risiko *stroke* tinggi secara lebih cepat dan efisien.

## Goals
1. Membangun prediksi yang mampu mengidentifikasi potensi *stroke* pada individu berdasarkan data klinis seperti hipertensi, kadar glukosa, kolesterol, usia, dan faktor risiko lainnya.
2. Meningkatkan efektivitas deteksi dini *stroke* secara otomatis menggunakan pendekatan *machine learning*, sehingga dapat membantu pemerintah atau lembaga kesehatan mempercepat capaian target 90%.
3. Menyediakan solusi berbasis data yang terukur dan dapat diintegrasikan ke dalam sistem pelayanan kesehatan.


## Solution Statement
Solusi 1: Membangun model klasifikasi baseline untuk prediksi stroke menggunakan algoritma seperti:
* Logistic Regression
* Decision Tree Classifier  

 Model ini akan digunakan sebagai baseline dengan metrik evaluasi seperti: Akurasi, Precision, Recall, dan F1-Score, khususnya pada kelas positif (stroke).

Solusi 2: Melakukan improvement dengan algoritma yang lebih kompleks, yaitu:
* XGBoost

  Model ini akan di-tuning menggunakan  Random Search untuk mengoptimalkan performa. Metrik evaluasi utama tetap fokus pada Recall, karena kesalahan negatif (false negative) dalam deteksi stroke harus diminimalisir.

Solusi 3:
Melakukan feature engineering seperti:
* Normalisasi data numerik (age, avg_glucose_level, BMI, dll.)
* Encoding variabel kategorik (gender, work_type, dll.)
* Handling imbalance data

# 2. Data Understanding

#### Data Understanding merupakan proses memahami impormasi dalam data dan menentukan kualitas dari tersebut.

## 2.1 Data Loading
#### Data loading merupakan tahapan untuk memuat dataset yang digunakan adat mudah dipahami. Informasi dataset telah dibersihkan dan dinormalisasi terlebih dahulu oleh pembuat, sehingga mudah digunakan dan dimanfaat oleh khalayak banyak.

---
### **Informasi Datasets**

|Jenis          |Keterangan                                         |
|---------------|---------------------------------------------------|
|Title          |Stroke Prediction Dataset                          |
|Source         |[Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)|
|License        |Data files © Original Authors                      |
|Visibility     |Public                                             |
|Tags           |Health, Health Conditions, Healthcare, Public Health, Binary Classification |
|Usability      |10.00                                              |

---
#### Attribute Information
|Attribute        |Describe                                                   |
|-----------------|-----------------------------------------------------------|
|id               |Unique identifier                                          |
|gender           |"Male", "Female", or "Order"                               |
|age              |age of the patient                                         |
|hypertenstion    |0 if the patient doesn't have hypertension, 1 if the patient has hypertension|
|heart_disease    | 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease|
|ever_married     |"No" or "Yes"                                              |
|work_type        |"children", "Govt_jov", "Never_worked", "Private" or "Self-employed"   |
|Residence_type   |"Rural" or "Urban"                                         |
|avg_glucose_level|average glucose level in blood                             |
|bmi              |body mass index                                            |
|smoking_status   |"formerly smoked", "never smoked", "smokes" or "Unknown"   |
|stroke           |1 if the patient had a stroke or 0 if not                  |

### Import Library yang dibutuhkan
"""

!pip install -q kaggle

# Import load data library

from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from google.colab import drive
drive.mount('/content/drive')

"""### Loading data"""

!mkdir ~/.kaggle
!cp /content/drive/MyDrive/Colab\ Notebooks/MLT/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d fedesoriano/stroke-prediction-dataset

!unzip /content/stroke-prediction-dataset.zip

df_stroke = pd.read_csv('/content/healthcare-dataset-stroke-data.csv')

df_stroke

"""Output Kode diatas memberikan informasi sebagai berikut:
*   Ada 5.110 baris (record atau jumlah pengamatan dalam dataset)
*   Terdapat 12 kolom id, age, hypertension, heart_disease, avg_glucose_level, bmi, stroke, gender, ever_married, work_type, Residence_type dan smoking_status

## 2.2 Exploratory Data Analysis (EDA)

##### Exploratory data analysis merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Teknik ini biasanya menggunakan bantuan statistik dan representasi grafis atau visualisasi.

### 2.2.1 EDA - Deskripsi Variabel

Berdasarkan informasi dari [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), variabel-variabel pada Diamond dataset adalah sebagai berikut:
1. id: Identitas unik untuk setiap pasien.
2. gender: Jenis kelamin pasien, dengan kemungkinan nilai: "Male" (laki-laki), "Female" (perempuan), atau "Other" (lainnya).
3. age: Usia pasien (dalam tahun).
4. hypertension: Status hipertensi pasien, dengan nilai:
    * 0: Tidak memiliki hipertensi
    * 1: Memiliki hipertensi
5. heart_disease: Status penyakit jantung pasien, dengan nilai:
    * 0: Tidak memiliki penyakit jantung
    * 1: Memiliki penyakit jantung
6. ever_married: Status pernikahan pasien, dengan nilai "Yes" (sudah menikah) atau "No" (belum menikah).
7. work_type: Jenis pekerjaan pasien, terdiri dari:
    * "children": Anak-anak (belum bekerja)
    * "Govt_job": Pegawai pemerintah
    * "Never_worked": Belum pernah bekerja
    * "Private": Pegawai swasta
    * "Self-employed": Wirausaha
8. Residence_type: Jenis tempat tinggal pasien, yaitu "Rural" (pedesaan) atau "Urban" (perkotaan).
9. avg_glucose_level: Rata-rata kadar glukosa dalam darah (dalam satuan mg/dL).
10. bmi: Indeks Massa Tubuh pasien (Body Mass Index).
11. smoking_status: Status merokok pasien, terdiri dari:
    * "formerly smoked": Pernah merokok
    * "never smoked": Tidak pernah merokok
    * "smokes": Masih merokok
    * "Unknown": Informasi tidak tersedia
12. stroke: Label target, dengan nilai:
    * 1: Pasien pernah mengalami stroke
    * 0: Pasien tidak pernah mengalami stroke
"""

df_stroke.info()

"""Dari info di atas terdapat 12 kolom sesuai dengan atribut yang dijelaskan pada informasi atribut pada bagian loading data dan pada deskripsi variabel. keterangan sebagai berikut:
* Terdapat 7 kolom numerik dengan tipe data **float64** dan **int64**, kolom yang dimaksud adalah id, age, hypertension, heart_disease, avg_glucose_level, bmi dan stroke.
* Terdapat 5 kolom object dengan tipe data **object** yaitu gender, ever_married, work_type, Residence_type dan smoking_status
"""

df_stroke.describe()

"""Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:

* Count adalah jumlah sampel pada data.
* Mean adalah nilai rata-rata.
* Std adalah standar deviasi.
* Min yaitu nilai minimum setiap kolom.
* Max adalah nilai maksimum.
---
Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
* 25% adalah kuartil pertama.
* 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
* 75% adalah kuartil ketiga.
"""

df_stroke.shape

"""|Jumlah Baris |Jumlah Kolom|
|-------------|------------|
|5110         |12          |

### 2.2.2 EDA - Menangani Missing Value
"""

df_stroke.isnull().sum()

df_stroke['bmi'].fillna(df_stroke['bmi'].median(), inplace=True)

df_stroke.isnull().sum()

"""###  2.2.3 EDA - Menangani Outliers"""

# Menampilkan visualisasi boxplot fitur numerikal
numerical_cols = df_stroke.select_dtypes(include=['float64']).columns
plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df_stroke[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Hitung Q1, Q3 dan IQR hanya untuk kolom numerikal
Q1 = df_stroke[numerical_cols].quantile(0.25)
Q3 = df_stroke[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

# Buat filter untuk menghapus baris yang mengandung outlier
filter = ~((df_stroke[numerical_cols] < (Q1 - 1.5 * IQR)) | (df_stroke[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

df_stroke = df_stroke[filter]

df_stroke.shape

"""Dataset sekarang telah bersih dan memiliki 4.260 sampel

### 2.2.4 EDA - Unvariate Analysis
"""

# menghapus kolom id
df_stroke = df_stroke.drop(columns=['id'])

numerical_features = df_stroke.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df_stroke.select_dtypes(include=['object']).columns

for feature in categorical_features:
  count = df_stroke[feature].value_counts()
  percent = 100*df_stroke[feature].value_counts(normalize=True)
  df_count = pd.DataFrame({'Jumlah sampel':count, 'persentase':percent})
  print(df_count)
  df_count.index.name = feature
  count.plot(kind='bar', title=feature)
  plt.show()

"""Berdasarkan informasi dari bar chart diatas terdapat persentase sebagai berikut:
* Gender perempuan sebanyak 59% dan laki-laki 40% dan sisanya lainnya.
* Status pernikahan sebanyak 62% sudah menikah sisanya belum menikah
* work_type sebanyak 56% private, 15% children, 14% self-employed 12% Govt_job dan sisanya tidakbekerja
* residence_type sebanyak 50% urban sisanya rural.
* smoking_status sebanyak 35% never_smoked, 32% unkonwn, 15% Formerly smoked dan sisanya smokes.
"""

df_stroke[numerical_features].hist(bins=50, figsize=(16,8))
plt.show()

"""Gambar di atas menunjukkan tiga grafik batang (histogram) yang menggambarkan sebaran data pasien berdasarkan tiga variabel numerik, yaitu:
1. Age(usia):
   * Grafik pertama menunjukkan jumlah pasien berdasarkan kelompok usia.
   * Terlihat bahwa pasien tersebar hampir merata dari usia muda hingga lanjut usia.
   * Tidak ada usia tertentu yang dominan, artinya stroke dapat terjadi pada semua kelompok usia, meskipun jumlah pasien sedikit lebih banyak pada usia 40–60 tahun.
2. Avg Glucose Level (Rata-rata Kadar Gula Darah):
   * Grafik kedua menunjukkan jumlah pasien berdasarkan kadar gula darah rata-rata mereka.
   * Sebagian besar pasien memiliki kadar gula darah antara 80 hingga 100 mg/dL.
   * Semakin tinggi kadar gula darah, jumlah pasiennya semakin sedikit.
   * Ini mengindikasikan bahwa kadar gula yang sangat tinggi lebih jarang, namun tetap penting karena bisa jadi faktor risiko stroke.
3. BMI (Indeks Massa Tubuh):
   * Grafik ketiga menggambarkan jumlah pasien berdasarkan nilai BMI (pengukuran berat badan relatif terhadap tinggi).
   * Mayoritas pasien memiliki BMI antara 25 hingga 30, yang masuk kategori overweight (kelebihan berat badan).
   * Ini menunjukkan bahwa banyak pasien stroke berada dalam rentang berat badan yang tidak ideal, yang dapat menjadi faktor risiko juga

Jadi **Kesimpulan** dari grafik-grafik ini, adalah:
- Pasien stroke datang dari berbagai usia, tapi lebih banyak pada usia paruh baya hingga tua.
- Kadar gula darah dan berat badan yang tidak ideal tampaknya sering ditemukan pada pasien stroke.
- Menjaga kadar gula darah dan berat badan dalam batas normal bisa membantu mengurangi risiko terkena stroke.

### 2.2.5 EDA - Multivariate Analysis

Categorical Feature
"""

for feature in categorical_features:
  sns.catplot(x=feature, y='stroke', data=df_stroke, kind='bar', dodge=False, height=4, aspect=3, palette="Set3")
  plt.title("Rata-rata 'stroke' Relatif terhadap - {}".format(feature))
  plt.show()

"""Dengan mengamati rata-rata stroke relatif terhadap fitur kategori diatas, didapatkan *insight* sebagai berikut:
1. Rata-rata stroke terhadap gender cenderung mirip. Rentangnya berada antara 3-3,5%.
2. Rata-rata stroke terhadap ever_merried tertinggi adalah sudah menikah dengan persentase sebesar 4,3%.
3. Rata-rata stroke terhadap work_Type secara umum self_employed dengan persentase sebesar 6.1%
4. Rata-rata stroke terhadap residence_type cendering mirip. Rentangnya berada diantara 3-3,5%
5. Rata-rata stroke terhadap smoking_status secara umum status formerly_smoked lebih tinggi diantara yang lainnya yaitu sekitar 5%.

Numerical Feature
"""

sns.pairplot(df_stroke[numerical_features], hue='stroke')
plt.show()

"""*Insight* dari gambar pairplot diatas adalah:
* Usia menjadi faktur penting dalam penyakit stroke ini: Stroke lebih banyak terkena pada orang yang lebih tua.
* Glukosa rata-rata tinggi dan hipertensi/jantung juga berkaitan dengan kejadian stroke

Matrix Korelasi
"""

plt.figure(figsize=(10, 8))
sns.heatmap(df_stroke[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Matrix Korelasi')
plt.show()

df_stroke[numerical_features].corr()['stroke'].sort_values(ascending=False)[1:]

"""# 3. Data Preparation

Upsample Data

## 3.1 Encoding Fitur Kategori
"""

for feature in categorical_features:
  print(feature)
  print(df_stroke[feature].unique())

df_stroke.reset_index(drop=True)

df_encoded = df_stroke.copy()

label_encoder = OneHotEncoder()
df_encoded['gender'] = df_encoded['gender'].replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
df_encoded['Residence_type'] = label_encoder.fit_transform(df_encoded[['Residence_type']]).toarray()
df_encoded['ever_married'] = label_encoder.fit_transform(df_encoded[['ever_married']]).toarray()
df_encoded['work_type'] = df_encoded['work_type'].replace({'Private':0,'Self-employed':1,'Govt_job':2,'children':-1,'Never_worked':-2}).astype(np.uint8)
df_encoded['smoking_status'] = df_encoded['smoking_status'].replace({'never smoked':0,'smokes':1,'formerly smoked':2,'unknown':-1}).astype(np.uint8)
df_encoded

"""Proses encoding ini digunakan untuk mengubah data non-numerik menjadi bentuk numerik dengan tujuan:
1. Membuat data dapat diproses oleh algoritma machine learning yang akan digunakan
2. Mempertahankan makna kategorikal dalam bentuk numerik agar informasi tidak hilang ketika dilakukannya perubahan.
"""

# Visualisasi stroke
df_encoded['stroke'].value_counts().plot(kind='bar', title='Jumlah Pasien Stroke')
plt.show()

"""## 3.2 Pembagian Data"""

X = df_encoded[['gender','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi', 'smoking_status']]
y = df_encoded['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_test.head()

"""Proses ini digunakan untuk melatih model (training) dan mengukur performa model dengan testing. Untuk kasus ini dilakukan pembagian dengan skala 80% training dan 20 testing

## 3.3 SMOTE

Proses SMOTE ini digunakan untuk mengatasi data yang tidak seimbang, untuk kasus yang sedang diselesaikan dibutuhkan proses ini dikarenakan data untuk orang yang kena stroke dan yang tidak kena, perbandingannya sangat jauh maka diperlukannlah teknik untuk mengatasi ketidakseimbangan tersebut seperti SMOTE.
"""

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train.ravel())

df_resampled = pd.concat([pd.DataFrame(X_train_resampled), pd.DataFrame(y_train_resampled)], axis=1)

df_resampled

"""### 3.3.1 Distribusi Data Training Setelah SMOTE"""

numerical_cols = [col for col in df_resampled.columns if col not in categorical_features]

for i in numerical_cols:
  ax = sns.displot(df_resampled[i], kde=True)
  plt.title(f'Distribusi {i} Sesudah SMOTE', fontsize = 15)
  plt.xlabel('')
  plt.ylabel('')
  plt.xticks(fontsize=8)
  plt.show()

"""### 3.3.2 Korelasi matriks setelah smote"""

plt.figure(figsize=(12, 10))
correlation_matrix = df_resampled.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='bwr')
plt.title('Correlation Matrix', size=15)

"""## 3.4 Standarisasi"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

"""Proses ini dibutuhkan ketika melakukan pemodelan dengan Logistic Regression

# 4. Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan.

## 4.1 Membangun model klasifikasi baseline untuk prediksi stroke

### 4.1.1 Algoritma Logistic Regression

Regresi logistik (kadang disebut model logistik atau model logit), dalam statistika digunakan untuk prediksi probabilitas kejadian suatu peristiwa dengan mencocokkan data pada fungsi logit kurva logistik
"""

model_lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model_lr.fit(X_train_scaled, y_train_resampled)

y_pred_lr = model_lr.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Akurasi: {accuracy:.4f}")

"""### 4.1.2 Algoritma Random Forest Classifier"""

model_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
model_rf.fit(X_train_scaled, y_train_resampled)

y_pred_rf = model_rf.predict(X_test_scaled)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Akurasi: {accuracy_rf:.4f}")

"""### 4.1.3.Plot perbandingan accuracy untuk kedua Model"""

# Plot perbandingan acc model
models = ['Logistic Regression', 'Random Forest']
accuracies = [accuracy, accuracy_rf]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
plt.ylim(0, 1)  # karena akurasi dalam skala 0–1
plt.title('Perbandingan Akurasi Model')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center', fontsize=12)

plt.tight_layout()
plt.show()

"""Pada plot ini dapat dilihat bahwa nilai accuracy Logistic Regression lebih tinggi dibandingkan dengan Decision Tree, dengan hal ini model Logistic Regression akan dilakukan improvement dengan hyperparameter tuning kemudian dibandingkan kembali dengan XGBoost

## 4.2 Melakukan improvement dengan algoritma yang lebih kompleks

### 4.2.1 Algoritma XGBoost
"""

scale = (len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
model_xgb = XGBClassifier(random_state=42, scale_pos_weight=scale)
model_xgb.fit(X_train_scaled, y_train_resampled)

y_pred_xgb = model_xgb.predict(X_test_scaled)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f'Akurasi: {accuracy_xgb:.2f}')

"""### 4.2.2 Hyperparameter Tunning XGBosst dan Logistic Regression dengan Random Search

#### Logistic Regression dengan Random Search
"""

param_dist = {
    'C': np.logspace(-4, 4, 20),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 500]
}

random_search = RandomizedSearchCV(
    model_lr,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train_resampled)

print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)
best_acc_lr = random_search.best_score_

"""#### XGBoost dengan Random Search"""

# Parameter space
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [1, 5, 10],
    'reg_alpha': [0, 0.5, 1]
}

random_search_xgb = RandomizedSearchCV(
    model_xgb,
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search_xgb.fit(X_train_scaled, y_train_resampled)

print("Best Parameters:", random_search_xgb.best_params_)
print("Best Accuracy:", random_search_xgb.best_score_)
best_acc_xgb = random_search_xgb.best_score_

"""### 4.1.3.Plot perbandingan accuracy untuk kedua Model"""

# Plot perbandingan acc model
models = ['Logistic Regression', 'XGBoost']
accuracies = [best_acc_lr, best_acc_xgb]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
plt.ylim(0, 1)  # karena akurasi dalam skala 0–1
plt.title('Perbandingan Akurasi Model')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center', fontsize=12)

plt.tight_layout()
plt.show()

"""Berdasarkan grafik diatas menunjukan bahwa nilai acc setelah dilakukan hyperparameter tuning pada model Logistic Regression dan XGBoost kedua berada pada nilai 90%. Namun, XGboost lebih besar dengan nilai acc 97%

# 5. Evaluasi

## 5.1 Evaluasi model baseline
"""

print("Logistic Regression Evaluation:")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

print("Decision Tree Evaluation:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Classification report untuk LR
report_lr = classification_report(y_test, y_pred_lr, output_dict=True)
# Classification report untuk RF
report_dt = classification_report(y_test, y_pred_rf, output_dict=True)

# Ambil metrik untuk kelas 0 dan 1
metrics = ['precision', 'recall', 'f1-score']
classes = ['0', '1']

data = []

for cls in classes:
    for metric in metrics:
        data.append({
            'Model': 'Logistic Regression',
            'Class': cls,
            'Metric': metric,
            'Score': report_lr[cls][metric]
        })
        data.append({
            'Model': 'Decision Tree',
            'Class': cls,
            'Metric': metric,
            'Score': report_dt[cls][metric]
        })

df_plot = pd.DataFrame(data)

plt.figure(figsize=(12, 6))
sns.barplot(data=df_plot, x='Metric', y='Score', hue='Model', ci=None, palette='Set2', dodge=True)
plt.title('Perbandingan Metrik Kelas 0 dan 1 untuk LR vs DT')
plt.ylim(0, 1)
plt.legend(title='Model')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

"""## 5.2 Evaluasi Model Algoritma Kompleks XGBoost"""

print("XGBoost Evaluation:")
y_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]
threshold = 0.9  # kamu bisa eksperimen nilainya
y_pred_threshold = (y_proba_xgb >= threshold).astype(int)

print("XGBoost Evaluation with Custom Threshold:")
print(confusion_matrix(y_test, y_pred_threshold))
print(classification_report(y_test, y_pred_threshold))

"""## 5.3 Evaluasi XGBoost dengan Hyperparameter Tuning"""

# Classification Report
best_xgb_model = random_search_xgb.best_estimator_
y_proba_xgb = best_xgb_model.predict_proba(X_test)[:, 1]
threshold = 0.9  # kamu bisa eksperimen nilainya
y_pred_threshold = (y_proba_xgb >= threshold).astype(int)

print("XGBoost Evaluation with Custom Threshold:")
print(confusion_matrix(y_test, y_pred_threshold))
print(classification_report(y_test, y_pred_threshold))

"""#Kesimpulan
Meskipun telah dilakukan berbagai pendekatan seperti penanganan data imbalance (SMOTE), pemilihan algoritma yang lebih kompleks (seperti XGBoost), serta optimasi hyperparameter menggunakan Random Search, model masih menunjukkan performa yang kurang memuaskan — terutama dalam hal recall dan f1-score pada kelas minoritas (stroke).
"""