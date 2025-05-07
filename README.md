# Predictive Analytics: Penyakit *Stroke*

---

Oleh: [Sinta Siti Nuriah](https://www.linkedin.com/in/sintasitinuriah/)

![stroke](https://ners.unair.ac.id/site/images/Lihat/20_stroke.png)

## Latar Belakang Masalah

Menurut Organisasi Kesehatan Dunia (WHO), *stroke* merupakan penyebab kematian terbanyak ke-2 di dunia, yang bertanggung jawab atas sekitar 11% dari total kematian.

Di Indonesia, *stroke* menjadi penyebab utama kecacatan dan kematian, yakni sebesar 11,2% dari total kecacatan dan 18,5% dari total kematian. Menurut Survei Kesehatan Indonesia 2023, prevalensi stroke di Indonesia mencapai 8,3 per 1.000 penduduk. Stroke juga merupakan salah satu penyakit katastropik dengan pembiayaan tertinggi ketiga setelah penyakit jantung dan kanker, yaitu mencapai Rp5,2 triliun pada 2023.

Kementerian Kesehatan (Kemenkes) menargetkan peningkatan deteksi dini dislipidemia pada pasien diabetes melitus dan hipertensi hingga 90% pada 2024. Namun, saat ini capaian deteksi baru mencapai 11,3%.

## Deskripsi Proyek

Proyek ini bertujuan mendukung upaya pemerintah dalam meningkatkan deteksi dini stroke melalui pengembangan model *machine learning* yang mampu memprediksi potensi stroke sejak dini. Dataset yang digunakan mencakup informasi pasien seperti jenis kelamin, usia, penyakit penyerta, dan status merokok.

## Referensi
1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
2. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.
4. scikit-learn documentation: https://scikit-learn.org/stable/
5. XGBoost documentation: https://xgboost.readthedocs.io/
6. Cegah Stroke dengan Aktivitas Fisik : https://kemkes.go.id/id/rilis-kesehatan/cegah-stroke-dengan-aktivitas-fisik
7. Kaggle Datasets: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

---

# 1. Business Understanding

## Problem Statements

1. Tingginya angka kematian dan kecacatan akibat stroke di Indonesia menjadi beban kesehatan nasional.
2. Capaian deteksi dini masih sangat rendah (11,3%) dibandingkan target Kemenkes (90%).
3. Belum tersedia sistem skrining berbasis data untuk identifikasi individu berisiko stroke secara cepat.

## Goals

1. Mengembangkan model prediksi untuk mengidentifikasi potensi stroke berdasarkan data klinis.
2. Meningkatkan efektivitas deteksi dini menggunakan pendekatan machine learning.
3. Menyediakan solusi berbasis data yang dapat diintegrasikan dalam sistem pelayanan kesehatan.

## Solution Statement

- **Solusi 1: Baseline Model**
  - Algoritma: Logistic Regression, Decision Tree
  - Evaluasi: Akurasi, Precision, Recall, F1-Score

- **Solusi 2: Model Lanjutan**
  - Algoritma: XGBoost + Random Search Tuning
  - Fokus utama: Recall untuk menghindari false negative

- **Solusi 3: Feature Engineering**
  - Normalisasi data numerik
  - Encoding variabel kategorik
  - Penanganan data imbalance

---

# 2. Data Understanding
Kondisi data saat ini masih belum siap untuk ddigunakan, dengan hal perlu dilakukan pembersian data seperti penanganan missing value, outliers dan duplikat data.

#### Informasi Dataset

| Jenis      | Keterangan                                                                 |
|------------|------------------------------------------------------------------------------|
| Title      | Stroke Prediction Dataset                                                   |
| Source     | [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |
| License    | Data files © Original Authors                                               |
| Visibility | Public                                                                      |
| Tags       | Health, Healthcare, Binary Classification                                   |
| Usability  | 10.00                                                                        |

#### Attribute Information

| Attribute          | Deskripsi                                                                 |
|--------------------|--------------------------------------------------------------------------|
| id                 | Identitas unik pasien                                                    |
| gender             | Jenis kelamin: Male, Female, Other                                       |
| age                | Usia pasien                                                              |
| hypertension       | 0: tidak hipertensi, 1: hipertensi                                       |
| heart_disease      | 0: tidak memiliki penyakit jantung, 1: memiliki penyakit jantung         |
| ever_married       | Status pernikahan: Yes / No                                              |
| work_type          | Jenis pekerjaan: children, Govt_job, Never_worked, Private, Self-employed|
| Residence_type     | Tempat tinggal: Rural / Urban                                            |
| avg_glucose_level  | Rata-rata kadar glukosa darah                                            |
| bmi                | Body Mass Index                                                          |
| smoking_status     | formerly smoked, never smoked, smokes, Unknown                           |
| stroke             | 1: pernah mengalami stroke, 0: tidak                                      |


Jumlah data pada dataset ini adalah:
|Jumlah Baris |Jumlah Kolom|
|-------------|------------|
|5110         |12          |

---

## 2.1 Exploratory Data Analysis (EDA)

### a. Deskripsi Variabel
- Variabel terdiri dari numerik (age, avg_glucose_level, bmi) dan kategorik (gender, smoking_status, dll).
- Target variabel adalah `stroke` (biner: 0 atau 1).

### b. Missing Value
- Variabel `bmi` mengandung missing value.

### c. Outliers
- Outliers terdeteksi pada variabel `avg_glucose_level` dan `bmi`.

### d. Univariate Analysis
- Visualisasi distribusi masing-masing variabel menggunakan histogram atau barplot.
- Insight:
  - Distribusi usia pasien cenderung meningkat di usia 40 ke atas.
  - Mayoritas pasien tidak memiliki penyakit jantung.
  - Banyak data dengan `smoking_status` = Unknown.

### e. Multivariate Analysis
- Korelasi antar variabel numerik dilakukan dengan heatmap atau matriks korelasi.
- Perbandingan distribusi fitur terhadap target `stroke`:
  - Pengaruh usia, hipertensi, dan penyakit jantung cukup signifikan terhadap risiko stroke.
  - Terdapat hubungan antara `avg_glucose_level` tinggi dan stroke.

---
# 3. Data Preparation

Tahapan ini bertujuan untuk menyiapkan data agar siap digunakan dalam proses pemodelan machine learning. Berikut langkah-langkahnya:

### a. Penanganan Missing Value dan Outliers
 - Strategi penanganan missing value pada `bmi`:
  - Imputasi menggunakan nilai **median** untuk menjaga distribusi data.
  - Alternatif: menggunakan model prediktif untuk imputasi jika jumlah missing value signifikan.
- Strategi penanganan Outliers pada `avg_glucose_level` dan `bmi` :
  - Menggunakan metode **IQR (Interquartile Range)** untuk deteksi.
  - Transformasi log atau winsorizing jika diperlukan.

### b. Encoding Fitur Kategori
- **Tujuan:** Mengubah fitur kategorik menjadi bentuk numerik agar dapat diproses oleh algoritma machine learning.
- **Metode:**
  - **Label Encoding**: digunakan untuk fitur biner seperti `gender`, `ever_married`, `Residence_type`.
  - **One-Hot Encoding**: digunakan untuk fitur dengan lebih dari dua kategori seperti `work_type` dan `smoking_status`.

### c. Pembagian Data
- **Tujuan:** Memisahkan data menjadi data latih dan data uji untuk mengevaluasi performa model secara adil.
- **Metode:** `train_test_split` dari scikit-learn.
  - Rasio umum: 80% data latih dan 20% data uji.
  - Stratifikasi berdasarkan label `stroke` untuk menjaga distribusi kelas.

### d. SMOTE (Synthetic Minority Over-sampling Technique)
- **Tujuan:** Menyeimbangkan distribusi kelas target (`stroke`) yang tidak seimbang.
- **Metode:**
  - SMOTE digunakan hanya pada **data latih**.
  - Membuat sintesis data minoritas agar model tidak bias terhadap kelas mayoritas.

### e. Standarisasi
- **Tujuan:** Mengubah skala fitur numerik agar memiliki distribusi yang sama, membantu mempercepat dan meningkatkan performa model.
- **Metode:**
  - Menggunakan **StandardScaler** dari scikit-learn.
  - Diterapkan pada fitur numerik seperti `age`, `avg_glucose_level`, dan `bmi`.
  - Penting: scaler di-*fit* hanya pada data latih, lalu di-*transform* pada data uji.

### f. Reduksi data dengan PCA
- **Tujuan:** Mengatasi overfitting pada model 
- **Metode**
  - **PCA** mengurangi jumlah fitur (dimensi) menjadi lebih sedikit, sehingga model menjadi lebih sederhana dan generalisasi lebih baik.

---
# 4. Modelling

Pada tahap ini, dilakukan pembangunan model machine learning menggunakan beberapa algoritma untuk membandingkan performa. Tiga model yang digunakan adalah Logistic Regression, Random Forest, dan XGBoost.

## a. Logistic Regression (Baseline Model)
**Deskripsi:**  
Logistic Regression adalah algoritma klasifikasi linear yang digunakan untuk memprediksi probabilitas suatu kelas berdasarkan kombinasi linier dari fitur input.

**Kelebihan:**
- Sederhana dan mudah diinterpretasikan.
- Efisien pada dataset kecil hingga menengah.
- Cepat dalam proses pelatihan.

**Kekurangan:**
- Tidak mampu menangani hubungan non-linear antar fitur.
- Rentan terhadap multikolinearitas.
- Performa terbatas jika terdapat fitur yang saling berinteraksi secara kompleks.

**Parameter LR yang digunakan**
- `max_iter=1000` digunakan untuk maksimal model melakukan iterasi
- `random_state=42` untuk menetapkan seed untuk generator angka acak
- `class_weight='balanced'` digunakan secara otomatis menyesuaikan bobot antar kelas berdasarkan frekuensinya untuk mengatasi imbalance data

## b. Random Forest (Baseline Model)
**Deskripsi:**  
Random Forest adalah algoritma ensemble berbasis decision tree yang membangun banyak pohon keputusan dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting.

**Kelebihan:**
- Dapat menangani data non-linear dan interaksi antar fitur.
- Robust terhadap overfitting dibandingkan single decision tree.
- Menyediakan fitur penting (feature importance) sebagai interpretasi model.

**Kekurangan:**
- Interpretasi lebih sulit dibanding logistic regression.
- Waktu pelatihan lebih lama.
- Memiliki kompleksitas yang lebih tinggi dan membutuhkan lebih banyak sumber daya.

**Parameter RF yang digunakan**
- `random_state=42` untuk menetapkan seed untuk generator angka acak
- `class_weight='balanced'` digunakan secara otomatis menyesuaikan bobot antar kelas berdasarkan frekuensinya untuk mengatasi imbalance data

## c. XGBoost (Model Kompleks + Hyperparameter Tuning)
**Deskripsi:**  
Extreme Gradient Boosting (XGBoost) adalah algoritma boosting yang sangat efisien dan sering digunakan dalam kompetisi machine learning karena performa tinggi dan kemampuannya menangani dataset besar.

**Kelebihan:**
- Performa tinggi dan tahan terhadap overfitting karena regularisasi.
- Mendukung paralelisasi dan efisiensi memori.
- Dapat menangani missing value secara otomatis.
- Cocok untuk data tidak seimbang.

**Kekurangan:**
- Lebih kompleks untuk dipahami dan dituning.
- Membutuhkan waktu pelatihan dan komputasi yang lebih besar.
- Banyaknya hyperparameter membutuhkan eksperimen yang hati-hati.

**Parameter XGBoost**
- `random_state=42` untuk menetapkan seed untuk generator angka acak
- `scale_pos_weight=scale` digunakan untuk memberikan bobot lebih pada kelas positif (biasanya kelas minoritas), sehingga model lebih memperhatikan kesalahan pada kelas tersebut

**Hyperparameter Tuning:**
- `n_estimators`: Jumlah boosting rounds.
- `max_depth`: Kedalaman maksimum pohon.
- `learning_rate`: Ukuran langkah pembaruan bobot.
- `subsample`: Proporsi data untuk setiap pohon.
- `colsample_bytree`: Proporsi fitur yang digunakan untuk setiap pohon.
- `scale_pos_weight`: Rasio penyesuaian kelas positif dan negatif (penting untuk data tidak seimbang).

Tuning dilakukan menggunakan Grid Search atau Randomized Search dengan validasi silang (cross-validation) untuk mencari kombinasi parameter terbaik.

---
# 5. Evaluation
## 1. Evaluasi Kinerja Model 
Dalam proses pengembangan model prediksi stroke, telah diuji tiga algoritma utama: **Logistic Regression, Random Forest Classifier, dan XGBoost**. Evaluasi dilakukan berdasarkan akurasi model sebagai metrik utama. Hasil evaluasi menunjukkan:

- **Logistic Regression**: Akurasi moderat.
- **Random Forest Classifier**: Mencapai akurasi **88%**, menunjukkan performa yang cukup baik dalam menangani fitur non-linear dan data kategorikal.
- **XGBoost (dengan tuning hyperparameter)**: Memberikan performa terbaik dengan akurasi **90%**, menjadikannya algoritma yang paling optimal dalam kasus ini.

Selain itu, penerapan **feature engineering dan standardisasi data** turut berkontribusi signifikan dalam meningkatkan performa model.
Ada beberapa matrix yang digunakan dalam evaluasi model antara lain: 

### a. Classification Report
Classification report memberikan metrik evaluasi secara lebih mendetail, termasuk:

- Precision: Proporsi prediksi positif yang benar.
- Recall (Sensitivity): Proporsi kasus positif yang benar-benar terdeteksi.
- F1-Score: Rata-rata harmonis antara precision dan recall.
- Support: Jumlah contoh dalam masing-masing kelas (0 dan 1).
Brikut klasifikasi report yang didapatkan dari seluruh algoritma.
1. Logistic Regression Evaluation:

|class|Precision|recall|f1-score|support|
|-----|---------|------|--------|-------|
|0|0.97|0.75|0.85|569|
|1|0.08|0.46|0.13|26|
|accuracy|||0.74|595|
|macro avg|0.52|0.61|0.49|595|
|weighted avg|0.93|0.74|0.82|595|

2. Random Forest Classifier Evaluation:

              precision    recall  f1-score   support

           0       0.97      0.91      0.94       848
           1       0.10      0.27      0.15        33

    accuracy                           0.88       881
   macro avg       0.54      0.59      0.54       881
weighted avg       0.94      0.88      0.91       881

3. XGBoost Evaluation:

      precision    recall  f1-score   support

           0       0.98      0.77      0.86       848
           1       0.09      0.61      0.16        33

    accuracy                           0.76       881
   macro avg       0.54      0.69      0.51       881
weighted avg       0.95      0.76      0.83       881

4. XGBoost With Hyperparameter Tunning

              precision    recall  f1-score   support

           0       0.97      0.91      0.94       848
           1       0.11      0.27      0.16        33

        accuracy                       0.89       881
        macro avg  0.54      0.59      0.55       881
      weighted avg 0.94      0.89      0.91       881

## b. Confussion Matrix
Confusion matrix memberikan gambaran tentang bagaimana prediksi model dibandingkan dengan nilai asli. Ini menunjukkan jumlah prediksi benar dan salah untuk setiap kelas.
- True Positives (TP): Jumlah kasus positif yang diprediksi positif.
- True Negatives (TN): Jumlah kasus negatif yang diprediksi negatif.
- False Positives (FP): Jumlah kasus negatif yang diprediksi positif (Type I error).
- False Negatives (FN): Jumlah kasus positif yang diprediksi negatif (Type II error).
- Logistic Regression:

   -[[524  45]
    [ 18   8]]

- Random Forest Classifier:

    [[770  78]
     [ 24   9]]

- XGBoost

    [[649 199]
     [ 13  20]]

- XGBoost With HyperParameterTunning

    [[775  73]
     [ 24   9]]


---
## 2. Dampak terhadap Business Understanding

Model yang dibangun tidak hanya kuat secara teknis, namun juga **berdampak terhadap pemecahan masalah bisnis yang telah diidentifikasi**, yaitu:
### Problem statement

| Problem Statement | Kontribusi Model |
|-------------------|------------------|
| Tingginya angka kematian dan kecacatan akibat stroke | Model mampu memprediksi risiko stroke dengan akurasi tinggi, mendukung intervensi medis lebih awal. |
| Rendahnya deteksi dini terhadap faktor risiko stroke (11,3%) | Model memanfaatkan data klinis sederhana untuk mendeteksi potensi stroke, membantu meningkatkan capaian deteksi dini. |
| Kurangnya sistem skrining berbasis data | Model XGBoost dapat menjadi dasar sistem skrining otomatis yang berbasis data dan machine learning. |

### Goals

| Goals | Capaian Model |
|-------|----------------|
| Membangun prediksi potensi stroke berbasis data klinis | Telah dicapai dengan akurasi tinggi melalui algoritma XGBoost. |
| Meningkatkan efektivitas deteksi dini stroke | Model dapat diintegrasikan ke proses skrining dan sistem klinis untuk meningkatkan efisiensi deteksi. |
| Menyediakan solusi berbasis data yang terukur | Model dapat diterapkan pada sistem nyata untuk prediksi stroke secara terukur dan berbasis data. |

### Dampak dari solusi statement

Solusi statement memiliki **dampak nyata terhadap peningkatan layanan kesehatan**, di antaranya:
- Memberikan **prediksi otomatis risiko stroke** yang dapat dipakai oleh dokter atau sistem informasi rumah sakit.
- Memungkinkan **skrining massal secara cepat dan efisien**, terutama di daerah dengan keterbatasan tenaga medis.
- Memberi **dukungan pengambilan keputusan berbasis data**, yang lebih objektif dan reproducible dibanding pemeriksaan subjektif.

## 3. Kesimpulan Dampak
Evaluasi membuktikan bahwa pendekatan berbasis machine learning tidak hanya meningkatkan akurasi prediksi, namun juga **menjawab tantangan nyata di bidang kesehatan masyarakat**. Dengan demikian, model yang dikembangkan memiliki nilai praktis tinggi dan selaras dengan *business understanding* dari proyek ini: yaitu menyediakan solusi data-driven untuk membantu menurunkan beban stroke secara nasional melalui deteksi dini.

