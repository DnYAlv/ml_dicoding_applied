# Laporan Proyek Machine Learning - Denny Alvito Ginting

## Domain Proyek
Pencemaran udara merupakan pencemaran udara yang terjadi di dalam atmosphere yang disebabkan oleh berbagai macam unsur yang berbahaya. [1] Contoh unsur-unsur tersebut atau lebih dikenal sebagai polutan antara lain karbon monoksida (CO), Nitrogen dioksida (No2), karbon dioksida (CO2), chlorofluorocarbon (CFC), sulfur dioksida(So2), Hidrokarbon (HC), Benda Partikulat, dan timah (Pb). Masuknya unsur-unsur tersebut kedalam atmosphere dapat disebabkan oleh 2 faktor, yaitu faktor manusia maupun faktor alam. 

Kualitas udara yang terus menurut memiliki keterkaitan yang erat dengan kesehatan manusia. Studi membuktikan bahwa polutan yang terdapat pada atmosphere menyebabkan 1.2% dari kematian di dunia, dan 50% dari kematian tersebut terjadi di negara berkembang. [2] Menurut website IQAir yang diakses pada tanggal 23 Juni 2022, Jakarta yang merupakan ibu kota Indonesia merupakan kota yang memiliki polusi udara terburuk ke 4 di dunia. Hal tersebut menjadi kekhawatiran bagi warga negara Indonesia yang tinggal di Jakarta.

Kualitas udara yang memburuk di Jakarta berasal dari berbagai macam kegiatan manusia. [3] Salah satu penyebab utama yaitu tingginya konsentrasi PM2.5. [3] Menurut Plt Deputi Bidang Klimatologi BMKG Urip Haryoko, faktor utama yang mempengaruhi konsentrasi PM2.5 di Jakarta berasal dari emisi dari transportasi dan industri.[3] Selain itu, pergerakan udara dan kadar uap air di udara juga mempengaruhi konsentrasi PM2.5.

Berdasarkan data serta informasi di atas, polutan udara memiliki keterkaitan yang sangat erat dengan kesehatan manusia. Maka dari itu, pengukuran polutan udara serta pemantauan kualitas udara berperan penting terhadap kesehatan masyarakat di kota tersebut. Dengan data-data yang didapatkan dari pengukuran polutan, pemerintah dapat memprediksi kualitas udara yang terjadi di masa yang akan datang. Hal tersebut dilakukan agar dapat memberikan peringatan dini dan masyarakat dapat mengambil keputusan terkait hal tersebut. Dengan data tersebut, pemerintah juga dapat melakukan aksi pencegahan dan meminimalisasi korban yang terjadi akibat polutan udara.

## Business Understanding

### Problem Statements
- Bagaimana membangun model prediksi kualitas udara di DKI Jakarta dengan menggunakan metode Logistic Regression, Naive Bayes dan K Nearest Neighbour (KNN)?
- Metode manakah yang memiliki akurasi yang paling tinggi?

### Goals
Penelitian ini bertujuan untuk:
- Membangun dan mencari model yang terbaik untuk memprediksi kualitas udara di Jakarta dengan membandingkan tiga metode yaitu Logistic Regression, Naive Bayes dan K Nearest Neighbour (KNN).
- Dapat membuat model yang dapat memprediksi kualitas udara di DKI Jakarta yang dapat memberikan peringatan dini.
- Dapat memberikan informasi kepada masyarakat luas tentang kualitas udara sehingga masyarakat dan pemerintah dapat mengambil tindakan untuk memperbaiki atau menjaga kualitas udara di Indonesia, khususnya di DKI Jakarta.

## Data Understanding
Penelitian ini menggunakan data dari Jakarta Open Data mengenai SPKU Pencemaran Udara di DKI Jakarta tahun 2021 yang bisa diakses pada link berikut:

https://data.jakarta.go.id/dataset/indeks-standar-pencemaran-udara-ispu-tahun-2021

Pada data yang kami ambil untuk penelitian ini, terdapat dua jenis data yaitu data kualitatif dan data kuantitatif.

- Data Kualitatif
Data Kualitatif merupakan data yang berbentuk kata - kata dan pada data ini, variabel yang berupa kualitatif yaitu tanggal, stasiun, jenis critical, dan variabel target “categori” mengenai kualitas udara.

- Data Kuantitatif
Data Kuantitatif merupakan data yang berbentuk angka dan dapat dihitung secara statistik dan matematis. Pada data ini, terdapat variabel yang termasuk data kuantitatif yaitu pm10, pm25, so2, co, o3, no2, dan max.

Kolom yang digunakan pada dataset ini yaitu:
| Kolom | Deskripsi |
|-------|-----------|
| tanggal | Tanggal pengukuran kualitas udara |
| stasiun | Lokasi pengukuran di stasiun yang ada |
| pm10 | Partikulat 10 mikrometer sebagai salah satu parameter indikator |
| pm25 | Partikulat 25 mikrometer sebagai salah satu parameter indikator |
| so2 | Kadar Sulfida yang merupakan salah satu parameter indikator |
| co | Kadar karbon monoksida |
| o3 | Kadar ozon |
| no2 | Kadar nitrogran dioksida |
| max | Nilai ukur paling tinggi dari seluruh parameter yang diukur pada waktu yang sama |
| critical | Parameter indikator yang hasil pengukurannya paling tinggi |
| categori | Kategori yang menentukan kualitas udara |

Dataset ini dikumpulkan dari berbagai data yang ada selama tahun 2021, dan di penelitian ini, dataset yang digunakan adalah dataset dari bulan Januari 2021 hingga Desember 2021 dengan jumlah data sebesar:

| Kolom | Jumlah Data |
|-------|-------------|
| 11    | 1825        |

## Data Preparation
Data yang telah dikumpulkan memiliki 1825 record dan 11 atribut termasuk variabel target. Setelah data dikumpulkan, maka data akan dianalisis untuk mengetahui dan mempelajari atribut yang tersedia. Pada tahap ini, dilakukan juga preparasi data untuk memastikan bahwa data tersebut layak untuk diteliti.

- **Data Cleaning**

Pada tahap ini, data yang dikumpulkan akan dibersihkan terlebih dahulu, dimana di data ini terdapat baris data yang memiliki isi tidak jelas yang kemudian akan diubah menjadi nilai NaN agar tidak membingungkan distribusi data. Selain itu juga, atribut - atribut juga akan ditransformasi sesuai dengan tipe datanya seperti contoh, terdapat beberapa atribut yang seharusnya bertipe data float seperti pm10, pm25, so2, co, o3, no2, dan max tetapi secara default bertipe data string.

- **Mengatasi Missing Value**

Pada tahap ini, terdapat beberapa atribut yang memiliki nilai kosong seperti variabel pm10, pm25, so2, co, o3, no2, dan critical. Untuk mengatasi kehilangan nilai pada atribut ini, kami memakai model berupa KNNImputer dari library Scikit-Learn untuk mengisi nilai kosong tersebut berdasarkan nilai rata - rata neighbor yang dimiliki oleh missing value tersebut. Selain itu, pada variabel target, terdapat label berupa “TIDAK ADA DATA” yang menunjukkan bahwa data tersebut tidak bisa dipakai karena tidak terdapat informasi mengenai kualitas udara di data tersebut, sehingga pada tahap tersebut, kami menghilangkan sampel data yang memiliki variabel target berupa “TIDAK ADA DATA”. Pada tahap ini, data berkurang hingga menjadi 1808 record. 

- **Feature Scaling dan Transformation**

Pada tahap ini, atribut - atribut numerik seperti max, pm10, pm25, so2, co, o3, dan no2 memiliki skala nilai yang berbeda antar atribut dan bisa menimbulkan bias pada saat memasangkan data ini kepada model machine learning. 
![image](https://user-images.githubusercontent.com/76100096/205822574-e736074b-6a7b-45e3-9935-61c4f6bb67a2.png)
![image](https://user-images.githubusercontent.com/76100096/205822695-a8262659-9e8e-4be7-8b13-4298e69a1ad8.png)
![image](https://user-images.githubusercontent.com/76100096/205822719-40fab4bb-cc4d-4323-822a-128ecccdd904.png)
![image](https://user-images.githubusercontent.com/76100096/205822728-c92810e7-bcdd-4e7e-be50-63016f4d054e.png)

Dari informasi ini, kami melakukan tahap normalisasi menggunakan MinMaxScaler dari library Scikit-Learn dengan tujuan untuk mengubah skala data tersebut menjadi skala antara 0 hingga 1 agar keseluruhan atribut memiliki distribusi dengan skala angka yang sama. Selain itu, variabel seperti tanggal akan kami ubah bentuknya dari bentuk tanggal menjadi bentuk bulan (mis. 2021 - 01 - 01, menjadi Januari), dan setelah itu, untuk membuat kategori data ini tidak terlalu banyak, kami mentransformasikan bentuk bulan tersebut berdasarkan kuarter tahun, sehingga terdapat 3 kuarter (Januari - April (Kuarter 1), Mei - Agustus (Kuarter 2), September - Desember (Kuarter 3)). 

Selain data numerik, kami juga melakukan transformasi data kategorik dengan menggunakan teknik One Hot Encoding dari library Scikit-Learn dengan menjadikan kategori yang ada dalam sebuah data menjadi atribut tambahan, misal seperti atribut tanggal yang telah diubah menjadi 3 kuarter menandakan atribut ini memiliki 3 kategori, dengan metode One Hot Encoding, kategori ini akan dijadikan 3 atribut baru yang masing - masing atribut berisikan angka antara 1 atau 0, sehingga apabila suatu record memiliki kategori kuarter 1, maka pada atribut baru bernama tanggal_kuarter_1 akan memiliki angka 1, dan sisa atribut lain seperti tanggal_kuarter_2 dan tanggal_kuarter_3 akan berisi angka 0.

- **Data Sampling**
Pada tahap ini, data yang telah dikumpulkan memiliki masalah data yang tidak imbang (variabel target didominasi oleh satu kategori). Untuk mengatasi ini, sesuai dengan tinjauan literatur yang kami lakukan, kami menggunakan teknik SMOTE untuk melakukan sampling agar data tersebut menjadi imbang dari distribusi kategori pada variabel target yang ada.

- **Data Reduction**
Pada tahap ini, kami akan melakukan pengurangan atribut yang berpotensi menjadi data noise dan menghambat performa model nanti. Untuk tahap pertama, pada variabel “critical”, terdapat suatu kategori yang mendominasi jumlah kategori terhadap kategori lain.

![image](https://user-images.githubusercontent.com/76100096/205822897-84acb430-77b1-409c-be18-bc8c1bfd996c.png)

Kategori yang mendominasi ini bisa menimbulkan bias pada data tersebut, sehingga kami menghilangkan atribut ini. Selain itu, kami juga melihat korelasi antar atribut untuk melihat kekuatan relasi antar atribut.

![image](https://user-images.githubusercontent.com/76100096/205822924-d3f3215b-89c8-4697-ad4b-565ec7e947b1.png)

Dari visualisasi korelasi di atas, terdapat beberapa variabel independen yang memiliki korelasi yang tinggi antara satu sama lain. Hal ini dapat menyebabkan masalah multikolinearitas karena suatu atribut bergantung pada atribut yang lain. Dengan demikian, kami akan menghilangkan atribut - atribut yang bermasalah ini yaitu pm10 dan pm25 (korelasi > 0.6).

Dengan demikian atribut tersisa yang akan dipakai pada tahap modelling adalah sebagai berikut:
  - max = Nilai ukur paling tinggi dari seluruh parameter yang diukur dalam waktu yang sama
  - so2 = Sulfida (dalam bentuk SO2) salah satu parameter yang diukur
  - co = Carbon Monoksida salah satu parameter yand diukur
  - o3 = Ozon salah satu parameter yang diukur
  - no2 = Nitrogen dioksida salah satu parameter yang diukur
  - stasiun_DKI2(Kelapa Gading) = lokasi pengukuran stasiun Kelapa Gading (One Hot Encoded)
  - stasiun_DKI3(Jagakarsa) = lokasi pengukuran stasiun Jagakarsa (One Hot Encoded)
  - stasiun_DKI4(Lubang Buaya) = lokasi pengukuran stasiun Lubang Buaya (One Hot Encoded)
  - stasiun_DKI5(Kebon Jeruk) Jakarta Barat = lokasi pengukuran stasiun Kebon Jeruk (One Hot Encoded)
  - tanggal_QUARTER 2 = tanggal berupa bulan di kuarter 2 (One Hot Encoded)
  - tanggal_QUARTER 3 = tanggal berupa bulan di kuarter 3 (One Hot Encoded)
  - categori = berupa kategori kualitas udara (Sehat / Tidak Sehat)

## Modelling
Pada tahap ini, saya akan mencoba melakukan training data ke dalam berbagai jenis model Machine Learning seperti Logistic Regression dengan optimasi berupa Gradient Descent, model Naive Bayes, dan model K - Nearest Neighbor. Dari ketiga model ini kami akan membandingkan performa model tersebut untuk mencari performa yang terbaik dalam mengklasifikasikan data yang telah dikumpulkan tersebut. 

Pada model Logistic Regression, proses training dilakukan dengan mencari garis regresi terbaik yang kemudian akan dipasangkan ke fungsi aktivasi berupa sigmoid sehingga bisa mengklasifikasikan kategori variabel target antara 1 atau 0. Proses training ini kemudian dioptimasi dengan Gradient Descent yaitu melakukan update Weight dan Bias dengan memanfaatkan turunan terhadap Weight dan Bias persamaan regresi yang dilakukan berkali - kali sesuai iterasi yang ditentukan (Epochs) agar mendapatkan loss terendah dan akurasi model tertinggi (Model tergeneralisasi).

Pada model Naive Bayes, proses training dilakukan dengan menggunakan teorema Bayes yang bekerja dengan menggunakan probabilitas yang diambil dengan memanfaatkan nilai rata - rata dan varians sebuah atribut yang dikategorikan sesuai dengan kategori variabel target. Hasil probabilitas yang ada ini kemudian dilanjutkan dengan mengambil probabilitas tertinggi dari tiap kelas pada variabel target, dengan demikian akan didapatkan kategori prediksi pada variabel target.

Pada model K - Nearest Neighbors, tidak terdapat proses training yang spesifik, melainkan dengan memanfaatkan euclidean distance untuk mendapatkan jarak antar suatu data terhadap data lain. Pada model ini, akan dilakukan proses tuning K untuk mendapatkan nilai K terbaik sesuai dengan performa akurasi tertinggi yang didapatkan model tersebut. Setelah didapatkan nilai K terbaik, K ini digunakan untuk mengambil nilai modus kelas dari variabel target pada sampel data tersebut dari tetangga terdekat sebanyak K. Apabila nilai modus berupa 1, berarti didapat prediksi berupa 1, dan begitu juga sebaliknya.

Proses modelling ini kemudian dilanjutkan dengan Cross Validation, yang merupakan tahap iterasi sehingga proses training dalam model bisa mencakupi keseluruhan dataset (agar kita bisa tahu apakah model ini overfit atau tidak) yang kemudian dibandingkan berdasarkan nilai akurasi, precision, recall, dan f1-score untuk mencari tahu model machine learning mana yang memiliki performa terbaik untuk mengklasifikasikan data yang telah diperoleh.

## Evaluation
Pada tahap ini dilakukan perbandingan 3 metode yang berbeda terhadap model yang dibangun untuk memprediksi kualitas udara di DKI Jakarta. Model yang pertama menggunakan metode Logistic Regression, Model yang kedua menggunakan metode K Nearest Neighbour (KNN) dan model yang ketika menggunakan metode Naive Bayes. Dalam eksperimen ini, seluruh model di uji dengan metode Cross Validation (5 - folds) untuk menghindari overfitting. Eksperimen ini bertujuan untuk mencari model yang memiliki performa terbaik.

### Model dengan metode Logistic Regression
Model pertama yang menggunakan metode Logistic Regression dengan Gradient Descent. Disini Gradient Descent menggunakan epochs (pengulangan) sebanyak 500 kali dengan 3 learning rate yang berbeda yaitu 0.01, 0.05 dan 0.1. Penggunaan 3 learning rate yang berbeda bertujuan untuk mencari model yang memiliki hasil terbaik dan didapatkan bahwa learning rate 0.1 merupakan learning rate terbaik. Pada prediksi tanpa menggunakan Cross Validation (5 - folds) didapatkan hasil sebagai berikut:

![image](https://user-images.githubusercontent.com/76100096/205823377-452c7097-7d8b-410a-ba71-80023da03a44.png)

| Metrics | Score |
|---------|-------|
| Accuracy | 1.0 |
| Precision | 1.0 |
| Recall | 1.0 |
| F1 | 1.0 |

**Classification Score**
|         | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
|    0    | 1.00      | 1.00   | 1.00     | 246     |
|    1    | 1.00      | 1.00   | 1.00     | 246     |
| Accuracy|           |        | 1.00     | 492     |
|Macro Avg| 1.00      | 1.00   | 1.00     | 492     |
|Weighted Avg| 1.00      | 1.00   | 1.00     | 492     |

Pada bagian prediksi dengan menggunakan Cross Validation (5 - folds) didapatkan hasil sebagai berikut:

| Metrics (Cross-Validation 5 - Folds) | Score |
|--------------------------------------|-------|
| Accuracy                             | 0.997 |
| Precision                            | 1.0   |
| Recall                               | 0.994 |
| F1                                   | 0.997 |

Dari gambar diatas, diketahui bahwa model pertama yang menggunakan metode Logistic Regression memiliki performa yang hampir sempurna.

### Model dengan metode KNN
Model kedua yang digunakan adalah K Nearest Neighbor (KNN) menggunakan euclidean distance. Pertama-tama dilakukan pencarian nilai K (jumlah neighbor terdekat) paling optimal. Pada percobaan ini, didapatkan nilai K paling optimal adalah 1.

![image](https://user-images.githubusercontent.com/76100096/205835161-3bb5cdc5-b8b6-418b-a8f2-3ccff070361b.png)

Pada prediksi tanpa menggunakan Cross Validation (5 - folds) didapatkan hasil sebagai berikut:

![image](https://user-images.githubusercontent.com/76100096/205823525-6a25faf0-0022-4c55-ab56-e6b6e78b1a14.png)

| Metrics | Score |
|---------|-------|
| Accuracy | 0.9837 |
| Precision | 1.0 |
| Recall | 0.9674 |
| F1 | 0.9837 |

**Classification Score**
|         | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
|    0    | 0.97      | 1.00   | 0.98     | 246     |
|    1    | 1.00      | 0.97   | 0.98     | 246     |
| Accuracy|           |        | 0.98     | 492     |
|Macro Avg| 0.98      | 0.98   | 0.98     | 492     |
|Weighted Avg| 0.98      | 0.98   | 0.98     | 492     |

Pada bagian prediksi menggunakan Cross Validation makamendapatkan nilai akurasi sebagai berikut:

| Metrics (Cross-Validation 5 - Folds) | Score |
|--------------------------------------|-------|
| Accuracy                             | 0.981 |
| Precision                            | 0.995 |
| Recall                               | 0.966 |
| F1                                   | 0.980 |

### Model dengan metode Gaussian Naive Bayes
Model ketiga menggunakan metode Gaussian Naive Bayes. Perhitungan likelihood pada model ini dilakukan dengan bantuan rumus sebagai berikut:

$$
P\left(x_i \mid y\right)=\frac{1}{\sqrt{2 \pi \sigma_y^2}} \exp \left(-\frac{\left(x_i-\mu_y\right)^2}{2 \sigma_y^2}\right)
$$

| Metrics (Cross-Validation 5 - Folds) | Score |
|--------------------------------------|-------|
| Accuracy                             | 0.933 |
| Precision                            | 0.953 |
| Recall                               | 0.910 |
| F1                                   | 0.931 |

Dari hasil diatas, dapat dilihat bahwa performa dari model ini tidak sebaik model yang pertama ataupun kedua.

### Model Final
Setelah percobaan tersebut, maka dapat dilihat bahwa dari ketiga model yang dicoba, model menggunakan Logistic Regression dengan Gradient Descent menghasilkan nilai akurasi tertinggi. Oleh sebab itu, final model yang digunakan pada proyek ini adalah Logistic Regression. Model final ini digunakan untuk melakukan prediksi menggunakan unseen dataset yang telah disiapkan sebelumnya dan mendapatkan hasil sebagai berikut:

| Metrics | Score |
|---------|-------|
| Accuracy | 1.0 |
| Precision | 1.0 |
| Recall | 1.0 |
| F1 | 1.0 |

**Classification Score**
|         | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
|    0    | 1.00      | 1.00   | 1.00     | 301     |
|    1    | 1.00      | 1.00   | 1.00     | 301     |
| Accuracy|           |        | 1.00     | 614     |
|Macro Avg| 1.00      | 1.00   | 1.00     | 614     |
|Weighted Avg| 1.00      | 1.00   | 1.00     | 614     |

Maka, dapat disimpulkan bahwa final model menggunakan Logistic Regression dengan Gradient Descent memiliki performa yang tinggi meski dengan menggunakan dataset yang tidak pernah dilihat.

## Conclusion
Dari berbagai macam model yang telah dibuat dengan Cross Validation (5 - folds),
yaitu Logistic Regression, K Nearest Neighbour (KNN), dan Naive Bayes. Didapatkan bahwa
model akhir menggunakan Logistic Regression memiliki performa yang sangat baik. Model
ini tetap menghasilkan performa yang sangat baik ketika menggunakan Unseen dataset (20%
dari dataset) sehingga dapat dikatakan bahwa model tersebut tidak Overfitting.

**Recommendation**
Alangkah lebih baik apabila data yang didapatkan bisa lebih besar, hal ini bertujuan untuk melihat apakah performa model ini tetap berjalan baik pada data yang banyak

## References
[1] disperkimta, “Sumber Dan Penyebab pencemaran udara,” Dinas Perumahan, Kawasan Permukiman dan Pertanahan, 2019. Available: https://disperkimta.bulelengkab.go.id/informasi/detail/artikel/sumber-dan-penyebab-pencemaran-udara-75

[2] IQAir, “Rangking Indeks kualitas udara dunia,” IQAir, 2022. Available: https://www.iqair.com/id/world-air-quality-ranking 

[3] C. N. N. Indonesia, “BMKG Jelaskan Faktor-Faktor Penyebab Kualitas udara Jakarta memburuk,” nasional, 21-Jun-2022. Available: 
https://www.cnnindonesia.com/nasional/20220621124257-20-811591/bmkg-jelaskan-faktor-faktor-penyebab-kualitas-udara-jakarta-memburuk


