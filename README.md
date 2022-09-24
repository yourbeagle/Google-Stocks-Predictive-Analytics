# Laporan Proyek Machine Learning - Wahyu Bagus Wicaksono
---

## Domain Proyek
Domain yang dipilih untuk proyek machine learning ini adalah pada sektor keuangan, dengan judul Predictive Analytics : Predictive Analytics Google Stock Price

### Latar Belakang
Ada dua jenis pasar saham yang ada di dunia ini, yaitu pasar saham ekuitas yang berupa saham atau sebuah surat berharga kemudian ada pasar saham komoditas yang bisa berupa emas, tembaga
nikel kemudian selain dari logam mulia tersebut komoditas juga mencakup seperti gandum, gula, kopi, dan lain-lainnya. Selain dari pasar saham juga, ada juga pasar perdagangan crypto yang
dimana, orang dapat dengan bebas menjual sebuah aset digital, tanpa ada uang asli yang membackup crypto tersebut, karena crypto sendiri berasas pada sebuah rasa saling percaya satu sama lain.

Investor yang memilih berinvestasi terhadap suatu saham tentu saja memiliki tujuan, salah satu tujuan para investor berinvestasi terhadap instrumen saham adalah menerima dividen yang dijanjikan 
oleh sebuah perusahaan tempat dimana berinvestasi yang dihasilkan dari keuntungan perusahaan tersebut. Selain dari dividen tentu saja para investor mengincar Capital Gain, dimana capital gain adalah
selisih antara nilai beli dan jual yang dapat memberikan banyak keuntungan. Tentu saja berinvestasi kepada instrumen saham mengalami perkembangan dari waktu ke waktu yang dimana dulu kita harus ke pasar
bursa untuk membeli sebuah lot, sekarang kita bisa membeli sebuah saham hanya dengan menekan tombol di handphone kita. Untuk mempermudah menebak harga suatu saham kita bisa menggunakan teknik forecasting.

Forecasting merupakan sebuah teknik untuk meramalkan atau memprediksi keadaan dimasa yang akan datang dengan menggunakan data-data yang telah ada di masa lalu. Dengan menggunakan data yang telah ada di masa lalu
pasti akan terbentuk suatu pola dan kecendurangan data, yang kemudian dapat kita formulasikan dalam suatu rumus yang dapat memprediksi data yang akan datang.

## Business Understanding


### Problem Statements

Berdasarkan latar belakang diatas, maka masalah yang dapat saya simpulkan adalah sebagai berikut :
- Bagaimana menganalisa harga dari saham google dimasa depan ?
- Bagaimana cara mengolah data agar dapat dilatih oleh model dengan baik ?
- Bagaimana cara membangun sebuah model yang dapat memprediksi dengan baik ?

### Goals

Tujuan proyek ini dibuat adalah sebagai berikut :
- Dapat memprediksi harga dari saham Google dengan akurat menggunakan machine learning.
- Membantu para investor untuk menentukan harga pembelian pada saham Google.
- Melakukan analisa dan pengolahan data yang optimal dan dapat diterima dengan baik oleh model machine learning.


### Solution statements
Solusi yang dapat dilakukan agar tujuan terpenuhi adalah : 
- Melakukan analisa dan pengolahan pada data dengan mengvisualisasikan data agar dapat mudah untuk dicernah oleh manusia dan mendapatkan insight tentang bagaimana data tersebut diolah. Berikut adalah analisa yang dapat dilakukan :
  - Menangani Missing Value pada data dengan menghapus data tersebut.
  - Menangani outlier pada data dengan menggunakan Metode IQR
  - Melakukan normalisasi pada data
  - Membuat model regresi untuk memprediksi harga yang akan datang.
  
- Berikut adalah algoritma yang digunakan pada proyek ini :
  - K-Neares Neighbors
  - Gradient Boosting Regression
  - Random Forest Regression

- Menggunakan Hyperparameter dan GridSearch juga membantu kita untuk menemukan sebuah parameter terbaik yang dapat digunakan pada suatu model

## Data Understanding


Dataset yang digunakan pada proyek ini adalah : [Google Stock Price (All Time)](https://www.kaggle.com/datasets/akpmpr/google-stock-price-all-time)

Dataset ini memiliki kolom seperti dataset price stok pada umumnya yaitu 7 kolom ["Date","Open","High","Low","Close","Volume","Adj Close"], pada dataset ini tidak memiliki
missing value dan berikut adalah penjelasan dari setiap kolom yang tersedia :
  - Date : Tanggal perdagangan berlangsung
  - Open : Harga pembukaan pada tanggal perdangangan berlangsung
  - High : Harga tertinggi pada tanggal perdangangan berlangsung
  - Low : Harga terendah pada tanggal perdangangan berlangsung
  - Close : Harga terakhir pada saat perdangan pada hari itu di tutup
  - Adj Close : Harga penutupan pada hari tersebut setelah disesuaikan
  - Volume : Volume transaksi yang terjadi pada tanggal perdagangan berlangsung

## Exploratory Data Analysis
Sebelum melakukan pemrosesan pada sebuah data, kita bisa mengeksplor data tersebut untuk mencari sebuah kolerasi antara data, mencari outlier, dan melakukan analisis
Univariate Analysis dan Multivariate Analysis

- Menangani Outlier
<br>Berikut adalah data numerik jika divisualisasikan, hanya Volume yang memiliki outlier.
<image src="https://github.com/yourbeagle/Google-Stocks-Predictive-Analytics/blob/master/images/iqr1.png" width=600/>
 Dan berikut adalah data numerik jika divisualisasikan, jika sudah menggunakan metode IQR yaitu dengan menghapus data yang berada diluar IQR yaitu antara 25% dan 75%. Setelah menggunakan metode tersebut, didapatkan sampel data sebanyak 3848 dan 6 kolom dan berikut adalah gambarnya.
<br>
<image src="https://github.com/yourbeagle/Google-Stocks-Predictive-Analytics/blob/master/images/iqr2.png" width=600/>
<br>

- Univariate Analysis
<br>Fitur yang akan diprediksi pada kasus ini terfokus kepada fitur 'Close','High','Open','Close'
<image src="https://github.com/yourbeagle/Google-Stocks-Predictive-Analytics/blob/master/images/unvariate.png" width=600 />
<br>

- Multivariate Analysis
<br>Dapat kita simpulkan bahwa fitur 'Close' memiliki terkaitan antara fitur 'Open', 'Low', 'High', dan juga 'Adj Close' namun tidak dengan fitur 'Volume'
<image src="https://github.com/yourbeagle/Google-Stocks-Predictive-Analytics/blob/master/images/multivariate.png" width=600/>
<br>

- Colleration Matrix
<br>Untuk melihat kolerasi antara data kita dapat memvisualisasikan menggunakan heatmap, dapat kita lihat semua data memiliki korelasi antara lain kecuali dengan volume
<image src="https://github.com/yourbeagle/Google-Stocks-Predictive-Analytics/blob/master/images/correlecation.png" width=600/>
<br>

## Data Preparation

Berikut ini adalah tahapan dalam menyiapkan data sebelum melakukan modeling :
### Melakukan Penangan Missing Value
Pada kasus saya, saya tidak memiliki missing value, namun kita dapat mengatasi missing value ini dengan menghapus value tersebut

### Membagi Dataset
Kita bisa membagi dataset menjadi dua yaitu train data dan test data, train data digunakan sebagai training model dan test data digunakan sebagai validasi apakah model tersebut sudah akurat atau belum akurat. Rasio yang saya gunakan pada proyek ini adalah 8:2, 8 adalah train data dan 2 adalah test data. Dengan pembagian tersebut didapatkan jumlah sampel train data yaitu 3078 sampel dan sampel test data yaitu 770 sampel dengan total data yang digunakan adalah 3848. Untuk melakukan splitting data kita bisa menggunakan library train_test_split dari scikit-learn.

### Menghapus atau Mengdrop Kolom yang tidak digunakan
Disini kita akan menghapus kolom Volume dan Adj Close karena kita tidak memerlukan kedua kolom tersebut.

### Normalisasi Data
Melakukan normalisasi data agar data lebih mudah diproses oleh model machine learning, pada kali ini saya menggunakan MinMaxScaler. MinMaxScaler berfungsi untuk mentransformasi kedalam bentuk 0 hingga 1.

## Modeling

Model yang akan digunakan proyek kali ini yaitu Gradient Boosting, K-Nearest Neighbors, dan Random Forest.

### Gradient Boosting
Gradient Boosting adalah sebuah algoritma pada machine learning yang menggunakan teknik ensembel learning dari decision tree untuk meprediksi nilai. Gradient Boostring mampu menangani data dan pattern yang kompleks. Untuk parameter yang digunakan pada model ini ada 3 yaitu:
  - learning_rate
  - n_estimators
  - criterion

#### Kelebihan
  - Hasil pemodelan lebih akurat
  - Stabil
#### Kekurangan
  - Waktu pemrosesan yang cukup lama
  - Tingkat kesulitan yang tinggi dalam pemilihan model

### K-Nearest Neighbors
K-Nearest Neighbors adalah sebuah algoritma pada machine learning yang bekerja dengan mengklasifikasikan data baru menggunakan kesamaan fitur untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam sebuah set pelatihan. Untuk parameter yang digunakan pada model ini ada 1 yaitu :
  - n_neighbors

#### Kelebihan
  - Mudah diimplementasikan
  - Efektif terhadap data yang besar
#### Kekurangan
  - Perlu menentukan nilai parameter K
  - Rentan terhadap variabel yang tidak informatif

### Random Forest
Random Forest adalah sebuah algoritma pada machine learning yang bekerja menggunakan teknik ensembel learning untuk memprediksi suatu nilai. Random Forest dapat bekerja secara bersamaan dalam satu waktu, sehingga tingkat keberhasilan menjadi lebih tinggi. Untuk parameter yang digunakan pada model ini ada 2 yaitu :
  - n_estimators
  - criterion

#### Kelebihan
  - Dapat mengatasi training data yang besar secara efisien
  - Dapat menangani missing values
#### Kekurangan
  - Kompleksitas yang tinggi
  - Waktu pemrosesan yang lama

Berdasarkan uraian di atas serta pada saat proses modeling dan evaluasi, menurut saya kedua algoritma bekerja dengan cukup baik dalam memprediksi adalah model Gradient Boosting. Hal ini dapat di lihat dari nilai akurasi yang cukup tinggi dibandingkan dengan kedua model yang lain.

## Evaluation

Untuk evaluasi pada machine learning model ini, metrik yang saya gunakan adalah mean squared error (mse). Dimana metrik ini menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.

<image src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-50d568506216f6ab6402504298c570e2_l3.svg" width=500 />
<br>

Berikut adalah hasil akurasi dari beberapa model yang dipakai :
<br>
<image src="https://github.com/yourbeagle/Google-Stocks-Predictive-Analytics/blob/master/images/modeltrain.png" width=600/>
<br>
<image src="https://github.com/yourbeagle/Google-Stocks-Predictive-Analytics/blob/master/images/modelacc.png" width=600/>
<br>

Dapat Dilihat dari gambar di atas semua model bekerja secara optimal namun secara akurasi model Gradient Boosting yang memiliki akurasi paling tinggi, disusul oleh Random Forest dan yang terakhir adalah K-Nearest Neighbors. Terdapat selisih yang sangat kecil diantara 2 model yaitu Gradient Boosting dan Random Forest, namun model Gradient Boosting memiliki nilai yang lebih tinggi.

