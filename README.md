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
  - Gradient Boostring Regression
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

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

