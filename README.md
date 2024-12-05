# Kredi Onay Sınıflandırma Projesi 🚀

![image](https://github.com/user-attachments/assets/7e77c57c-c13a-4b79-848d-c75d77659a10)


**Kredi Onay Sınıflandırma Projesi**'ne hoş geldiniz! 🎉 Bu proje, kredi başvurusunun onaylanıp onaylanmayacağını tahmin etmek için makine öğrenimi modelleri kullanarak yapılan bir çalışmadır. **Streamlit** kullanarak etkileşimli görselleştirme yapılmış olup, veriler kategorik ve sayısal özellikler, aykırı değerler ve ölçekleme işlemlerini işlemek için ön işleme tabi tutulmuştur.

## 📚 Kullanılan Kütüphaneler

Projede kullanılan kütüphaneler:

```python
import warnings
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import plotly.express as px
from sklearn.svm import SVC
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.naive_bayes import MultinomialNB
from feature_engine.outliers import OutlierTrimmer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
```

- **Veri Ön İşleme**: Aykırı değerlerin işlenmesi, verilerin ölçeklendirilmesi ve kategorik değişkenlerin kodlanması.
- **Model Eğitimi**: Çeşitli modeller eğitildi: Lojistik Regresyon, Destek Vektör Makinesi (SVC), Naive Bayes, XGBoost vb.
- **Değerlendirme**: Model performansını değerlendirmek için doğruluk, karışıklık matrisi ve sınıflandırma raporu gibi metrikler kullanıldı.

## 🎯 Proje Amacı

Bu projenin amacı, kredi başvurularının onaylanıp onaylanmayacağını tahmin etmektir. Gelir, istihdam durumu, kredi puanı gibi çeşitli özelliklere göre başvuru onayı tahmin edilir. Veri seti, modelleri eğitmek için kullanılmış ve performansları karşılaştırılmıştır.

## 🧑‍💻 Projeyi Çalıştırma

1. Depoyu klonlayın:
    ```bash
    git clone https://github.com/yourusername/loan-approval-classification.git
    ```
2. Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```
3. Streamlit uygulamasını çalıştırın:
    ```bash
    streamlit run app.py
    ```

Projenin detaylı adımlarını, veri işleme, model eğitimi ve değerlendirme işlemlerini web uygulamasında da keşfedebilirsiniz. 🚀

## 📊 Ana Özellikler

- **Streamlit** ile etkileşimli kredi onayı tahmin uygulaması.
- **Aykırı değer kesme**, **ölçekleme** ve **kodlama** gibi veri işleme adımları.
- **Lojistik Regresyon**, **Destek Vektör Makinesi** gibi farklı sınıflandırma modellerinin karşılaştırılması.

## 💡 Gelecek Çalışmalar

- Daha iyi performans için **Sinir Ağı** gibi ileri düzey modellerin keşfi.
- Kullanıcı arayüzünün daha etkileşimli hale getirilmesi.
- Kredi onayı için gerçek zamanlı veri tahmini eklenmesi.

## 🌍 Projeyi Online Olarak Keşfedin

Projeyi canlı olarak görmek ve kredi onayı tahminleri için etkileşimli uygulamayı keşfetmek için [Canlı Proje](https://loanapprovalclassification-furkansukan.streamlit.app/) bağlantısını takip edebilirsiniz.


---
##### **İletişim**
Herhangi bir sorunuz veya geri bildiriminiz için aşağıdaki kanallardan ulaşabilirsiniz:
- **Email:** [furkansukan10@gmail.com](furkansukan10@gmail.com)
- **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/furkansukan/)
- **Kaggle:** [Kaggle Profile](https://www.kaggle.com/furkansukan)
- **Website:** [Project Website](https://loanapprovalclassification-furkansukan.streamlit.app/)


[EN]

# Loan Approval Classification Project 🚀

Welcome to the **Loan Approval Classification Project**! 🎉 This project demonstrates the process of predicting loan approval status (approved or not) using machine learning models. It's built with **Streamlit** for interactive visualization, and the data is preprocessed to handle categorical and numerical features, outliers, and scaling.

## 📚 Libraries Used

Here are the libraries used in this project:

```python
import warnings
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import plotly.express as px
from sklearn.svm import SVC
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.naive_bayes import MultinomialNB
from feature_engine.outliers import OutlierTrimmer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
```

- **Data Preprocessing**: We handle outliers, scale data, and encode categorical variables.
- **Model Training**: We train various models, including Logistic Regression, Support Vector Machine (SVC), Naive Bayes, XGBoost, and more.
- **Evaluation**: We use performance metrics such as accuracy, confusion matrix, and classification report to assess model performance.

## 🎯 Project Goal

The goal of this project is to predict whether a loan application will be approved based on several features such as income, employment status, credit score, etc. The dataset is used to train multiple models, and their performance is compared.

## 🧑‍💻 How to Run the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/loan-approval-classification.git
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

You can also explore the detailed steps of the project, including data processing, model training, and evaluation, on the web application. 🚀

## 📊 Key Features

- Interactive loan approval prediction with **Streamlit**.
- Data preprocessing steps, including **outlier trimming**, **scaling**, and **encoding**.
- Comparison of multiple classification models like **Logistic Regression**, **Support Vector Machine**

## 💡 Future Work

- Explore more advanced models like **Neural Networks** for better performance.
- Enhance the user interface for better interactivity.
- Add real-time data prediction for loan approval.

## 🌍 Find the Project Online

Check out the live version of the project and explore the interactive application for loan approval predictions: [Live Project](https://loanapprovalclassification-furkansukan.streamlit.app/)


📞 Contact
If you have any questions or suggestions, feel free to reach out to me:

Feel free to reach out for questions or feedback via:
- **Email:** [furkansukan10@gmail.com](furkansukan10@gmail.com)
- **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/furkansukan/)
- **Kaggle:** [Kaggle Profile](https://www.kaggle.com/furkansukan)
- **Website:** [Project Website](https://loanapprovalclassification-furkansukan.streamlit.app/)
