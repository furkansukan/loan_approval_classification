import warnings
import joblib
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import streamlit as st
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

# Veriyi yükleme
df = pd.read_csv("loan_data.csv")

# Sayfa Seçimi
st.sidebar.title("Menü")
page = st.sidebar.radio("Sayfa Seçin:", ["Ana Sayfa", "Veri Analizi", "Model Tahmini"])

# Sayfa 1: Ana Sayfa
if page == "Ana Sayfa":
    st.title("Ana Sayfa - Veriye Genel Bakış")

    # Verinin ilk birkaç satırını göster
    st.subheader("Verinin İlk Satırları")
    st.write(df.head())

    # Veri tiplerini değiştirme
    df['person_age'] = df['person_age'].astype('int')

    # Sütunları kategorik ve sayısal olarak ayırma
    cat_cols = [var for var in df.columns if df[var].dtypes == 'object']
    num_cols = [var for var in df.columns if df[var].dtypes != 'object']
    st.session_state.cat_cols = cat_cols
    st.session_state.num_cols = num_cols
    st.subheader("Sütun Türleri")
    st.write(f"Kategorik Sütunlar: {cat_cols}")
    st.write(f"Sayısal Sütunlar: {num_cols}")

    # Kategorik sütunlardaki değer sayımlarını göster
    st.subheader("Kategorik Değişkenlerin Dağılımı")
    for i in cat_cols:
        st.write(f"{i} için Değer Dağılımı:")
        st.write(df[i].value_counts())
        st.write("\n")

# Sayfa 2: Veri Analizi
elif page == "Veri Analizi":
    st.title("Veri Analizi")

    cat_cols = st.session_state.cat_cols
    num_cols = st.session_state.num_cols
    # Kategorik Değişkenlerin Görselleştirilmesi
    def plot_categorical_column(dataframe, column):
        plt.figure(figsize=(7, 7))
        ax = sns.countplot(x=dataframe[column])
        total_count = len(dataframe[column])
        threshold = 0.05 * total_count
        category_counts = dataframe[column].value_counts(normalize=True) * 100
        ax.axhline(threshold, color='red', linestyle='--', label=f'0.05% of total count ({threshold:.0f})')

        for p in ax.patches:
            height = p.get_height()
            percentage = (height / total_count) * 100
            ax.text(p.get_x() + p.get_width() / 2., height + 0.02 * total_count, f'{percentage:.2f}%', ha="center")

        plt.title(f'Label Cardinality for "{column}" Column')
        plt.ylabel('Count')
        plt.xlabel(column)
        plt.tight_layout()

        plt.legend()
        st.pyplot(plt)


    # Kategorik sütunları çiz
    for col in cat_cols:
        plot_categorical_column(df, col)

    # Sayısal Değişkenlerin Dağılımı (Histogram)
    st.subheader("Sayısal Değişkenlerin Dağılımı")
    df[num_cols].hist(bins=30, figsize=(12, 10))
    st.pyplot(plt)

    # Target label distribution (Pie chart)
    label_prop = df['loan_status'].value_counts()
    plt.pie(label_prop.values, labels=['Rejected (0)', 'Approved (1)'], autopct='%.2f')
    plt.title('Target label proportions')
    st.pyplot(plt)

    # Veriyi ölçekleme ve dönüşüm
    st.subheader("Veri Ölçekleme ve Dönüşümler")

    skewed_cols = ['person_age', 'person_income', 'person_emp_exp',
                   'loan_amnt', 'loan_percent_income',
                   'cb_person_cred_hist_length', 'credit_score']

    norm_cols = ['loan_int_rate']
    mms = MinMaxScaler()
    ss = StandardScaler()

    # Sayısal sütunları ölçekle
    df[skewed_cols] = ss.fit_transform(df[skewed_cols])
    df[norm_cols] = mms.fit_transform(df[norm_cols])

    # Eğitim verisinde kategorik değerleri dönüştür
    df['person_education'].replace({
        'High School': 0,
        'Associate': 1,
        'Bachelor': 2,
        'Master': 3,
        'Doctorate': 4
    }, inplace=True)

    gender_mapping = {'male': 0, 'female': 1}
    home_ownership_mapping = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    loan_intent_mapping = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4,
                           'DEBTCONSOLIDATION': 5}
    previous_loan_defaults_mapping = {'No': 0, 'Yes': 1}

    df['person_gender'] = df['person_gender'].map(gender_mapping)
    df['person_home_ownership'] = df['person_home_ownership'].map(home_ownership_mapping)
    df['loan_intent'] = df['loan_intent'].map(loan_intent_mapping)
    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map(previous_loan_defaults_mapping)

    # Outlier trimming işlemi
    trimmer = OutlierTrimmer(capping_method='iqr', tail='right',
                             variables=['person_age', 'person_gender', 'person_education', 'person_income',
                                        'person_emp_exp', 'person_home_ownership', 'loan_amnt',
                                        'loan_intent', 'loan_int_rate', 'loan_percent_income',
                                        'cb_person_cred_hist_length', 'credit_score',
                                        'previous_loan_defaults_on_file'])

    df2 = trimmer.fit_transform(df)
    st.session_state.df2 = df2
    # Korelasyon matrisi
    st.subheader("Korelasyon Matrisi")
    plt.figure(figsize=(15, 8))
    sns.heatmap(df2.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    st.pyplot(plt)




# Sayfa 3: Model Tahminleme
elif page == "Model Tahmini":
    st.title("Model Tahmini")
    # Veriyi ölçekleme ve dönüşüm
    st.subheader("Veri Ölçekleme ve Dönüşümler")

    skewed_cols = ['person_age', 'person_income', 'person_emp_exp',
                   'loan_amnt', 'loan_percent_income',
                   'cb_person_cred_hist_length', 'credit_score']

    norm_cols = ['loan_int_rate']
    mms = MinMaxScaler()
    ss = StandardScaler()

    # Sayısal sütunları ölçekle
    df[skewed_cols] = ss.fit_transform(df[skewed_cols])
    df[norm_cols] = mms.fit_transform(df[norm_cols])

    # Eğitim verisinde kategorik değerleri dönüştür
    df['person_education'].replace({
        'High School': 0,
        'Associate': 1,
        'Bachelor': 2,
        'Master': 3,
        'Doctorate': 4
    }, inplace=True)

    gender_mapping = {'male': 0, 'female': 1}
    home_ownership_mapping = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    loan_intent_mapping = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4,
                           'DEBTCONSOLIDATION': 5}
    previous_loan_defaults_mapping = {'No': 0, 'Yes': 1}

    df['person_gender'] = df['person_gender'].map(gender_mapping)
    df['person_home_ownership'] = df['person_home_ownership'].map(home_ownership_mapping)
    df['loan_intent'] = df['loan_intent'].map(loan_intent_mapping)
    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map(previous_loan_defaults_mapping)

    # Outlier trimming işlemi
    trimmer = OutlierTrimmer(capping_method='iqr', tail='right',
                             variables=['person_age', 'person_gender', 'person_education', 'person_income',
                                        'person_emp_exp', 'person_home_ownership', 'loan_amnt',
                                        'loan_intent', 'loan_int_rate', 'loan_percent_income',
                                        'cb_person_cred_hist_length', 'credit_score',
                                        'previous_loan_defaults_on_file'])

    df2 = trimmer.fit_transform(df)
    threshold = 0.1

    correlation_matrix = df2.corr()
    high_corr_features = correlation_matrix.index[abs(correlation_matrix["loan_status"]) > threshold].tolist()
    high_corr_features.remove("loan_status")
    print(high_corr_features)

    X_selected = df[high_corr_features]
    Y = df["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, Y, test_size=0.2, random_state=42)


    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Modelin doğruluğunu hesapla
    score = model.score(X_train, y_train)

    # Skoru Streamlit'te yazdır
    st.write(f"Logistic Regression Modelin eğitim verisi üzerindeki doğruluk skoru: {score:.4f}")

    Y_pred = model.predict(X_test)
    st.write("Accuracy", (accuracy_score(y_test, Y_pred)))

    model2 = SVC()
    model2.fit(X_train, y_train)

    Y_pred2 = model2.predict(X_test)


    # Skoru Streamlit'te yazdır
    st.write(f"SVC (Support Vector Classifier) Modelin eğitim verisi üzerindeki doğruluk skoru: {score:.4f}")
    st.write("Accuracy", (accuracy_score(y_test, Y_pred)))

    # Confusion matrix hesaplama
    conf_matrix2 = confusion_matrix(y_test, Y_pred2)

    # Streamlit arayüzüne başlık eklemek
    st.title("Confusion Matrix Heatmap")

    # Heatmap için figür ayarlama
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix2, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Negative", "Predicted Positive"],
                yticklabels=["Actual Negative", "Actual Positive"])

    # Etiketler ve başlık ekleme
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Heatmap")

    # Streamlit üzerinde gösterme
    st.pyplot(plt)