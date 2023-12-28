import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# Fungsi untuk mempersiapkan data
def prepare_data(df):
    # Mengubah kolom kategorikal menjadi numerik menggunakan LabelEncoder
    label_encoder = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

    # Hapus kolom 'Target' jika ada dalam dataframe
    if 'Target' in df.columns:
        df = df.drop('Target', axis=1)

    return df


# Fungsi untuk melatih model regresi logistik
def train_logistic_regression(df):
    # Memisahkan fitur (X) dan target (y)
    X = df.drop('Target', axis=1)
    y = df['Target']

    # Splitting data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    # Membuat dan melatih model regresi logistik
    model = LogisticRegression()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X=X_train, y=y_train)

    # Evaluasi model
    accuracy = model.score(X_test, y_test)

    return model, accuracy

# Fungsi untuk melakukan prediksi kelulusan
def predict_graduation_status(data, model):
    # Melakukan pre-processing pada data baru
    prepared_data = prepare_data(data.copy())

    # Prediksi kelulusan menggunakan model
    predictions = model.predict(prepared_data)
    predicted_probabilities = model.predict_proba(prepared_data)

    # Mengembalikan hasil prediksi dan probabilitas prediksi
    return predictions, predicted_probabilities

# Fungsi untuk melakukan prediksi kelulusan berdasarkan file CSV yang diunggah
def predict_graduation_status_from_csv(uploaded_file, model):
    # Membaca file CSV menjadi DataFrame
    data = pd.read_csv(uploaded_file)

    # Memanggil fungsi predict_graduation_status untuk prediksi berdasarkan data CSV yang diunggah
    predictions, probabilities = predict_graduation_status(data, model)

    return predictions, probabilities

# Load dataset
dataAwal = pd.read_csv("Dataset/dataset.csv")
dataFinal = pd.read_csv("Dataset/dataset_final.csv")

# Sidebar
st.sidebar.title('Tugas Akhir AI')
option = st.sidebar.selectbox('Pilih Fitur:', ('Tampilkan Data', 'Visualisasi Data', 'Prediksi Kelulusan', 'Akurasi Model'))

# Main content
st.title('Dashboard: Klasifikasi Siswa Dropout👩🏻‍🎓')

if option == 'Tampilkan Data':
    st.subheader('Dataset')
    st.write(dataFinal)
    st.subheader('Statistik Deskriptif')
    st.write(dataFinal.describe())

elif option == 'Visualisasi Data':
    st.subheader('Visualisasi')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Heatmap for correlation
    st.write('Korelasi antar fitur:')
    plt.figure(figsize=(30, 30))
    sns.heatmap(dataFinal.corr(), annot=True)
    st.pyplot()

    # Pie chart for 'Target' distribution
    st.write('Distribusi Target:')
    plt.figure(figsize=(4, 4))
    dataAwal["Target"].value_counts().plot.pie(autopct="%1.1f%%")
    st.pyplot()

elif option == 'Prediksi Kelulusan':
    st.subheader('Prediksi Kelulusan dari File CSV')
    # Upload file CSV untuk prediksi
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        # Memuat model yang telah dilatih sebelumnya
        trained_model, _ = train_logistic_regression(prepare_data(dataFinal))

        # Prediksi kelulusan berdasarkan file CSV yang diunggah
        predictions, probabilities = predict_graduation_status_from_csv(uploaded_file, trained_model)

        # Tampilkan hasil prediksi
        st.write("Prediksi Kelulusan:", predictions)
        st.write("Probabilitas Prediksi:", probabilities)

elif option == 'Akurasi Model':
    st.subheader('Akurasi Model')

    # Training model regresi logistik dan menampilkan akurasi
    trained_model, accuracy = train_logistic_regression(prepare_data(dataFinal))

    st.write(f'Accuracy Score: `{accuracy}`')

    # Splitting the data
    X = dataFinal.iloc[:, :-1]
    y = dataFinal['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Evaluasi model
    y_pred = trained_model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.write(f'Precision Score: `{precision}`')
    st.write(f'Recall Score: `{recall}`')

    # Confusion Matrix
    st.write('Confusion Matrix:')
    st.write(cm)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=0.3, square=True, cmap='YlGnBu')
    st.pyplot()
