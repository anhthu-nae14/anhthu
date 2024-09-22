Bai 1
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report

# Tải tập dữ liệu (thay thế bằng đường dẫn tệp cục bộ)
dataset = "D:\học máy ứng dụng\Phan Anh Thu - 2274802010872 - lab2\Education.csv"
data = pd.read_csv(dataset)

print("Dữ liệu đầu tiên để kiểm tra:")
print(data.head())

# LabelBinarizer chuyển đổi 'Dương'/'Âm' thành giá trị nhị phân (1 cho 'Dương', 0 cho 'Âm')
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(data['Label']).ravel()  # .ravel() chuyển đổi vector cột thành mảng 1 chiều

# Sử dụng TfidfVectorizer để trích xuất tính năng
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['Text'])  # Chuyển đổi văn bản thành các tính năng số

# Chia dữ liệu thành các tập training và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train và dự đoán bằng Bernoulli Naive Bayes
bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(X_train, y_train)
y_pred_bernoulli = bernoulli_nb.predict(X_test)

# Đánh giá mô hình
print("\nBernoulli Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_bernoulli))
print("Classification Report:")
print(classification_report(y_test, y_pred_bernoulli))

# Train và dự đoán bằng cách sử dụng Multinomial Naive Bayes
multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train, y_train)
y_pred_multinomial = multinomial_nb.predict(X_test)

# Đánh giá Multinomial Naive Bayes model
print("\nMultinomial Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_multinomial))
print("Classification Report:")
print(classification_report(y_test, y_pred_multinomial))

# kết quả
print("\nSo sánh kết quả giữa Bernoulli và Multinomial Naive Bayes:")
print(f"Bernoulli Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_bernoulli)}")
print(f"Multinomial Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_multinomial)}")

bai 2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Tải tập dữ liệu (thay thế bằng đường dẫn tệp cục bộ)
dataset = "D:\học máy ứng dụng\Phan Anh Thu - 2274802010872 - lab2\drug200.csv"  # Replace with the correct path to your file
data = pd.read_csv(dataset)

print("Dữ liệu đầu tiên để kiểm tra:")
print(data.head())

# Tiền xử lý: Chuyển đổi các tính năng phân loại thành các giá trị số
label_encoder = LabelEncoder()

# Mã hóa 'Sex', 'BP', 'Cholesterol' và 'Drug'
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['BP'] = label_encoder.fit_transform(data['BP'])
data['Cholesterol'] = label_encoder.fit_transform(data['Cholesterol'])
data['Drug'] = label_encoder.fit_transform(data['Drug'])

print("\nDữ liệu sau khi mã hóa nhãn:")
print(data.head())

# Chia tách các tính năng (X) và mục tiêu (y)
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]  # Đặc trưng
y = data['Drug']  # Biến mục tiêu: Thuốc

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Áp dụng Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)  # Train cái model
y_pred = gnb.predict(X_test)  # Dự đoán trên tập kiểm tra

# Đánh giá mô hình Gaussian Naive Bayes
print("\nGaussian Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

