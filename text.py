import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

file_path = r'C:\Users\aravi\Desktop\Skills\naukri.xlsx'
df = pd.read_excel(file_path)

df = df.dropna(subset=['Key-Skills', 'Role Category', 'Functional Area', 'Industry'])

X = df['Key-Skills']
y_role_category = df['Role Category']
y_functional_area = df['Functional Area']
y_industry = df['Industry']

vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

role_model = RandomForestClassifier(n_estimators=100, random_state=42)
role_model.fit(X_tfidf, y_role_category)

func_model = RandomForestClassifier(n_estimators=100, random_state=42)
func_model.fit(X_tfidf, y_functional_area)

industry_model = RandomForestClassifier(n_estimators=100, random_state=42)
industry_model.fit(X_tfidf, y_industry)

joblib.dump(role_model, 'role_category_model.pkl')
joblib.dump(func_model, 'functional_area_model.pkl')
joblib.dump(industry_model, 'industry_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

def predict_fields(skills):
    skills_tfidf = vectorizer.transform([skills])
    role_pred = role_model.predict(skills_tfidf)
    func_pred = func_model.predict(skills_tfidf)
    industry_pred = industry_model.predict(skills_tfidf)
    return role_pred[0], func_pred[0], industry_pred[0]

new_skills = "Python, Machine Learning, Data Analysis"
role, func, industry = predict_fields(new_skills)
print(f"Predicted Role Category: {role}")
print(f"Predicted Functional Area: {func}")
print(f"Predicted Industry: {industry}")
