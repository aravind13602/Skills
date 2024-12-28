from flask import Flask, render_template, request
import joblib


role_model = joblib.load('role_category_model.pkl')
func_model = joblib.load('functional_area_model.pkl')
industry_model = joblib.load('industry_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        skills = request.form['skills']
        role, func, industry = predict_fields(skills)
        return render_template('index.html', role=role, func=func, industry=industry)
    return render_template('index.html')


def predict_fields(skills):
    skills_tfidf = vectorizer.transform([skills])
    role_pred = role_model.predict(skills_tfidf)
    func_pred = func_model.predict(skills_tfidf)
    industry_pred = industry_model.predict(skills_tfidf)
    return role_pred[0], func_pred[0], industry_pred[0]


if __name__ == '__main__':
    app.run(debug=True)
