import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from features import extract_features

df = pd.read_csv('data/dataset.csv')
X_raw, y = df['sentence'].values, df['quality'].values

tfidf = TfidfVectorizer(max_features=250, ngram_range=(1, 2))
scaler = StandardScaler()

X_struct = scaler.fit_transform([extract_features(t) for t in X_raw])
X_tfidf = tfidf.fit_transform(X_raw).toarray()
X = np.hstack([X_struct, X_tfidf])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_tr, y_tr)
rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42).fit(X_tr, y_tr)
mlp = MLPClassifier((64, 32, 16), max_iter=800, alpha=0.001, random_state=42).fit(X_tr, y_tr)

for name, model in [('Decision Tree', dt), ('Random Forest', rf), ('Neural Network', mlp)]:
    print(f'{name} Accuracy: {model.score(X_te, y_te)*100:.1f}%')

joblib.dump({'dt': dt, 'rf': rf, 'mlp': mlp, 'scaler': scaler, 'tfidf': tfidf}, 'models.pkl')
print("Successfully trained & saved models.pkl")
