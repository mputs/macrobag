from macrobag import Macrobag
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np

# Maak een kunstmatige 'rare event' dataset (10 positief, 1000 negatief)

rng = np.random.RandomState(42)

X = rng.randn(1010, 4)
y = np.array([1]*10 + [0]*1000)

# Initialiseer de micro-motor (SVM met kansen ingeschakeld)
base_svm = SVC()

print ("---\nSVM\n---")

# Bouw het macrobag ensemble
model = Macrobag(base_estimator=base_svm, n_bootstrap=50)
model.fit(X, y)

# Dit geeft de ongecalibreerde macro-kans die je daarna in Bayesccal stopt
ongecalibreerde_kansen = model.predict_proba(X)
print(f"SVM: average = {ongecalibreerde_kansen[:,0].mean()}");
print(f"     variance = {ongecalibreerde_kansen[:,0].var()}");

print ("-------------------\nLogistic Regression\n-------------------")
print ("With predict_proba")
model = Macrobag(base_estimator = LogisticRegression(), n_bootstrap = 50);
model.fit(X,y)
ongecalibreerde_kansen = model.predict_proba(X)
print(f"LR: average = {ongecalibreerde_kansen[:,0].mean()}");
print(f"    variance = {ongecalibreerde_kansen[:,0].var()}");

print("With predict")
model = Macrobag(base_estimator = LogisticRegression(), n_bootstrap = 50, predict_proba_if_present = False);
model.fit(X,y)
ongecalibreerde_kansen = model.predict_proba(X)
print(f"LR: average = {ongecalibreerde_kansen[:,0].mean()}");
print(f"    variance = {ongecalibreerde_kansen[:,0].var()}");
