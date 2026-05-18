import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class Macrobag(BaseEstimator, ClassifierMixin):
    """
    Macrobag: A resampling learning technique for rare event training.
    """
    def __init__(self, base_estimator, n_bootstrap=100):
        self.base_estimator = base_estimator
        self.n_bootstrap = n_bootstrap
        
    def fit(self, X, y, replace = False):
        # check integrity of sklearn standards
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        # Split the data in the rare (positive) and abundant (negative) class 
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        n_pos = len(pos_idx)
        
        if n_pos == 0:
            raise ValueError("De positieve klasse (1) ontbreekt in de dataset.")
            
        self.estimators_ = []
        
        # The Macro-Bagging loop
        for i in range(self.n_bootstrap):
            chosen_neg_idx = np.random.choice(neg_idx, size=n_pos, replace=replace)
            
            # combine into perfect balanced  50/50 subset
            subset_idx = np.concatenate([pos_idx, chosen_neg_idx])
            X_subset = X[subset_idx]
            y_subset = y[subset_idx]
            
            # 3. Train the micro-estimator (eg. SVM) on the specific subset
            from sklearn.base import clone
            estimator = clone(self.base_estimator)
            estimator.fit(X_subset, y_subset)
            
            self.estimators_.append(estimator)
            
        return self
        
    def predict_proba(self, X):
	# check if the model is already a fitted model
        check_is_fitted(self, ['estimators_', 'classes_'])
        X = check_array(X)
        
        # collect all the responses of all the models and average them. 
        probas = np.array([clf.predict(X) for clf in self.estimators_])
        
        macro_proba = np.mean(probas, axis=0)
        
        return np.vstack([1 - macro_proba, macro_proba]).T

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)

