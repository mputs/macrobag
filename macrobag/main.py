import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from warnings import warn

class Macrobag(BaseEstimator, ClassifierMixin):
    """
    Macrobag: A resampling learning technique for rare event training.
    """
    def __init__(self, base_estimator, n_bootstrap=100, threshold = 0.5, random_state = 42, predict_proba_if_present = True):
        self.base_estimator = base_estimator
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold
        self.random_state = random_state
        self.predict_proba_if_present = predict_proba_if_present
        
    def fit(self, X, y, replace = False):
        # check integrity of sklearn standards
        X, y = check_X_y(X, y)

        # extract the classes out of the training set.
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
           raise ValueError("Macrobag needs two classes. Please your trainingset!")
        if len(classes) > 2:
           raise ValueError("Macrobag currently supports binary classification only!")

        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]

        self.classes_ = np.unique(y)
        
        # Split the data in the rare (positive) and abundant (negative) class 
        min_idx = np.where(y == minority_class)[0]
        maj_idx = np.where(y == majority_class)[0]
        
        n_min = len(min_idx)
        
        if n_min == 0:
            raise ValueError("no positive class available in dataset.")
            
        self.estimators_ = []
        
        # The Macro-Bagging loop
        rng = np.random.RandomState(self.random_state)
        for i in range(self.n_bootstrap):
            chosen_maj_idx = rng.choice(maj_idx, size=n_min, replace=replace)
            
            # combine into perfect balanced  50/50 subset
            subset_idx = np.concatenate([min_idx, chosen_maj_idx])
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
        if hasattr(self.estimators_[0], "predict_proba") and self.predict_proba_if_present: #caution: i only check one base_estimator now
                probas = np.array([clf.predict_proba(X)[:,1] for clf in self.estimators_])
        else:
                if self.predict_proba_if_present: warn( "predict_proba() is not available. Falling back to predict(); returned values are vote proportions, not calibrated probabilities.", 
                                                        UserWarning, stacklevel=2) 
                probas = np.array([clf.predict(X) for clf in self.estimators_])
        
        macro_proba = np.mean(probas, axis=0)
        
        return np.vstack([1 - macro_proba, macro_proba]).T

    def predict(self, X):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)
  
