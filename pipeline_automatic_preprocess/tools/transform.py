import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression

def num_pipeline(select_method, scaler_strategy="ROBUST"):
    if select_method.lower() == 'mode':
        select_method = 'most_frequent'
        
    if scaler_strategy.upper() in ["ROBUST", "ROBUSTSCALER"]:
        scaler = RobustScaler()
    elif scaler_strategy.upper() in ["STANDARD", "STANDARDSCALER"]:
        scaler = StandardScaler()
    else:
        raise ValueError("The input of scaler_strategy must be one of the possible values: ['ROBUST', 'STANDARD']")

    return Pipeline([
        ('imputer', SimpleImputer(strategy=select_method)),
        ('scaler', scaler)
    ])

def cat_pipeline(select_method):
    if select_method.upper() == "ONE-HOT":
        encoder = CustomOneHotEncoder()
    else:
        encoder = CustomOrdinalEncoder()
    
    return Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent").set_output(transform="pandas")),
        ('encoder', encoder)
    ])



class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        try:
            self._oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            self._oh = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.columns_ = []

    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['string', 'object']).copy()
        if X_cat.shape[1] == 0:
            self.columns_ = []
            self._oh.fit(pd.DataFrame(index=X_cat.index))
            return self
        
        self._oh.fit(X_cat)
        self.columns_ = self._oh.get_feature_names_out(X_cat.columns)
        return self

    def transform(self, X, y=None):
        X_cat = X.select_dtypes(include=['string', 'object']).copy()
        if X_cat.shape[1] == 0:
            return pd.DataFrame(index=X.index)
        X_cat_oh = self._oh.transform(X_cat)
        return pd.DataFrame(X_cat_oh, columns=self.columns_, index=X.index)


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.columns_ = []

    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['string', 'object']).copy()
        if X_cat.shape[1] == 0:
            self.columns_ = []
            self._encoder.fit(pd.DataFrame(index=X_cat.index))
            return self
        
        self._encoder.fit(X_cat)
        self.columns_ = list(X_cat.columns)
        return self

    def transform(self, X, y=None):
        X_cat = X.select_dtypes(include=['string', 'object']).copy()
        if X_cat.shape[1] == 0:
            return pd.DataFrame(index=X.index)

        X_cat_encoded = self._encoder.transform(X_cat)
        return pd.DataFrame(X_cat_encoded, columns=self.columns_, index=X.index)



# We define a transformer for the data from the scikit-learn library (from sklearn.compose import ColumnTransformer)


class DataFramePreparer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_strategy, category_strategy="ONE-HOT", scaler_strategy="ROBUST"):
        self._full_pipeline = None
        self.columns_ = None
        self.input_features_ = None
        self.numeric_strategy = numeric_strategy
        self.category_strategy = category_strategy
        self.scaler_strategy = scaler_strategy

        if self.numeric_strategy.lower() not in ['mean', 'mode', 'median']:
            raise ValueError("The input of numeric_strategy must be one of the possible values: ['mean', 'mode', 'median']")
        
        if self.category_strategy.upper() not in ['ONE-HOT', 'LABEL']:
            raise ValueError("The input of category_strategy must be one of the possible values: ['ONE-HOT', 'LABEL']")

        if self.scaler_strategy.upper() not in ['ROBUST', 'ROBUSTSCALER', 'STANDARD', 'STANDARDSCALER']:
            raise ValueError("The input of scaler_strategy must be one of the possible values: ['ROBUST', 'STANDARD']")

    def fit(self, X, y=None):
        X0 = X.copy()
        self.input_features_ = list(X0.columns)

        numericcolumns_ = list(X0.select_dtypes(exclude=['object', 'string']).columns)
        categorycolumns_ = list(X0.select_dtypes(include=['object', 'string']).columns)


        # This Column transformers must be take all numeric collumns and apply the strategy defined on the previous code (inputation and scaling)
        # The second step is apply on categoric column the custom strategy apply the One-Hot Enconding
        # This returns us a final dataset with completely numercic data
        
        self._full_pipeline = ColumnTransformer([
            ("num", num_pipeline(self.numeric_strategy, self.scaler_strategy), numericcolumns_),
            ("cat", cat_pipeline(self.category_strategy), categorycolumns_),
        ])
        self._full_pipeline.fit(X0)

        out_cols = []
        out_cols.extend(numericcolumns_)
        cat_pipeline_obj = self._full_pipeline.named_transformers_["cat"]
        cat_encoder = cat_pipeline_obj.named_steps["encoder"]
        if hasattr(cat_encoder, "columns_") and len(cat_encoder.columns_) > 0:
            out_cols.extend(list(cat_encoder.columns_))
        self.columns_ = out_cols
        return self

    def transform(self, X, y=None):
        X0 = X.copy()
        X_prep = self._full_pipeline.transform(X0)
        return pd.DataFrame(X_prep, columns=self.columns_, index=X.index)


def build_full_pipeline(numeric_strategy="mean", category_strategy="ONE-HOT", scaler_strategy="ROBUST", max_iter=1000, random_state=15, **model_params):
    return Pipeline([
        ('preparer', DataFramePreparer(
            numeric_strategy=numeric_strategy,
            category_strategy=category_strategy,
            scaler_strategy=scaler_strategy
        )),
        ('model', LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            **model_params
        ))
    ])


if __name__ == "__main__":
    import os
    import sys

    os.chdir(os.path.join(os.getcwd(), '..'))

    df = pd.read_csv('data/Titanic-Dataset.csv')
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    X = df.drop("Survived", axis=1)
    y = df["Survived"].copy()

    full_pipeline = build_full_pipeline(
        numeric_strategy="mean",
        category_strategy="ONE-HOT",
        scaler_strategy="ROBUST"
    )

    full_pipeline.fit(X, y)

    # Predicción completa en una sola llamada
    predictions = full_pipeline.predict(X)
    print("Predicciones primeras 5 muestras:", predictions[:5])
    print("Accuracy sobre train:", full_pipeline.score(X, y))
