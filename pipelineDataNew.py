
def transformData(data,feature_selection):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
    ])
    
    # from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import OrdinalEncoder
    some_data=data[feature_selection]
    # y=some_data[["stroke"]]
    # y=OrdinalEncoder().fit_transform(y)
    
    # some_data=some_data.drop(columns=["stroke"])
    num_cols=some_data.select_dtypes(include=["int","float"]).columns.to_list()
    cat_cols=some_data.select_dtypes(exclude=["int","float"]).columns.to_list()
    from sklearn.compose import ColumnTransformer
    full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_cols ),
    ("cat",  OrdinalEncoder(), cat_cols)
    ])
    X = full_pipeline.fit_transform(some_data)
    return X
def  Prediction(X_new,model):
    X_new=transformData(X_new)
    y_predict = model.predict(X_new)
    return y_predict

