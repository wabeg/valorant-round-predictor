from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

def show_top5_correlated(df):
    for column in df:
        print(f'Top 5 correlated columns with {column}:')
        print(df.corr()[column].sort_values(ascending=False).head())
        print()

def pipeline(X,y):
    
    #X is the dataframe of features
    #y is the target variable
    
    
    #split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)
    
    #print baseline
    print(f'baseline: {y_test.value_counts(normalize=True)[1]}')
    print()
    
    
    #scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #oversample data
    oversample = SMOTE(random_state=42)
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    X_test, y_test = oversample.fit_resample(X_test, y_test)
    
    #fit logistic regression model
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train,y_train)
    lr_score = lr.score(X_test,y_test)
    lr_confusion_matrix = confusion_matrix(y_test, lr.predict(X_test), normalize='true')
    
    #print logistic regression results
    print('0: Logistic Regression Score:', lr_score)
    # print('Logistic Regression Confusion Matrix:')
    # print(lr_confusion_matrix)
    print()
    
    #fit xgboost model
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train,y_train)
    xgb_score = xgb.score(X_test,y_test)
    xgb_confusion_matrix = confusion_matrix(y_test, xgb.predict(X_test), normalize='true')
    
    #print xgboost results
    print('1: XGBoost Score:', xgb_score)
    # print('XGBoost Confusion Matrix:')
    # print(xgb_confusion_matrix)
    print()
    
    #fit random forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train,y_train)
    rf_score = rf.score(X_test,y_test)
    rf_confusion_matrix = confusion_matrix(y_test, rf.predict(X_test), normalize='true')
    
    #print random forest results
    print('2: Random Forest Score:', rf_score)
    # print('Random Forest Confusion Matrix:')
    # print(rf_confusion_matrix)
    print()
    
    #fit knn model
    knn = KNeighborsClassifier()
    knn.fit(X_train,y_train)
    knn_score = knn.score(X_test,y_test)
    knn_confusion_matrix = confusion_matrix(y_test, knn.predict(X_test), normalize='true')
    
    #print knn results
    print('3: KNN Score:', knn_score)
    # print('KNN Confusion Matrix:')
    # print(knn_confusion_matrix)
    print()
    
    
    #fit decision tree model
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train,y_train)
    dt_score = dt.score(X_test,y_test)
    dt_confusion_matrix = confusion_matrix(y_test, dt.predict(X_test), normalize='true')
    
    #print decision tree results
    print('4: Decision Tree Score:', dt_score)
    # print('Decision Tree Confusion Matrix:')
    # print(dt_confusion_matrix)
    print()
    
    #fit gradient boosting model
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train,y_train)
    gb_score = gb.score(X_test,y_test)
    gb_confusion_matrix = confusion_matrix(y_test, gb.predict(X_test), normalize='true')
    
    #print gradient boosting results
    print('5: Gradient Boosting Score:', gb_score)
    # print('Gradient Boosting Confusion Matrix:')
    # print(gb_confusion_matrix)
    print()
    
    #fit adaboost model
    ada = AdaBoostClassifier(random_state=42)
    ada.fit(X_train,y_train)
    ada_score = ada.score(X_test,y_test)
    ada_confusion_matrix = confusion_matrix(y_test, ada.predict(X_test), normalize='true')
    
    #print adaboost results
    print('6: AdaBoost Score:', ada_score)
    # print('AdaBoost Confusion Matrix:')
    # print(ada_confusion_matrix)
    print()
    
    #fit gaussian naive bayes model
    gnb = GaussianNB()
    gnb.fit(X_train,y_train)
    gnb_score = gnb.score(X_test,y_test)
    gnb_confusion_matrix = confusion_matrix(y_test, gnb.predict(X_test), normalize='true')
    
    #print gaussian naive bayes results
    print('7: Gaussian Naive Bayes Score:', gnb_score)
    # print('Gaussian Naive Bayes Confusion Matrix:')
    # print(gnb_confusion_matrix)
    print()
    
    #fit quadratic discriminant analysis model
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train,y_train)
    qda_score = qda.score(X_test,y_test)
    qda_confusion_matrix = confusion_matrix(y_test, qda.predict(X_test), normalize='true')
    
    #print gaussian naive bayes results
    print('8: Quadratic Discriminant Analysis Score:', qda_score)
    # print('Quadratic Discriminant Analysis Confusion Matrix:')
    # print(qda_confusion_matrix)
    
    return lr, xgb, rf, knn, dt, gb, ada, gnb, qda, X_test, y_test, scaler