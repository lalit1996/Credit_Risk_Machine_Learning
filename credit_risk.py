import numpy as np
import pandas as pd
import pyodbc as sql
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import  StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import  Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


OE = OneHotEncoder(sparse_output=False)
Log = LogisticRegression()
STD = StandardScaler()
Imp = SimpleImputer(strategy='mean')


def connect_to_SQL():
    conn = sql.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=DESKTOP-F3Q1PP2\PRACTICESERVER;'      # e.g. "DESKTOP-12345\\SQLEXPRESS"
        'DATABASE=Credit_Risk;'
        'Trusted_Connection=yes;'
    )
    return conn


def load_data_to_dataframe(query):
    conn = connect_to_SQL()

    cursor = conn.cursor()
    data = cursor.execute(query)
    column = [x[0] for x in cursor.description]
    dataframe = pd.DataFrame.from_records(data.fetchall(),columns=column)
    return dataframe

def data_preprocessing():
    dataframe = load_data_to_dataframe('select * from [Credit_Risk].[dbo].[credit_risk_dataset]')
    categorical_features = ['person_home_ownership','loan_intent','loan_grade']
    numerical_feature = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_default_on_file','cb_person_cred_hist_length']
    Cat_Pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('encoded',OneHotEncoder(drop='first',sparse_output=False))]
    )
    Num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='mean')),
        ('scale',StandardScaler())
    ])

    combinedata = ColumnTransformer(
        transformers=[
            ('numerical',Num_pipeline,numerical_feature),
            ('Categorical',Cat_Pipeline,categorical_features)
        ]
    )

    data = combinedata.fit_transform(dataframe)
    onehotencoded_column = combinedata.named_transformers_['Categorical'].named_steps['encoded']
    onehotencoded_column = onehotencoded_column.get_feature_names_out(categorical_features)
    onehotencoded_column1 = list(onehotencoded_column)
    column = numerical_feature + onehotencoded_column1
    final_data = pd.DataFrame(data,columns=column,index=dataframe.index)
    return final_data,dataframe



#
def defualter():
    final_data, dataframe = data_preprocessing()
    X = final_data
    Y= dataframe['loan_status']

    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

    final_pip = Pipeline([
        ('log',LogisticRegression())
    ])

    final_pip.fit(x_train,y_train)
    prediction = final_pip.predict(x_test)
    prediction_DF = pd.DataFrame(np.array(prediction),columns=['Predicted_Y'])
    overall = pd.concat([y_test,prediction_DF],axis=1)
    acc = accuracy_score(y_test,prediction_DF)
    con = confusion_matrix(y_test,prediction_DF)
    print(con)
    print(acc)




defualter()




#
# def Credit_Defaulter(query):
#
#
#     column = []
#     for i in data.description:
#         column.append(i[0])
#     data_pd = pd.DataFrame.from_records(data.fetchall(),columns=column)
#
#     X = data_pd.drop(['loan_status'],axis=1)
#     Y = data_pd[['loan_status']]
#
#
#     Encoded_matrix = OE.fit_transform(X[['person_home_ownership','loan_intent','loan_grade']])
#     Encoded_matrix = pd.DataFrame(Encoded_matrix,columns=OE.get_feature_names_out(['person_home_ownership','loan_intent','loan_grade']))
#     X_Scaling = X.drop(['person_home_ownership','loan_intent','loan_grade'],axis=1)
#     X_Scaling_imputed = Imp.fit_transform(X_Scaling)
#     X_Scaled = STD.fit_transform(X_Scaling_imputed)
#     X_Scaled = pd.DataFrame(X_Scaled,columns=X_Scaling.columns)
#     X_Scaled = pd.concat([X_Scaled,Encoded_matrix],axis=1)
#
#     x_train, x_test, y_train, y_test = train_test_split(X_Scaled,Y,test_size=0.2,shuffle=True)
#
#
#     model = Log.fit(x_train,y_train)
#     predicted_y = model.predict(x_test)
#     predicted_y = pd.DataFrame(predicted_y,columns=['Predicted_y'])
#     tested_y = pd.DataFrame(y_test,columns=['tested_y'])
#     overall = pd.concat([x_test,predicted_y,tested_y],axis=1)
#     # print(overall.loc[overall['Predicted_y']!=overall['loan_status']])
#     # print(overall)
#     # print(y_test)
#     print(predicted_y)
#     # overall.to_excel(r"C://Users//dell//OneDrive//Desktop//Machine Learning//Credit_Risk//data.xlsx")
#
#
#


# load_data_to_dataframe('select * from [Credit_Risk].[dbo].[credit_risk_dataset]')
#
# defualter()