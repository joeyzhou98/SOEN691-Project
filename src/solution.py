import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from pyspark.ml import Pipeline
from pyspark.sql.functions import when
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoderEstimator, Normalizer, MinMaxScaler
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import sklearn as sk
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
# from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



# credit from: 1. https://zhuanlan.zhihu.com/p/43629505
# 2. https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
# 3. change time format
# https://stackoverflow.com/questions/12604909/pandas-how-to-change-all-the-values-of-a-column
# https://blog.csdn.net/chenpe32cp/article/details/82180537

all_features=[
    "date_account_created"
    ,"timestamp_first_active"
    ,"gender"
    ,"age"
    ,"signup_method"
    ,"signup_flow"
    ,"language"
    ,"affiliate_channel"
    ,"affiliate_provider"
    ,"first_affiliate_tracked"
    ,"signup_app"
    ,"first_device_type"
    ,"first_browser"
    ,"country_destination"
    ]

features=["date_account_created"
    ,"timestamp_first_active"
    ,"gender","age"
    ,"signup_method"
    ,"signup_flow"
    ,"language"
    ,"affiliate_channel"
    ,"affiliate_provider"
    ,"first_affiliate_tracked"
    ,"signup_app"
    ,"first_device_type"
    ,"first_browser"]

class_label=["country_destination"]

categorical_features=[
    "gender"
    ,"signup_method"
    ,"language"
    ,"affiliate_channel"
    ,"affiliate_provider"
    # ,"first_affiliate_tracked"
    ,"signup_app"
    ,"first_device_type"
    ,"first_browser"
    ,"country_destination"]

numberical_feature=["date_account_created"
    ,"timestamp_first_active","age","signup_flow"]


#Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

# data_predicator("../data/train_users_2.csv")

def remove_all_missing(datafile):

    df=pd.read_csv(datafile)
    df=df.loc[:,all_features]# contains all the feature and class

    #remove aga outlier
    median = df.loc[df['age'] < 80, 'age'].median()
    df["age"] = np.where(df["age"] > 80, median, df['age'])
    df["age"] = np.where(df["age"] < 16, median, df['age'])

    # timestamp: keep the key info from timestamp for both "date_account_created" and "timestamp_first_active"
    def clean(x):
        x = x.replace("-", "")
        x = x[4:9]
        return int(x)

    df.update(df["date_account_created"].apply(clean))

    # print(df["date_account_created"])
    def clean2(x):
        x = x[4:8]
        return int(x)

    df.update(df["timestamp_first_active"].astype(str).apply(clean2))
    df.update(df["gender"].apply(lambda x: x.replace("-unknown-", "unknown")))
    df.update(df["first_browser"].apply(lambda x: x.replace("-unknown-", "unknown")))

    df = df.dropna()
    # df[all_features] = df[all_features].apply(LabelEncoder().fit_transform)
    # df = df[df.country_destination != "NDF"]
    # print(len(df))#123429
    # print(df)
    return df
# remove_all_missing("../data/train_users_2.csv")


def undersampling(datafile):
    # undersampling according to the size of other class
    data = remove_all_missing(datafile)
    other = data[data.country_destination == "other"]
    rest = data[data.country_destination != "US"]
    us = data[data.country_destination == "US"]
    us_under = us.sample(15000)
    under_df = pd.concat([us_under, rest])
    # print(under_df)
    # print(len(us_under))

    return under_df

# undersampling("../data/train_users_2.csv")

# credit: https://blog.csdn.net/m0_37324740/article/details/77169771
# http://blog.madhukaraphatak.com/class-imbalance-part-2/
# undersampling the majority class
# def over_sampling(datafile):
#     df = undersampling(datafile)
#     # combine oversampling and undersampling togeter with SMOTEENN
#     smote_enn = SMOTEENN(random_state=0)
#     X_resampled, y_resampled = smote_enn.fit_resample(df[features], df.country_destination)
#     print(sorted(Counter(y_resampled).items()))
#     back = pd.DataFrame(np.hstack((X_resampled, y_resampled[:, None]))) #[516489 rows x 14 columns]
#     # print(back)
#     return back
# # over_sampling("../data/train_users_2.csv")

def one_hot(datafile):
    spark=init_spark()
    df=spark.read.format("csv").option("header","true").load(datafile)
    df1=df.select(
    #     "date_account_created"
    # ,"timestamp_first_active",
    "gender"
    ,"age"
    ,"signup_method"
    ,"signup_flow"
    ,"language"
    ,"affiliate_channel"
    ,"affiliate_provider"
    ,"first_affiliate_tracked"
    ,"signup_app"
    ,"first_device_type"
    ,"first_browser"
    ,"country_destination")
    # print(df1)
    age_average = (df1.agg({"age": "sum"}).collect()[0][0]) / (df1.select("age").count())
    df2=df1.fillna({'age':age_average})
    # df3=df2.withColumn("age", when(df["age"]<=17, age_average).otherwise(df["age"]))
    # indexers = [StringIndexer(inputCol="gender", outputCol="gender_numeric").fit(df2)]
    df3=df2.dropna()

    indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in categorical_features]

    encoder = OneHotEncoderEstimator(
        inputCols=[indexer.getOutputCol() for indexer in indexers],
        outputCols=["{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers]
    )

    assembler = VectorAssembler(
        inputCols=encoder.getOutputCols(),
        outputCol="cat_features"
    )

    # combine all the numberical_feature togeher
    assembler2=VectorAssembler(
        inputCols=numberical_feature,
        outputCol="num_features"
    )

    pipeline = Pipeline(stages=indexers+[encoder,assembler])
    df_r = pipeline.fit(df2).transform(df2)
    # df_r.show()


    # combine all the numberical_feature togeher

    return df_r

# one_hot("../data/train_users_2.csv")

def normalization(data_file):
    data=filling_missing(data_file)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data)
    df = pd.DataFrame(x_scaled)
    # print(df)
    return df


def Constrained_kMeans(datafile):
    spark = init_spark()
    df = one_hot(datafile)
    scaler = MinMaxScaler(inputCol="cat_features", outputCol="scaledFeatures")
    # Compute summary statistics and generate MinMaxScalerModel
    scalerModel = scaler.fit(df)
    scaledData = scalerModel.transform(df)
    # print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))
    # scaledData.select("cat_features", "scaledFeatures").show()
    rdd=scaledData.rdd

    def initial_centorids():

        return ""

    def closed_centorids(vec):

        return ""


    return ""

Constrained_kMeans("../data/train_users_2.csv")

#213451
def filling_missing(datafile):
    df=pd.read_csv(datafile)
    df=df.loc[:,all_features]
    # print(len(df))
    # data = df.dropna()
    # print(len(data))#123429
    df.update(df["first_affiliate_tracked"].fillna("no_tracked"))
    df.update(df["age"].fillna(df.age.mean()))
    # age: https://stackoverflow.com/questions/45386955/python-replacing-outliers-values-with-median-values
    median = df.loc[df['age'] < 80, 'age'].median()
    df["age"] = np.where(df["age"] > 80 , median, df['age'])
    df["age"] = np.where(df["age"] < 16, median, df['age'])

    # timestamp: keep the key info from timestamp for both "date_account_created" and "timestamp_first_active"
    def clean(x):
        x = x.replace("-", "")
        x=x[4:9]
        return int(x)
    df.update(df["date_account_created"].apply(clean))
    # print(df["date_account_created"])
    def clean2(x):
        x=x[4:8]
        return int(x)
    df.update(df["timestamp_first_active"].astype(str).apply(clean2))
    # print(df["timestamp_first_active"])
    # print(len(df))213451
    df.update(df["gender"].apply(lambda x:x.replace("-unknown-", "unknown")))
    df.update(df["first_browser"].apply(lambda x: x.replace("-unknown-", "unknown")))
    # df.update(df["country_destination"].apply(lambda x: x.replace("NDF", "")))
    df = df.dropna()
    # print(len(df))213451
    df = df[df.country_destination != "NDF"]
    # print(df['country_destination'].value_counts())NDF      124543
    #     # print(len(df))88908
    #     # print(df.country_destination)
    le=LabelEncoder()
    labels=df.country_destination
    le.fit(labels)
    la=list(le.classes_)
    # print(la)['AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NL', 'PT', 'US', 'other']
    df[all_features] = df[all_features].apply(LabelEncoder().fit_transform)

    # print(df)
    return df


# credit: https://blog.csdn.net/m0_37324740/article/details/77169771
# http://blog.madhukaraphatak.com/class-imbalance-part-2/
# undersampling the majority class
def imbalance_undersampling(datafile):
    df = filling_missing(datafile)
    # combine oversampling and undersampling togeter with SMOTEENN
    smote_enn = SMOTEENN(random_state=0)
    X_resampled, y_resampled = smote_enn.fit_resample(df[features], df.country_destination)
    print(sorted(Counter(y_resampled).items()))
    back = pd.DataFrame(np.hstack((X_resampled, y_resampled[:, None]))) #[516489 rows x 14 columns]
    # print(back)
    return back
# imbalance_undersampling("../data/train_users_2.csv")













# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
def classifier_RF(data_file):
    # data=filling_missing(data_file)0.662842488864894%
    data=imbalance_undersampling(data_file)
    data.columns=all_features
    print(data)
    # data=remove_missing(data_file)
    # Split the data into training and testing sets
    train_features,test_features,train_labels,test_labels = train_test_split(data[features],data[class_label].values.ravel(),
                                                                                test_size=0.25,
                                                                                random_state=42)
    model = RandomForestClassifier(n_estimators=1000,random_state=42) #1.0.9999706620117058%
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    # model=RandomForestClassifier(n_estimators=100, min_samples_leaf=20, max_features='sqrt', random_state=10, max_depth=13)
    accuracy = metrics.accuracy_score(predictions, test_labels)#Accuracy:0.705685618729097%
    print(predictions)
    print('Accuracy:{0}%'.format(accuracy))
    all_score= metrics.classification_report(predictions, test_labels)
    print(all_score)

    return ""

# classifier_RF("../data/train_users_2.csv")

from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric

def costomize_distance_KM(vec1,vec2):
    #
    # user_function = lambda point1, point2: point1[0] + point2[0] + 2
    # metric = distance_metric(type_metric.USER_DEFINED, func=user_function)
    #
    # # create K-Means algorithm with specific distance metric
    # start_centers = [[4.7, 5.9], [5.7, 6.5]];
    # kmeans_instance = kmeans(sample, start_centers, metric=metric)
    #
    # # run cluster analysis and obtain results
    # kmeans_instance.process()
    # clusters = kmeans_instance.get_clusters()

    return " "

# normalization("../data/train_users_2.csv")


def normalization(data_file):
    data=filling_missing(data_file)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data)
    df = pd.DataFrame(x_scaled)
    # print(df)
    return df






# credit: http://midday.me/article/f8d29baa83ae41ec8c9826401eb7685e
def classifier_KM(datafile):
    df = pd.read_csv(datafile)
    df = df.loc[:, all_features]
    df.update(df["first_affiliate_tracked"].fillna("no_tracked"))

    df.update(df["age"].fillna(df.age.mean()))
    # age: https://stackoverflow.com/questions/45386955/python-replacing-outliers-values-with-median-values
    median = df.loc[df['age'] < 80, 'age'].median()
    df["age"] = np.where(df["age"] > 80, median, df['age'])
    df["age"] = np.where(df["age"] < 16, median, df['age'])

    # timestamp: keep the key info from timestamp for both "date_account_created" and "timestamp_first_active"
    def clean(x):
        x = x.replace("-", "")
        x = x[4:9]
        return int(x)
    df.update(df["date_account_created"].apply(clean))

    def clean2(x):
        x = x[4:8]
        return int(x)
    df.update(df["timestamp_first_active"].astype(str).apply(clean2))
    df.update(df["gender"].apply(lambda x: x.replace("-unknown-", "unknown")))
    df.update(df["first_browser"].apply(lambda x: x.replace("-unknown-", "unknown")))
    df = df.dropna()
    # y=df[df.country_destination != "NDF"]
    df = df[df.country_destination == "NDF"]

    df[all_features] = df[all_features].apply(LabelEncoder().fit_transform)
    df=df[features]

    # print(df)
    #
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(df)
    # df = pd.DataFrame(x_scaled)
    # df.columns = features
    # print(df)
    #
    # model = KMeans(n_clusters=11)
    # result=model.fit_predict(df)
    # centers=model.cluster_centers_
    #
    #
    # cluster_map = pd.DataFrame()
    # cluster_map['data_index'] = df.index.values
    # cluster_map['cluster'] = model.labels_
    # print(cluster_map )
    # for cen in centroid_list:

    return df

# classifier_KM("../data/train_users_2.csv")



def centroids(datafile):

    df=filling_missing(datafile)
    data=df
    le = preprocessing.LabelEncoder()

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)
    df = pd.DataFrame(x_scaled)
    df.columns = all_features

    X=df[features]
    y=data["country_destination"]
    clf = NearestCentroid()
    clf.fit(X, y)
    centroids=clf.centroids_
    return centroids


# https://www.cnblogs.com/why957/p/9318410.html
def classsifier_KNN(datafile):

    knn = KNeighborsClassifier()
    centroids_list = pd.DataFrame(centroids(datafile))

    test=classifier_KM(datafile)
    print(test)
    print(centroids_list)
    train = centroids_list
    target=centroids_list.index.values

    knn.fit(train, target)
    # ['AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NL', 'PT', 'US', 'other'][2 1 0 0 0 2 0 1 1 1 1]
    # 0
    # 124465
    # 1
    # 77
    # 4
    # 1

    print(type(test))
    result=pd.Series(knn.predict(test))

    print(result.value_counts())

    # centroid_map = pd.DataFrame()
    # centroid_map['centroid_index'] = centroids_list.index.values
    # centroid_map['centroid'] = [x for x in centroids_list]



# classsifier_KNN("../data/train_users_2.csv")


