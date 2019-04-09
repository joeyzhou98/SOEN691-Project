from pyspark.sql import SparkSession
from pyspark.sql import Row


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def clean_DAC_and_TFA(datafile):

    spark = init_spark()

    rdd = spark.read.csv(datafile, header=True) \
        .rdd \
        .zipWithIndex() \
        .map(lambda x: Row(index=x[1],
                           id=x[0][0],
                           date_account_created=x[0][1],
                           timestamp_first_active=x[0][2],
                           date_first_booking=x[0][3],
                           gender=x[0][4],
                           age=x[0][5],
                           signup_method=x[0][6],
                           signup_flow=x[0][7],
                           language=x[0][8],
                           affiliate_channel=x[0][9],
                           affiliate_provider=x[0][10],
                           first_affiliate_tracked=x[0][11],
                           signup_app=x[0][12],
                           first_device_type=x[0][13],
                           first_browser=x[0][14],
                           country_destination=x[0][15]))

    df = spark.createDataFrame(rdd).persist()

    dac_rdd = df.rdd \
        .map(lambda x: x["date_account_created"]) \
        .map(lambda x: x.replace("-", "")) \
        .map(lambda x: int(x[4:9])) \
        .zipWithIndex() \
        .map(lambda x: Row(index=x[1], dac=x[0]))

    dac_df = spark.createDataFrame(dac_rdd)

    tfa_rdd = df.rdd \
        .map(lambda x: x["timestamp_first_active"]) \
        .map(lambda x: int(x[4:8])) \
        .zipWithIndex() \
        .map(lambda x: Row(index=x[1], tfa=x[0]))

    tfa_df = spark.createDataFrame(tfa_rdd)

    df1 = df.join(dac_df, on="index") \
        .join(tfa_df, on="index") \
        .orderBy("index") \
        .drop("index") \
        .drop("date_account_created") \
        .drop("timestamp_first_active") \
        .withColumnRenamed("dac", "date_account_created") \
        .withColumnRenamed("tfa", "timestamp_first_active")

    df1.toPandas().to_csv("../data/train_users_2.csv", header=True)


# clean_DAC_and_TFA("../data/train_users_2.csv")
