#
### Store results as Hive table using Spark 
#
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, SQLContext, HiveContext
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.sql.functions import array, col, explode, lit, struct, udf
from pyspark.sql import DataFrame
from datetime import datetime
import pandas as pd

### Configure SparkConf
conf = SparkConf()

#resourcePool = 'haddeveloper'
resourcePool = 'haddatascientist'

conf.set('spark.dynamicAllocation.enabled', 'true')
conf.set('spark.dynamicAllocation.initialExecutors', 2)
conf.set('spark.dynamicAllocation.minExecutors', 2)
conf.set("spark.executor.memory","4g")
conf.set("spark.driver.memory","4g")
conf.set("spark.default.parallelism","200")             #standard 100
conf.set("spark.sql.shuffle.partitions",200)              #standard 8, this is an integer not a string
conf.set('spark.yarn.queue', resourcePool) # Resource Pool
conf.set('spark.sql.crossJoin.enabled', 'true')
conf.set("spark.sql.autoBroadcastJoinThreshold", -1)    # needed to allow 2_001 sql file to run

conf.set('spark.yarn.executor.memoryOverhead', '8gb')
#conf.set("spark.shuffle.blockTransferService", "nio")
sessionName = locals().get('sessionName', 'dmitry_dev')

### Create spark session, sparkContext, HiveContext, and SQLContext
spark = SparkSession\
    .builder\
    .appName(sessionName)\
    .config(conf=conf)\
    .enableHiveSupport()\
    .getOrCreate()

sc = spark.sparkContext
hc = HiveContext(sc)

today = datetime.now().strftime("%Y_%m_%d") 
data_bad_egr = pd.read_csv('EGR/EGR_model/data/data_bad_egr.csv')

#Add fleetnames
import nav_connect as nv
odbc = nv.odbc()
q = '''
select vin, fleetname from wia.ras_sales
'''
df_fleet = odbc.read_sql(q)
data_bad_egr = pd.merge(data_bad_egr,df_fleet, on='vin')
                                                 
         
def store_to_hive(df, tableName):
    spark_df = spark.createDataFrame(df)
    hc.sql('DROP TABLE IF EXISTS {} PURGE'.format(tableName))
    spark_df.createOrReplaceTempView("mytempTable") 
    hc.sql('create table {} as select * from mytempTable'.format(tableName));
    #hc.sql('invalidate metadata {}'.format(tableName))

tableName = 'analyticsplayground.a26_vins_by_probability_egr_{}'.format(today)    
store_to_hive(data_bad_egr[['vin','fleetname', 'prob', 'miles']], tableName)
print('{} is stored to Hive.'.format(tableName))

print('All done.')
