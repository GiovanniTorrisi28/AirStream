from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

from pyspark.ml.classification import MultilayerPerceptronClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# 1. Inizializza Spark
spark = SparkSession.builder.appName("AirQualityTraining").getOrCreate()

# 2. Carica dataset di training
df = spark.read.csv("dataset/dataset_training.csv", header=True, inferSchema=True, sep=";")

# 3. Definisci features e label
feature_cols = [
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5",
    "pm10", "nh3", "temperature", "wind_speed", "humidity"]

# Converti la colonna 'next_aqi' in formato numerico (necessario per Spark MLlib)
label_indexer = StringIndexer(inputCol="next_aqi", outputCol="label")

# Assembla le feature in un unico vettore, se si normalizza scrivere 'features_raw' in outputcol
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
# Normalizza le features (StandardScaler → media 0, varianza 1)
# scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)


# Classificazione con random forest

rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
pipeline = Pipeline(stages=[label_indexer, assembler, rf])


# Classificazione con logistic regression
"""
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,  
    regParam=0 # regolarizzazione L2
)
pipeline = Pipeline(stages=[label_indexer,assembler, scaler, lr])
"""

# Classificazione con rete neurale
"""
# 4. Definizione MLP
# Numero input = numero feature, numero output = 5 (classi AQI)
layers = [len(feature_cols),128, 64, 32, 16, 5]  # input → hidden1 → hidden2 → output

mlp = MultilayerPerceptronClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=300,
    blockSize= 32,
    seed=1234,
    layers=layers
)
# 5. Pipeline
pipeline = Pipeline(stages=[label_indexer, assembler, scaler, mlp])
"""

# 4. Allena il modello
model = pipeline.fit(df)

# 5. Salva il modello
model.save("models/random_forest_v2")

print("✅ Modello addestrato e salvato con successo!")
