from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

# 1. Inizializza Spark
spark = SparkSession.builder.appName("AirQualityValidation").getOrCreate()

# 2. Carica dataset di validazione
df_val = spark.read.csv("dataset/dataset_validation.csv", header=True, inferSchema=True, sep=";")

# 3. Carica il modello salvato
model = PipelineModel.load("models/random_forest_v2")

# importanza delle features per random forest
# Estrai il classificatore dalla pipeline
rf_model = model.stages[-1]

# Ottieni le importanze di random forest
importances = rf_model.featureImportances.toArray()
feature_cols = [
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5",
    "pm10", "nh3", "temperature, wind_speed", "humidity"]

# Stampa abbinate alle feature
print("Stampa dell'importanza delle features")
for feat, imp in zip(feature_cols, importances):
    print(f"{feat}: {imp:.4f}")


"""
# Estrai il classificatore dalla pipeline
lr_model = model.stages[-1]

# I coefficienti sono diversi per ogni classe (multiclasse softmax)
coefficients = lr_model.coefficientMatrix

print("Coefficienti per ciascuna classe:")
print(coefficients)

# Se vuoi vederli col nome delle feature
for i, row in enumerate(coefficients.toArray()):
    print(f"Classe {i}:")
    for feat, coeff in zip(feature_cols, row):
        print(f"   {feat}: {coeff:.4f}")
"""

# 4. Applica il modello al dataset di validazione
predictions = model.transform(df_val)

# 5. Valutazione
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"✅ Accuracy sul validation set: {accuracy:.4f}")

results = predictions.withColumn("prediction", col("prediction") + 1).select("datetime", "next_aqi", "prediction")

# 6. Salva in CSV (una sola cartella con un file)
#results.coalesce(1).write.csv("results/mlp_results_validation.csv", header=True, mode="overwrite", sep = ';')

#print("✅ Risultati delle predizioni salvati in results/mlp_results_validation.csv")


# F1-score
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="f1"
)
f1_score = evaluator_f1.evaluate(predictions)
print(f"✅ F1-score: {f1_score:.4f}")