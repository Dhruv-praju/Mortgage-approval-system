from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, FloatType
from pyspark.sql.functions import col, when, count, isnan
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler, FeatureHasher
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, LinearSVC, NaiveBayes, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, FMClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import warnings
import sys
import os
warnings.filterwarnings("ignore")

spark = SparkSession.builder \
    .appName("Final") \
    .getOrCreate()

sc = spark.sparkContext

## 1. DATA INGESTION

# load data from .txt file
inputTextFile = sys.argv[1]
outputDir = sys.argv[2]

# df = spark.read.text(inputTextFile)

df = spark.read.option("header", "true") \
    .option("delimiter", "|") \
    .csv(inputTextFile)

# take subset of large dataset
# total_rows = df.count()
# fraction = 10000/total_rows
# df = df.sample(fraction=fraction, withReplacement=False)

print("...................Data Ingestion Successfull......................")

## 2. FEATURE SELECTION

loan_details_df = df.select("activity_year", "lei", "loan_type","loan_purpose", "loan_amount", "interest_rate","loan_term", "action_taken")
borrower_details_df = df.select("income","applicant_age","applicant_sex", "applicant_credit_score_type", "co_applicant_age", "co_applicant_credit_score_type")
property_details_df = df.select("derived_msa_md", "state_code", "county_code", "property_value", "total_units", "occupancy_type")

columns = loan_details_df.columns + borrower_details_df.columns + property_details_df.columns

df = df.select(columns)

df = df.drop('activity_year')

print("....................Feature selection Successfull...................")
## 3. DATA CHECKS/CLEANING

# change data type
coltype_map = {
    "lei": StringType(),
    "loan_type": IntegerType(),
    "loan_purpose": IntegerType(),
    "loan_amount": FloatType(),
    "interest_rate": FloatType(),
    "loan_term": IntegerType(),
    "action_taken": IntegerType(),
    "income": FloatType(),
    "applicant_age": StringType(),
    "applicant_sex": IntegerType(),
    "applicant_credit_score_type": IntegerType(),
    "co_applicant_age": StringType(),
    "co_applicant_credit_score_type": IntegerType(),
    "derived_msa_md": IntegerType(),
    "state_code": StringType(),
    "county_code": StringType(),
    "property_value": FloatType(),
    "total_units": IntegerType(),
    "occupancy_type": IntegerType(),
}

for col_name, data_type in coltype_map.items():
    df = df.withColumn(col_name, col(col_name).cast(data_type))

# check Missing count for each column
missing_info = df.select([
    count(when(col(c).isNull() | isnan(col(c)) |(col(c) == "NA"), c)).alias(c) for c in df.columns
])

# median imputation for 'income'
from pyspark.ml.feature import Imputer

income_imputer = Imputer(inputCols=['income'], outputCols=['income'], strategy="median")
df = income_imputer.fit(df).transform(df)

# mean imputation for `interest_rate`
interest_rate_imputer = Imputer(inputCols=["interest_rate"], outputCols=["interest_rate"], strategy="mean")
df = interest_rate_imputer.fit(df).transform(df)

# mode imputation for 'loan_Term'
from pyspark.sql.functions import col, count, when
mode_loan_term = (
    df.groupBy("loan_term")
    .count()
    .orderBy(col("count").desc())
    .first()[0]
)
df = df.withColumn( "loan_term", when(col("loan_term").isNull(), mode_loan_term).otherwise(col("loan_term")))

# remove rows that has no 'property_value'
df = df.filter(col('property_value')!=0)

# remove rows that has no 'state_code'
df = df.filter(col('state_code')!='NA')

# remove rows that has no 'county_code'
df = df.filter(col('county_code')!='NA')

# remove rows that has no 'total_units'
df = df.filter(col('total_units')!=0)

# clean target variable
df = df.withColumn('action_taken', when(col('action_taken')!=1, 0).otherwise(1) )

## DATA SAVING/LOADING AS PARQUET
outputPath = os.path.join(outputDir, 'cleaned_dataset')
df.write.parquet(outputPath, mode='overwrite')

print("...............Data saved as Parquet Sucessfully.................")

# load the dataset
df = spark.read.parquet(outputPath)

## 4. TRAIN TEST SPLIT
train_df, test_df = df.randomSplit(weights=[0.8, 0.2], seed=100)
print("............Data splited to train-test Sucessfully................")

## 5. FEATURE ENGINEERING

# define categorical and numerical features
categorical_features = [field.name for field in train_df.schema.fields if field.dataType is StringType()]
numerical_features = [field for field in train_df.columns if field not in categorical_features]

continuous_numerical_features = ["loan_amount", "interest_rate", "loan_term", "income", "property_value"]
discrete_numerical_features = [item for item in numerical_features if item not in continuous_numerical_features]
categorical_features.remove('county_code')
cat_indexed_features = [f"{cat}_indexed" for cat in categorical_features]

# perform Label encoding on categorical features
labelEncoder = StringIndexer(inputCols=categorical_features, 
                           outputCols=cat_indexed_features, handleInvalid="skip")

# perform feature hashing on 'county_code' as a lot of distinct values
hasher = FeatureHasher(inputCols=["county_code"], outputCol="county_code_hashed", numFeatures=1000)

# perform Standard scaling on continuous numerical features
numAssembler = VectorAssembler(inputCols=continuous_numerical_features, 
                            outputCol="con_num_features")

numScaler = StandardScaler(inputCol="con_num_features", outputCol="con_num_features_scaled")

# assemble all the features together
featureAssembler = VectorAssembler(inputCols=["con_num_features_scaled", "county_code_hashed"]+cat_indexed_features, outputCol='features')

# make the Pipeline
transformPipeline = Pipeline(stages = [labelEncoder, hasher, numAssembler, numScaler, featureAssembler])

# train it
transformPipeModel = transformPipeline.fit(train_df)

train_df = transformPipeModel.transform(train_df)

print("...............Feature Engineering Successfull.................")

## 6. MODEL PREDICTION
def evaluate_model(predictions, label_col='action_taken', prediction_col='prediction', raw_prediction_col='rawPrediction'):
    '''It returns classification evaluation metrics like accuracy, precision, f1, recall and roc'''
    
    # Initialize evaluators
    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName='accuracy')
    evaluator_precision = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName='weightedPrecision')
    evaluator_recall = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName='weightedRecall')
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName='f1')
    evaluator_roc = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=raw_prediction_col, metricName='areaUnderROC')

    # Calculate metrics
    accuracy = evaluator_accuracy.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    f1_score = evaluator_f1.evaluate(predictions)
    roc_auc = evaluator_roc.evaluate(predictions)

    # Return all metrics as a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'roc_auc': roc_auc
    }

    return metrics

models = {
    'Logistic Regression': LogisticRegression(featuresCol='features', labelCol='action_taken'),
    'Support Vector Machine': LinearSVC(featuresCol='features', labelCol='action_taken'),
    # 'Naive Bayes': NaiveBayes(featuresCol='features', labelCol='action_taken'),
    'Factorization Machine': FMClassifier(featuresCol='features', labelCol='action_taken'),
    'Decision Tree': DecisionTreeClassifier(featuresCol='features', labelCol='action_taken', maxBins=4000),
    'Random Forest': RandomForestClassifier(featuresCol='features', labelCol='action_taken', maxBins=4000),
    'Gradient Boosting Trees': GBTClassifier(featuresCol='features', labelCol='action_taken', maxBins=4000),
}

for algo in models:
    print(f"========== {algo} ============")

    # Train the model
    model = models[algo]
    trained_model = model.fit(train_df)

    # Evaluate on Test data
    test_df_transformed = transformPipeModel.transform(test_df)
    test_predictions = trained_model.transform(test_df_transformed)

    results = evaluate_model(test_predictions)
    print("accuracy: {:.4f}".format(results['accuracy']))
    print("precison: {:.4f}".format(results['precision']))
    print("recall: {:.4f}".format(results['recall']))
    print("f1-score: {:.4f}".format(results['f1_score']))
    print("ROC: {:.4f}".format(results['roc_auc']))

    print('\n')

print(".............. DONE Sucessfully!!! ..........................")