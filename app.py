import streamlit as st
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel

spark = SparkSession.builder \
    .appName("Final App") \
    .getOrCreate()

sc = spark.sparkContext
# Streamlit UI for user inputs
st.title("Mortage Application System")

# Input fields
lei = st.text_input("Legal Entity Identifier (LEI)", "01J4SO3XTWZF4PP38209")
loan_type = st.selectbox("Loan Type", [1, 2, 3, 4], format_func=lambda x: {
    1: "Conventional",
    2: "FHA Insured",
    3: "VA Guaranteed",
    4: "RHS or FSA Guaranteed"
}[x])
loan_purpose = st.selectbox("Loan Purpose", [1, 2, 31, 32, 4], format_func=lambda x: {
    1: "Home Purchase",
    2: "Home Improvement",
    31: "Refinancing",
    32: "Cash-out Refinancing",
    4: "Other Purpose"
}[x])
loan_amount = st.number_input("Loan Amount", 0.0, 10000000.0, step=1000.0)
interest_rate = st.number_input("Interest Rate", 0.0, 20.0, step=0.01)
loan_term = st.number_input("Loan Term (Months)", 0, 480, step=12)
# action_taken = st.selectbox("Action Taken", [1, 2, 3, 4, 5, 6, 7, 8], format_func=lambda x: {
#     1: "Loan Originated",
#     0: "Rejected"
# }[x])
income = st.number_input("Income (in Thousands)", 0.0, 1000000.0, step=1.0)
applicant_age = st.selectbox("Applicant Age", ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74", "NA"])
applicant_sex = st.selectbox("Applicant Sex", [1, 2, 4], format_func=lambda x: {
    1: "Male",
    2: "Female",
    4: "NA"
}[x])
applicant_credit_score_type = st.number_input("Applicant Credit Score Type", 1, 10, step=1)
co_applicant_age = st.selectbox("Co-Applicant Age", ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74", "NA"])
co_applicant_credit_score_type = st.number_input("Co-Applicant Credit Score Type", 1, 10, step=1)
derived_msa_md = st.number_input("Derived MSA/MD", 0, 100000, step=1)
state_code = st.text_input("State Code (2 Characters)", "CA")
county_code = st.text_input("County Code (5 Characters)", "06037")
property_value = st.number_input("Property Value (in Thousands)", 0.0, 10000000.0, step=1000.0)
total_units = st.number_input("Total Units", 1, 1000, step=1)
occupancy_type = st.selectbox("Occupancy Type", [1, 2, 3], format_func=lambda x: {
    1: "Principal Residence",
    2: "Second Residence",
    3: "Investment Property"
}[x])



print(type(occupancy_type))

# Define the schema
schema = StructType([
    StructField("lei", StringType(), True),
    StructField("loan_type", IntegerType(), True),
    StructField("loan_purpose", IntegerType(), True),
    StructField("loan_amount", FloatType(), True),
    StructField("interest_rate", FloatType(), True),
    StructField("loan_term", IntegerType(), True),
    StructField("action_taken", IntegerType(), True),
    StructField("income", FloatType(), True),
    StructField("applicant_age", StringType(), True),
    StructField("applicant_sex", IntegerType(), True),
    StructField("applicant_credit_score_type", IntegerType(), True),
    StructField("co_applicant_age", StringType(), True),
    StructField("co_applicant_credit_score_type", IntegerType(), True),
    StructField("derived_msa_md", IntegerType(), True),
    StructField("state_code", StringType(), True),
    StructField("county_code", StringType(), True),
    StructField("property_value", FloatType(), True),
    StructField("total_units", IntegerType(), True),
    StructField("occupancy_type", IntegerType(), True)
])



# Submit button
if st.button("Submit Data"):
    ###### PYSPARK
    
    # Create a datapoint
    datapoint = Row(
        lei=lei,
        loan_type=loan_type,
        loan_purpose=loan_purpose,
        loan_amount=float(loan_amount),
        interest_rate=float(interest_rate),
        loan_term=int(loan_term),
        action_taken=0,
        income=float(income),
        applicant_age=applicant_age,
        applicant_sex=applicant_sex,
        applicant_credit_score_type=int(applicant_credit_score_type),
        co_applicant_age=co_applicant_age,
        co_applicant_credit_score_type=int(co_applicant_credit_score_type),
        derived_msa_md=int(derived_msa_md),
        state_code=state_code,
        county_code=county_code,
        property_value=float(property_value),
        total_units=int(total_units),
        occupancy_type=occupancy_type
    )

    # Create a PySpark DataFrame
    datapoint_df = spark.createDataFrame([datapoint], schema=schema)

    # datapoint_df

    # Prediction
    lr_model = LogisticRegressionModel.load("models/Logistic Regression")
    transform_model = PipelineModel.load('models/transformPipeModel')

    transform_point = transform_model.transform(datapoint_df)
    predict_point = lr_model.transform(transform_point)

    result = predict_point.select('prediction')
    print(result.show())
    res = result.collect()[0][0]

    st.subheader(f"Prediction result: {int(res)}")
    # Display DataFrame
    # st.write("Your DataFrame:")
    # st.dataframe(datapoint_df.toPandas())

    # Optionally save the data as a Parquet file
    # datapoint_df.write.mode("overwrite").parquet("loan_data.parquet")
    # st.success("Data saved as 'loan_data.parquet'")
