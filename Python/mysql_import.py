import pandas as pd
from sqlalchemy import create_engine

# Load cleaned dataset
df = pd.read_csv("Superstore_Cleaned.csv", encoding="latin1")

# Create MySQL connection
engine = create_engine(
    "mysql+pymysql://root:geetansh15@localhost/retail_analytics"
)

# Import data into MySQL
df.to_sql(
    name="superstore_data",
    con=engine,
    if_exists="replace",   # Creates/replaces the table
    index=False
)

print("Data imported successfully!")