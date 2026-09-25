from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

def split_data(df: DataFrame, seed: int = 42):
    train_df, val_df, test_df = df.randomSplit([0.70, 0.15, 0.15], seed=seed)
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    spark = SparkSession.builder.appName("Training").getOrCreate()
    unsw = spark.read.parquet("data/processed/clean_unsw.parquet")
    
    train_df, val_df, test_df = split_data(unsw)
    
    for name, split in [("train", train_df),("validation", val_df),("test", test_df)]:
        print(name)
        split.groupBy("label").count().show()
        
    train_df.write.mode("errorifexists").parquet("/Users/thonph/Desktop/KLTN/data/splits/unsw_train")
    val_df.write.mode("errorifexists").parquet("/Users/thonph/Desktop/KLTN/data/splits/unsw_val")
    test_df.write.mode("errorifexists").parquet("/Users/thonph/Desktop/KLTN/data/splits/unsw_test")
    
    spark.stop()