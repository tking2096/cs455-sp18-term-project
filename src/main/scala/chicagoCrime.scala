import org.apache.spark.sql.SparkSession

object chicagoCrime {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().appName("chicagoCrime")
      .master("spark://concord.cs.colostate.edu:46010")
      .getOrCreate()

    // Read in CSV file
    val crime = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(args(0))
      .persist()

    val compactCrimes = crime.select("Date", "Latitude", "Longitude")

    compactCrimes.coalesce(1)
      .write.format("csv")
      .option("header", "true")
      .save("hdfs://concord:30101/cs455Project/compacted/01_04_CrimesDataSet")



  }

}
