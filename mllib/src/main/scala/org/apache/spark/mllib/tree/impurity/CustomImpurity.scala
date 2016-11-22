package org.apache.spark.mllib.tree.impurity


/** patched immpurity to use for supply chain problem where the problem is of classification.
  * The Classes have a linear co-rellation between them
  * works similar to regression but will store the probabilities of each class
  * Modified version of gini impurity which factors in Linear corelation.
  */
object CustomImpurity extends  Impurity{
  /**
   * :: DeveloperApi ::
   * information calculation for multiclass classification
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
  override def calculate(counts: Array[Double], totalCount: Double): Double = {

    var median = 0 ;
    if (totalCount == 0) {
      return 0
    }

    var impurityPercentile = 0.5
    if(params.contains("impurityPercentile")) {
      val percentile: Double = params("impurityPercentile").toDouble
      if(percentile > 0 && percentile < 1) {
        impurityPercentile = percentile
      }
    }

    val prediction  = getPrediction(counts, totalCount, impurityPercentile)

    var variance:Double = 0.0

    var frequency:Double = 0;
    for(index <- 0 to counts.length - 1 ) {
      frequency = counts(index)/totalCount
      variance += frequency*(prediction - index)*(prediction - index)
    }

    variance
  }

  def getPrediction(counts: Array[Double], totalCount: Double, impurityPercentile: Double): Double = {
    var prediction = -1;
    val cutOff = impurityPercentile * totalCount;
    var total: Double = 0;

    while (total <= cutOff) {
      prediction += 1
      total += counts(prediction);
    }
    prediction
  }

  /**
   * :: DeveloperApi ::
   * information calculation for regression
   * @param count number of instances
   * @param sum sum of labels
   * @param sumSquares summation of squares of the labels
   * @return information value, or 0 if count = 0
   */
  override def calculate(count: Double, sum: Double, sumSquares: Double): Double = {
    throw new UnsupportedOperationException("CustomImpurity.calculate")
  }
}


private[tree] class CustomImpurityAggregator(numClasses: Int)
  extends ImpurityAggregator(numClasses) with Serializable {

  /**
   * Update stats for one (node, feature, bin) with the given label.
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def update(allStats: Array[Double], offset: Int, label: Double, instanceWeight: Double): Unit = {
    if (label >= statsSize) {
      throw new IllegalArgumentException(s"Custom Impurity Aggregator given label $label" +
        s" but requires label < numClasses (= $statsSize).")
    }
    if (label < 0) {
      throw new IllegalArgumentException(s"Custom Impurity Aggregator given label $label" +
        s"but requires label is non-negative.")
    }
    allStats(offset + label.toInt) += instanceWeight
  }

  /**
   * Get an [[ImpurityCalculator]] for a (node, feature, bin).
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def getCalculator(allStats: Array[Double], offset: Int): CustomImpurityCalculator = {
    new CustomImpurityCalculator(allStats.view(offset, offset + statsSize).toArray)
  }

}

/**
 * Stores statistics for one (node, feature, bin) for calculating impurity.
 * Unlike [[CustomImpurityAggregator]], this class stores its own data and is for a specific
 * (node, feature, bin).
 * @param stats  Array of sufficient statistics for a (node, feature, bin).
 */
private[spark] class CustomImpurityCalculator(stats: Array[Double]) extends ImpurityCalculator(stats) {

  /**
   * Make a deep copy of this [[ImpurityCalculator]].
   */
  def copy: CustomImpurityCalculator = new CustomImpurityCalculator(stats.clone())

  /**
   * Calculate the impurity from the stored sufficient statistics.
   */
  def calculate(): Double = CustomImpurity.calculate(stats, stats.sum)

  /**
   * Number of data points accounted for in the sufficient statistics.
   */
  def count: Long = stats.sum.toLong

  /**
   * Prediction which should be made based on the sufficient statistics.
   */
  def predict: Double = if (count == 0) {
    0
  } else {
    indexOfLargestArrayElement(stats)
  }

  /**
   * Probability of the label given by [[predict]].
   */
  override def prob(label: Double): Double = {
    val lbl = label.toInt
    require(lbl < stats.length,
      s"CustomImpurityCalculator.prob given invalid label: $lbl (should be < ${stats.length}")
    require(lbl >= 0, "CustomImpurity does not support negative labels")
    val cnt = count
    if (cnt == 0) {
      0
    } else {
      stats(lbl) / cnt
    }
  }

  override def toString: String = s"CustomImpurityCalculator(stats = [${stats.mkString(", ")}])"

}