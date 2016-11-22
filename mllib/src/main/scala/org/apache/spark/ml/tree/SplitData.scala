package org.apache.spark.ml.tree


/**
 * Info for a [[org.apache.spark.ml.tree.Split]]
 *
 * @param featureIndex  Index of feature split on
 * @param leftCategoriesOrThreshold  For categorical feature, set of leftCategories.
 *                                   For continuous feature, threshold.
 * @param numCategories  For categorical feature, number of categories.
 *                       For continuous feature, -1.
 */
case class SplitData(
                      val featureIndex: Int,
                      val leftCategoriesOrThreshold: Array[Double],
                      val numCategories: Int) {

  def getSplit: Split = {
    if (numCategories != -1) {
      new CategoricalSplit(featureIndex, leftCategoriesOrThreshold, numCategories)
    } else {
      assert(leftCategoriesOrThreshold.length == 1, s"DecisionTree split data expected" +
        s" 1 threshold for ContinuousSplit, but found thresholds: " +
        leftCategoriesOrThreshold.mkString(", "))
      new ContinuousSplit(featureIndex, leftCategoriesOrThreshold(0))
    }
  }
}

object SplitData {
  def apply(split: Split): SplitData = split match {
    case s: CategoricalSplit =>
      SplitData(s.featureIndex, s.leftCategories, s.numCategories)
    case s: ContinuousSplit =>
      SplitData(s.featureIndex, Array(s.threshold), -1)
  }
}
