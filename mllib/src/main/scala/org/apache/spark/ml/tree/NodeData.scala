package org.apache.spark.ml.tree


/**
 * Info for a [[Node]]
 * @param id  Index used for tree reconstruction.  Indices follow a pre-order traversal.
 * @param impurityStats  Stats array.  Impurity type is stored in metadata.
 * @param gain  Gain, or arbitrary value if leaf node.
 * @param leftChild  Left child index, or arbitrary value if leaf node.
 * @param rightChild  Right child index, or arbitrary value if leaf node.
 * @param split  Split info, or arbitrary value if leaf node.
 */
case class NodeData(
                     val id: Int,
                     val prediction: Double,
                     val impurity: Double,
                     val impurityStats: Array[Double],
                     val gain: Double,
                     val leftChild: Int,
                     val rightChild: Int,
                     val split: SplitData)

object NodeData {
  /**
   * Create [[NodeData]] instances for this node and all children.
   *
   * @param id  Current ID.  IDs are assigned via a pre-order traversal.
   * @return (sequence of nodes in pre-order traversal order, largest ID in subtree)
   *         The nodes are returned in pre-order traversal (root first) so that it is easy to
   *         get the ID of the subtree's root node.
   */
  def build(node: Node, id: Int): (Seq[NodeData], Int) = node match {
    case n: InternalNode =>
      val (leftNodeData, leftIdx) = build(n.leftChild, id + 1)
      val (rightNodeData, rightIdx) = build(n.rightChild, leftIdx + 1)
      val thisNodeData = NodeData(id, n.prediction, n.impurity, n.impurityStats.stats,
        n.gain, leftNodeData.head.id, rightNodeData.head.id, SplitData(n.split))
      (thisNodeData +: (leftNodeData ++ rightNodeData), rightIdx)
    case _: LeafNode =>
      (Seq(NodeData(id, node.prediction, node.impurity, node.impurityStats.stats,
        -1.0, -1, -1, SplitData(-1, Array.empty[Double], -1))),
        id)
  }
}