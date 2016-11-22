package org.apache.spark.ml.tree


/**
 * Info for one [[Node]] in a tree ensemble
 *
 * @param treeID  Tree index
 * @param nodeData  Data for this node
 */
case class EnsembleNodeData(
                             val treeID: Int,
                             val nodeData: NodeData)

object EnsembleNodeData {
  /**
   * Create [[EnsembleNodeData]] instances for the given tree.
   *
   * @return Sequence of nodes for this tree
   */
  def build(tree: DecisionTreeModel, treeID: Int): Seq[EnsembleNodeData] = {
    val (nodeData: Seq[NodeData], _) = NodeData.build(tree.rootNode, 0)
    nodeData.map(nd => EnsembleNodeData(treeID, nd))
  }
}
