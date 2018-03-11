package pca

import java.util.Arrays

import JSci.maths.statistics.{FDistribution, NormalDistribution}
import breeze.linalg.{DenseMatrix, DenseVector, eigSym, fliplr, flipud}

import scala.util.control.Breaks.{break, breakable}

class PCA(
           private var data: DenseMatrix[Double],
           private var prContributeRate: Double) {

  private var featureValues: DenseVector[Double] = _

  //The vector of eigenvalues of covariance matrix
  private def run(): PCAModel = {
    val (trainData,mean,std) = standard(data)
    val (loadMatrix, prinVector, k, cov) = computePrinciple(trainData, prContributeRate)
    val m = featureValues.length
    val n = data.rows
    val t2 = computeT2scl(0.95, k, n)
    val qscl = computeQscl(k, m)
    new PCAModel(prinVector, loadMatrix,mean,std, t2, qscl)
  }

  private def standard(matrix: DenseMatrix[Double]):(DenseMatrix[Double],DenseVector[Double],DenseVector[Double]) = {
    val data = matrix.toArray //矩阵优先按列存储
    val row = matrix.rows
    val col = matrix.cols
    val dataArr = Array.fill[Array[Double]](col)(Array(0.0))
    val col_mean_Arr = Array.fill[Double](col)(0.0)
    val col_std_Arr = Array.fill[Double](col)(0.0)
    //把每一列数据转换成数组，求取平均值、标准差
    for (i <- 0 until col) {
      val arr = Arrays.copyOfRange(data, i * row, i * row + row)
      val mean = arr.sum / (1.0 * row)
      val variance = arr.map(x => math.pow((x - mean), 2.0)).reduce(_ + _) / (row - 1)
      val std = math.sqrt(variance)
      dataArr(i) = arr.map(x => (x - mean) / std)
      col_mean_Arr(i)=mean
      col_std_Arr(i)=std

    }
    (new DenseMatrix[Double](row, col, dataArr.flatten),
      new DenseVector[Double](col_mean_Arr),
      new DenseVector[Double](col_std_Arr)
    )
  }

  private def computeT2scl(
                            p: Double,
                            k: Double,
                            n: Double): Double = {
    val coefficient = k * (n - 1) * (n + 1) / (n * (n - k))
    val f = new FDistribution(k, n - k)
      .inverse(0.95)
    coefficient * f
  }

  private def computeQscl(
                           k: Double,
                           m: Double): Double = {
    val featureValueArr = featureValues.data
    var theta1 = 0.0
    var theta2 = 0.0
    var theta3 = 0.0
    for (i <- k.toInt until m.toInt) {
      theta1 = theta1 + featureValueArr(i)
      theta2 = theta2 + math.pow(featureValueArr(i), 2.0)
      theta3 = theta3 + math.pow(featureValueArr(i), 3.0)
    }
    val h0 = 1 - 2 * theta1 * theta3 / (3 * theta2 * theta2)
    val normalDistributition = new NormalDistribution(0.0, 1.0)
    val ca = normalDistributition
      .inverse(0.95)
    val param1 = (
      1 + h0 * ca * math.sqrt(2 * theta2) / theta1
        + theta2 * h0 * (h0 - 1) / (theta1 * theta1))
    val qscl = theta1 * math.pow(param1, 1 / h0)
    qscl
  }

  private def computePrinciple(
                                matrix: DenseMatrix[Double],
                                contriButionRate: Double)
  : (DenseMatrix[Double], DenseVector[Double], Int, DenseMatrix[Double]) = {
    val col = matrix.cols
    val row = matrix.rows
    val covarianceMatrix = BLASUtil.multiply(matrix.t, matrix)
      .:*=(1.0 / (row - 1))
    // 1/m*X^T*X
    val feature = eigSym(covarianceMatrix)
    featureValues = flipud(feature.eigenvalues)
    //与特征值对应
    val featureVectors = fliplr(feature.eigenvectors).copy
    val rowNum = covarianceMatrix.rows
    val pNum = selectPNum(contriButionRate, featureValues)
    val prinVector = new DenseVector(Arrays.copyOfRange(featureValues.toArray, 0, pNum))
    //负荷矩阵
    val LoadMatrix = new DenseMatrix[Double](
      rowNum,
      pNum,
      Arrays.copyOfRange(featureVectors.toArray, 0, rowNum * pNum))
    (LoadMatrix, prinVector, pNum, covarianceMatrix)
  }

  private def selectPNum(
                          varContribution: Double,
                          sValues: DenseVector[Double]): Int = {
    var num = 1
    var sum = 0.0
    val svSum = sValues.sum
    breakable {
      for (i <- sValues) {
        sum += i
        if (sum * 1.0 / svSum * 1.0 > varContribution) break
        else num += 1
      }
    }
    num
  }
}

object PCA {
  def train(data: DenseMatrix[Double], contributeRate: Double): PCAModel = {
    new PCA(data, contributeRate).run()
  }
  //指定数据集和方差贡献率
  def train(data: DenseMatrix[Double]): PCAModel = {
    new PCA(data, 0.95).run()
  }
}