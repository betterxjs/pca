package pca

import java.util.Arrays

import breeze.linalg.{DenseMatrix, DenseVector, diag, pinv}

/**
  * Created by xiejunshuai on 2017/11/20.
  */
class PCAModel(val prinVector: DenseVector[Double],
               val loadMatrix: DenseMatrix[Double],
               val mean:DenseVector[Double],
               val std:DenseVector[Double],
               val T2scl: Double,
               val Qscl: Double) {

  def estimate(data: DenseVector[Double]): Double = {
    val indicate = this.computeT2(data)
    if (indicate > T2scl) 1.0 else 0.0
  }

  def computeT2(data: DenseVector[Double]): Double = {
    val pVtoBrzMatrix = diag(prinVector)
    //对角阵S
    val pinvS = pinv(pVtoBrzMatrix)
    //pinv(S)
    val pT = loadMatrix.t
    //v=pvp'x'
    val pv = BLASUtil.multiply(loadMatrix, pinvS)
    val pvpt = BLASUtil.multiply(pv, pT)
    val pvpTxT = BLASUtil.multiply(pvpt, data)
    val result = data.dot(pvpTxT)
    // t2=xpvp'x'
    result
  }

  def computeQ(data: DenseVector[Double]): Double = {
    val n = data.size
    val i = DenseMatrix.eye[Double](n)
    //p*p^t
    val ppt = BLASUtil.multiply(loadMatrix, loadMatrix.t)
    //I-PP'
    val subtractMatrix = i + (ppt.:*=(-1.0))

    val row = subtractMatrix.rows
    val col = subtractMatrix.cols

    val vector = BLASUtil.multiply(subtractMatrix, data)
    val result = data.dot(vector) //Q=x(I-PP')x'
    result
  }

  def computeContributionRate(data: DenseVector[Double])
  : Array[Double] = {
    val n = data.size
    val iMatrix = DenseMatrix.eye[Double](n)
    val iArr = Array.fill[DenseMatrix[Double]](n)(DenseMatrix.zeros(n, 1))

    for (j <- 0 until iMatrix.cols) {
      iArr(j) = new DenseMatrix[Double](n, 1, iMatrix(::, j).toArray)
    }
    val t2ContriArr = new Array[Double](data.size)
    //S
    val pVtoBrzMatrix = diag(prinVector)
    val pinvS = pinv(pVtoBrzMatrix)
    val pT = loadMatrix.t
    // loadMatrix.multiply(pinvS).multiply(pT)
    val d = BLASUtil.multiply(BLASUtil.multiply(loadMatrix, pinvS), pT)
    //T2 Contributition Rate
    for (i <- 0 until t2ContriArr.length) {
      //Contripart = d *I(:,i)*I(:,i)' * x
      val part0 = BLASUtil.multiply(d, iArr(i))
      //d.multiply(iArr(i))
      val part1 = BLASUtil.multiply(part0, iArr(i).t)
      val contriPart = BLASUtil.multiply(part1, data)
      t2ContriArr(i) = data.dot(contriPart) //BLASUtil.dot(data, latterPart)
    }
    t2ContriArr
  }

  def standard(matrix: DenseMatrix[Double]) = {
    val data = matrix.toArray
    //优先按列存储
    val row = matrix.rows
    val col = matrix.cols
    //println(col)
    val dataArr = Array.fill[Array[Double]](col)(Array(0.0))
    for (i <- 0 until col) {
      val arr = Arrays.copyOfRange(data, i * row, i * row + row)
      val mean_i = mean.data(i)
      //println(mean)
      //val variance = arr.map(x => math.pow((x - mean_i), 2.0))
      //              .reduce(_ + _) / (row - 1)
      val std_i =std.data(i)
      dataArr(i) = arr.map(x => (x - mean_i) / std_i)
    }
    new DenseMatrix[Double](row, col, dataArr.flatten)
  }

}
