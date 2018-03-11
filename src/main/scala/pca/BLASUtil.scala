package pca

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.github.fommil.netlib.F2jBLAS

object BLASUtil extends Serializable {
  val f2jBLAS = new F2jBLAS

  def multiply(a: BDM[Double], b: BDV[Double]): BDV[Double] = {
    val c: BDV[Double] = new BDV[Double](Array.fill(a.rows)(0.0))
    gemv(1.0, a, b, 1.0, c)
    c
  }

  def gemv(alaph: Double,
           A: BDM[Double],
           x: BDV[Double],
           beta: Double,
           C: BDV[Double]): Unit = {
    val tStrA = if (A.isTranspose) "T" else "N"
    val mA = if (!A.isTranspose) A.rows else A.cols
    val nA = if (!A.isTranspose) A.cols else A.rows
    f2jBLAS.dgemv(tStrA, mA, nA, alaph, A.data, mA, x.data, 1, beta,
      C.data, 1)
  }

  def multiply(a: BDV[Double], b: BDM[Double]): BDM[Double] = {
    val vtoMatrix = a.toDenseMatrix
    multiply(vtoMatrix, b)
  }

  def multiply(a: BDM[Double], b: BDM[Double]): BDM[Double] = {
    val c: BDM[Double] = BDM.zeros(a.rows, b.cols)
    gemm(1.0, a, b, 1.0, c)
    c
  }

  def gemm(
            alaph: Double,
            A: BDM[Double],
            B: BDM[Double],
            beta: Double,
            C: BDM[Double]) = {
    val tAstr = if (A.isTranspose) "T" else "N"
    val tBstr = if (B.isTranspose) "T" else "N"
    val lda = if (!A.isTranspose) A.rows else A.cols
    val ldb = if (!B.isTranspose) B.rows else B.cols
    //val C:BDM[Double]=BDM.zeros(A.rows,B.cols)
    require(A.cols == B.rows,
      s"The columns of A don't match the rows of B. A: ${A.cols}, B: ${B.rows}")
    require(B.cols == C.cols,
      s"The columns of A don't match the rows of B. A: ${B.cols}, B: ${C.cols}")
    require(A.rows == C.rows,
      s"The columns of A don't match the rows of B. A: ${A.rows}, B: ${B.rows}")
    f2jBLAS.dgemm(tAstr, tBstr, A.rows, B.cols, A.cols, alaph, A.data, lda, B.data, ldb, beta, C.data, C.rows)
  }

  private def axpy(a: Double, x: BDV[Double], y: BDV[Double]): Unit = {
    val n = x.size
    f2jBLAS.daxpy(n, a, x.data, 1, y.data, 1)

  }
}
