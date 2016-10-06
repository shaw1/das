"""Ensemble weak-constraint 4D-Var.

Date Created:       23 November 2015
"""
import numpy as np
import scipy.linalg as la
import math

class enw4dvar(object):
    """Ensemble weak-constraint 4D-Var.

    This class contains the necessary functionality for implementing
    an ensemble-based weak-constraint four-dimensional variational
    (w4D-Var) data assimilation system using the incremental method.
    The linear system from the incremental method is solved using the
    conjugate gradient method in a matrix-free way, as only the
    matrix-vector product is required.

    The hybrid w4D-Var class in hyw4dvar.py is derived from this
    class, as all of the functionality can be extended to work for
    a hybrid model error covariance matrix.

    Attributes:
        model: A class specifying the underlying model dynamics.
    
        sqrtBdata: The band Cholesky factorization of the background
            error covariance matrix that stored in lower form. For
            example, if B is a 6 by 6 matrix with bandwidth 2, then
            B is stored as:
                b00 b11 b22 b33 b44 b55
                b10 b21 b32 b43 b54 *
                b20 b31 b42 b53 *   *
            A band diagonal matrix needs to be stored with shape
            (1, n).
    
        sigo_squared: Observation error variance vector.
    
        Qdata: A list of lists containing the data for each model
            error covariance matrix in the data assimilation scheme.
            Specifically, Qdata[i], for i = 0, 1, ..., window - 1 (see
            below for window), is a list of vectors q[j] such that the
            ith model error covariance equals the sum(q[j] * q[j].T).
    
        q: A list of model error bias vectors for each time level. In
            particular, q = [q1, q2, ..., qN].
    
        window: The number of time levels in a data assimilation
            cycle.
    
        obsloc: List of arrays to indicate the states which are
            observed. If the state vector x == [x0, x1, x2, x3] and
            obsloc == np.array([1, 2]), then y = h(x) returns
            y == [x1, x2].
    
        Cb: Symmetric band matrix representing the localization matrix
            for the ensemble model error covariance matrices. This is
            stored in the same way as Bdata.
    
        nobs: The number of observations, computed as the length of
            obsloc.
    
        sqrtCb: The band Cholesky factorization of Cb, computed from
            the specification of Cb.

        sigo: Observation error standard deviation.

    Functionality:
        Bprod: Background error covariance-vector product.
    
        sqrtBprod: Product of the Cholesky factor sqrtB with a vector.
    
        sqrtBprod_adj: Product of the transpose of the Cholesky factor
            sqrtB.T with a vector.
    
        Rinvprod: Inverse observation error covariance-vector product.

        sqrtRprod: The product of a square root matrix of R with a
            vector.

        sqrtRprod_adj: The product of the transpose of a square root
            matrix of R with a vector.

        sqrtRinvprod: The product of a square root of the inverse of
            the observation error covariance with a vector.

        sqrtRinvprod_adj: The product of the transpose a square root
            of the inverse observation error covariance with a vector.

        Cprod: Localization matrix-vector product.

        sqrtCprod: Cholesky factor of the localization matrix-vector
            product.

        sqrtCprod_adj: Transpose of the Cholesky factor of the
            localization matrix-vector product.
    
        Qprod: Model error covariance-vector product, where the model
            error covariance is assumed to be an ensemble covariance
            matrix with localization.
    
        sqrtQprod: Q^(1/2) * x.
    
        sqrtQprod_adj: Q^(T/2) * x.
    
        forecast: Nonlinear forecast model wrapper for
            self.model.forecast(x, ndt).
    
        forecast_tlm: Tangent linear forecast model wrapper for
            self.model.forecast_tlm(x, xd, ndt).
    
        forecast_adj: Adjoint forecast model wrapper for
            self.model.forecast_adj(x, yb, ndt).
    
        h: Observation operator.
    
        h_tlm: Tangent linear model of observation operator.
    
        h_adj: Adjoint of observation operator.
    
        blkh: Four-dimensional observation operator.
    
        blkh_tlm: Tangent linear model of four-dimensional
            observation operator.
    
        blkh_adj: Adjoint of four-dimensional observation operator.
    
        blkFprod: Product of block matrix F with a four-dimensional
            vector.
    
        blkFprod_adj: Product of block matrix F^T with a
            four-dimensional vector.
    
        blkFinvprod: Product of F^(-1) and a vector.
    
        blkFinvprod_adj: Product of F^(-T) and a vector.
    
        Pprod: Product of four-dimensional covariance model
            diag(B, Q1, ..., QN) with a four-dimensional vector.
    
        sqrtPprod: Product of the square root of the four-dimensional
            covariance model with a four-dimensional vector.
    
        sqrtPprod_adj: Product of the square root of the
            four-dimensional covariance model transpose with a
            four-dimensional vector.
    
        blkRinvprod: Product of the inverse four-dimensional
            observation error covariance with a four-dimensional
            observation space vector.

        blksqrtRinvprod: Product of the inverse of the square root of
            the four-dimensional observation error covariance with a
            four-dimensional vector.
    
        blksqrtRinvprod_adj: Product of the transpose of the inverse
            square root of the four-dimensional observation error
            covariance with a four-dimensional vector.

        blksqrtRprod: Product of the square root of the
            four-dimensional observation error covariance with a
            four-dimensional vector.
    
        blksqrtRprod_adj: Product of the transpose of the square root
            of the four-dimensional observation error covariance with
            a four-dimensional vector.
    
        Preconditioned_by_P_HessianProduct: Product of a
            four-dimensional state space vector with the Hessian of
            the cost functional preconditioned by sqrtP.

        Preconditioned_by_R_HessianProduct: Product of a
            four-dimensional observation space vector with the Hessian
            of the cost functional preconditioned by sqrtR.

        incremental: Incremental algorithm for the w4D-Var.

        _ConjugateGradient: Conjugate gradient algorithm used for the
            incremental method.

        _symmetric_band_product: Product of a band symmetric matrix
            stored in lower form.

        _lower_triangular_band_product: Computes the product of a band
            lower triangular matrix with a vector.

        _lower_triangular_band_product_adj: Computes the product of
            the transpose of a band lower triangular matrix with a
            vector.
    """
    def __init__(self, model, sqrtBdata, sigo_squared, Qdata, q, window, obsloc, \
                 Cb):
        """Initializes the class object to the specified inputs.

        Descriptions of each data member is provided in the comments
        above.
        """
        self.model = model
        self.sqrtBdata = sqrtBdata
        self.sigo_squared = sigo_squared
        self.Qdata = Qdata
        self.q = q
        self.window = window
        self.obsloc = obsloc
        self.Cb = Cb

        # Computation of data members derived from arguments
        self.nobs = len(obsloc[0])
        self.sqrtCb = la.cholesky_banded(Cb, lower=True)
        self.sigo = np.sqrt(sigo_squared)

    def Bprod(self, x):
        """Computes the product B * x.

        The background error covariance matrix is set using the band matrix
        Bdata.

        Argument:
            x: Vector to multiply.

        Returns:
            Matrix-vector product.
        """
        y = self._lower_triangular_band_product_adj(self.sqrtBdata, x)
        y = self._lower_triangular_band_product(self.sqrtBdata, y)

        return y

    def sqrtBprod(self, x):
        """Computes the product B^(1/2) * x.

        Using the Cholesky factorization of B, stored as sqrtBdata, the
        product of B^(1/2) with x is computed.

        Argument:
            x: Vector to multiply.

        Returns:
            Matrix-vector product B^(1/2) * x.
        """
        return self._lower_triangular_band_product(self.sqrtBdata, x)

    def sqrtBprod_adj(self, x):
        """Computes the product B^(T/2) * x.

        Using the Cholesky factorization of B, stored as sqrtBdata,
        the product of B^(T/2) with x is computed.

        Argument:
            x: Vector to multiply.

        Returns:
            Matrix-vector product B^(T/2) * x.
        """
        return self._lower_triangular_band_product_adj(self.sqrtBdata, x)
        
    def Rinvprod(self, x):
        """Product of R^(-1) with a vector x.

        The observation error covariance matrix R is assumed to be
        diagonal.

        Argument:
            x: The vector to multiply.

        Returns:
            Product of R^(-1) with x.
        """
        return x / self.sigo_squared

    def sqrtRprod(self, x):
        """Product of R^(1 / 2) with a vector x.

        The observation error covariance matrix R is assumed to be
        diagonal.

        Argument:
            x: The vector to multiply of length nobs.

        Returns:
            Product of R^(1 / 2) with x.
        """
        return x * self.sigo

    def sqrtRprod_adj(self, x):
        """Product of R^(T / 2) with a vector x.

        The observation error covariance matrix R is assumed to be
        diagonal.

        Argument:
            x: The vector to multiply of length nobs.

        Returns:
            Product of R^(T / 2) with x.
        """
        return x * self.sigo

    def sqrtRinvprod(self, x):
        """Product of R^(-1) with a vector x.

        The observation error covariance matrix R is assumed to be
        diagonal.

        Argument:
            x: The vector to multiply of length nobs.

        Returns:
            Product of R^(-1) with x.
        """
        return x / self.sigo

    def sqrtRinvprod_adj(self, x):
        """Product of R^(-T) with a vector x.

        The observation error covariance matrix R is assumed to be
        diagonal.

        Argument:
            x: The vector to multiply of length nobs.

        Returns:
            Product of R^(-T) with x.
        """
        return x / self.sigo

    def Cprod(self, x):
        """Localization matrix-vector product.

        The localization matrix is a symmetric band matrix stored in lower
        form.

        Argument:
            x: The vector to multiply.

        Returns:
            y: Product of localization matrix with x.
        """
        return self._symmetric_band_product(self.Cb, x)

    def sqrtCprod(self, x):
        """Cholesky factor of the localization matrix-vector product.
        
        Argument:
            x: The vector to multiply.

        Returns:
            Matrix-vector product C^(1/2) * x.
        """
        return self._lower_triangular_band_product(self.sqrtCb, x)

    def sqrtCprod_adj(self, x):
        """The product C^(T/2) * x.

        The matrix C^(T/2) is the transpose of the Cholesky factor of
        the banded localization matrix C.

        Argument:
            x: The vector to multiply.

        Returns:
            Matrix-vector product.
        """
        return self._lower_triangular_band_product_adj(self.sqrtCb, x)

    def Qprod(self, x, i):
        """Product of Q[i] with a vector x.

        The model error covariance is the sum of the outer product of
        the vectors stored in self.Qdata[i] with localization applied.
        Utilizing the special structure of Q[i], the computation is
        (C \circ Qe[i]) * x == sum((C * (x \circ q[j])) \circ q[j])
        i is the index corresponding to Q[i] for time t_{i + 1} and q[j]
        come from self.Qdata[i].

        Arguments:
            x: Vector to multiply.
            i: Time index.

        Returns:
            Matrix-vector product.
        """
        return np.sum([self.Cprod(x * q) * q for q in self.Qdata[i]], axis=0)

    def sqrtQprod(self, x, i):
        """Product of Q^(1 / 2) with x, using covariance localization.

        The matrix is a rectangular matrix of size n by (n * Ne), and is
        (C * Qe)^(1 / 2) = [diag(x_1) C^(1 / 2), ..., diag(x_Ne) C^(1 / 2)].
        
        Arguments:
            x: The vector to multiply of length n * Ne.
            i: Index indicating which model error covariance to use.

        Returns:
            y: Product with x, length of n.
        """
        Ne_times_n = len(x)
        Qdata = self.Qdata[i]
        sqrtCprod = self.sqrtCprod
        Ne = len(Qdata)
        
        n = Ne_times_n / len(Qdata)
        y = np.zeros(n)

        for j in xrange(Ne):
            y += sqrtCprod(x[j * n : (j + 1) * n]) * Qdata[j]

        return y

    def sqrtQprod_adj(self, x, i):
        """Product of Q^(T / 2) with x, using covariance localization.

        The matrix is a rectangular matrix of size (n * Ne) by n, and is
        (C * Qe)^(1 / 2) = [C^(T / 2) diag(x_1); ...; C^(T / 2) diag(x_Ne)].
        
        Arguments:
            x: The vector to multiply of length n.
            i: Index indicating which model error covariance to use.

        Returns:
            y: Product with x, length of n * Ne.
        """
        n = len(x)
        Qdata = self.Qdata[i]
        Ne = len(Qdata)
        sqrtCprod_adj = self.sqrtCprod_adj

        Ne_times_n = n * Ne
        y = np.zeros(Ne_times_n)

        for j in xrange(Ne):
            y[j * n : (j + 1) * n] = sqrtCprod_adj(Qdata[j] * x)

        return y

    def forecast(self, x, ndt=1):
        """Nonlinear forecast model.

        Essentially a wrapper to call the forecast function stored in
        the data member self.model.

        Arguments:
            x: Vector to forecast.
            nd:- Number of time-steps to forecast, default is 1

        Returns:
            The nonlinear forecast of x.
        """
        return self.model.forecast(x, ndt)

    def forecast_tlm(self, x, xd, ndt=1):
        """Tangent linear forecast model.

        Essentially a wrapper to call the tangent linear model at x
        in the direction of xd stored in the data member self.model.

        Arguments:
            x: State to set the tangent linear model.
            xd: Direction vector.
            ndt: Number of time-steps to forecast, default is 1

        Returns:
            Tangent linear model forecast of xd.
        """
        return self.model.forecast_tlm(x, xd, ndt)

    def forecast_adj(self, x, yb, ndt=1):
        """Adjoint forecast model.

        Essentially a wrapper to call the adjoint forecast model at x
        in the direction of yb stored in the data member self.model.
        
        Arguments:
            x: State to set the adjoint model.
            yb: Direction vector.
            ndt: Number of time-steps, default is 1

        Returns:
            Adjoint forecast model forecast of yb.
        """
        return self.model.forecast_adj(x, yb, ndt)

    def h(self, x, i):
        """Observation operator.

        The observation operator is assumed linear and observations
        are of the true state at the locations stored in self.obsloc.

        Arguments:
            x: Input vector of size n == self.model.n.
            i: Index indicating which observation operator to use

        Returns:
            Observation equivalent of x of length self.nobs.
        """
        return x[self.obsloc[i]]

    def h_tlm(self, x, xd, i):
        """Tangent linear observation model.

        Since the observation operator is assumed linear, the input
        x is unused and the output is the result of H * xd, where H
        is the tangent linear model of h. Since observations are
        taken at self.obsloc, the product H * xd returns the
        components of xd at self.obsloc.

        Arguments:
            x: State to set the tangent linear model.
            xd: Direction vector of size n == self.model.n.
            i: Index indicating which observation operator to use

        Returns:
            Tangent linear observation model in the direction of xd.
        """
        return xd[self.obsloc[i]]

    def h_adj(self, x, yb, i):
        """Adjoint of the observation operator.

        Since the observation operator is linear, the input x is
        unused and the output is the result of H.T * yb, where H is
        the tangent linear model of h. Since observations are taken
        at self.obsloc, the product xb = H.T * yb contains the values
        of yb at the locations specified by self.obsloc and is zero
        otherwise.

        Arguments:
            x: State to set the adjoint model.
            yb: Direction vector of size self.nobs == len(self.obsloc)
            i: Index indicating which observation operator to use

        Returns:
            Adjoint observation operator in the direction of yb.
        """
        xb = np.zeros(x.shape)
        xb[self.obsloc[i]] = yb
        
        return xb

    """The rest of the methods apply to x being a four-dimensional state."""
    def blkh(self, x):
        """Four-dimensional observation operator.

        Since the three-dimensional observation operator is linear, this
        four-dimensional operator is linear for each block of x, where
        x = [x[0], x[1], ..., x[window - 1]].

        Argument:
            x: Four-dimensional state vector.

        Returns:
            y: Four-dimensional observation space equivalent of x.
        """
        nobs = self.nobs
        window = self.window
        n = self.model.n
        
        y = np.empty(window * nobs)

        for j in xrange(window):
            y[j * nobs : (j + 1) * nobs] = self.h(x[j * n : (j + n) * n], j)

        return y
                
    def blkh_tlm(self, x, xd):
        """Four-dimensional tangent linear observation model.

        Arguments:
            x: Four-dimensional state of length window * n to set the
                tangent linear model.
            xd: Direction vector of length window * n.

        Returns:
            yd: Four-dimensional tangent linear observation model in
                the direction of xd of length window * nobs.
        """
        nobs = self.nobs
        window = self.window
        n = self.model.n
        
        yd = np.empty(window * nobs)

        for j in xrange(window):
            yd[j * nobs : (j + 1) * nobs] = self.h_tlm(x[j * n : (j + 1) * n], \
            xd[j * n : (j + 1) * n], j)

        return yd
    
    def blkh_adj(self, x, yb):
        """Four-dimensional adjoint observation model.

        Arguments:
            x: Four-dimensional state of length window * n to set the
                adjoint model.
            yb: Direction vector of length window * nobs.

        Returns:
            xb: Four-dimensional adjoint observation model in the
                direction of yb of length window * n.
        """
        nobs = self.nobs
        window = self.window
        n = self.model.n
        
        xb = np.empty(window * n)

        for j in xrange(window):
            xb[j * n : (j + 1) * n] = self.h_adj(x[j * n : (j + 1) * n], \
            yb[j * nobs : (j + 1) * nobs], j)

        return xb

    def blkFprod(self, x, xd):
        """Performs multipliaction of blkF with 4D-vector xd.

        Arguments:
            x: Four-dimensional vector to evaluate TLMs.
            xd: Vector to multiply with blkF.

        Returns:
            y: Result of multiplying xd.
        """
        n = self.model.n
        window = self.window
        
        y = np.empty(window * n)
        y[0 : n] = xd[0 : n]

        for i in xrange(1, window):
            behind = np.arange((i - 1) * n, i * n)
            current = np.arange(i * n, (i + 1) * n)
        
            y[current] = self.forecast_tlm(x[behind], y[behind]) + xd[current]

        return y

    def blkFprod_adj(self, x, yb):
        """Performs multipliaction of blkF.T with 4D-vector yb.

        Arguments:
            x: Four-dimensional vector to evaluate adjoints.
            yb: Vector to multiply with blkF.T.

        Returns:
            y: Result of multiplying yb.
        """
        n = self.model.n
        window = self.window
        
        y = np.empty(window * n)
        y[(window - 1) * n : window * n] = yb[(window - 1) * n : window * n]

        for i in xrange(window - 2, -1, -1):
            current = np.arange(i * n, (i + 1) * n)
            ahead = np.arange((i + 1) * n, (i + 2) * n)

            y[current] = self.forecast_adj(x[current], y[ahead]) + yb[current]

        return y

    def blkFinvprod(self, x, xd):
        """Performs multipliaction of F^(-1) with 4D-vector xd.

        Arguments:
            x: Four-dimensional vector to evaluate TLMs.
            xd: Vector to multiply.

        Returns:
            y: Result of multiplying xd.
        """
        n = self.model.n
        window = self.window

        y = np.empty(window * n)
        y[0 : n] = xd[0 : n]

        for i in xrange(1, window):
            behind = np.arange((i - 1) * n, i * n)
            current = np.arange(i * n, (i + 1) * n)

            y[current] = xd[current] - self.forecast_tlm(x[behind], xd[behind])

        return y

    def blkFinvprod_adj(self, x, yb):
        """Performs multipliaction of F^(-T) with 4D-vector yb.

        Arguments:
            x: Four-dimensional vector to evaluate TLMs.
            yb: Vector to multiply.

        Returns:
            y: Result of multiplying yb.
        """
        n = self.model.n
        window = self.window

        y = np.empty(window * n)
        y[(window - 1) * n : window * n] = yb[(window - 1) * n : window * n]

        for i in xrange(0, window - 1):
            current = np.arange(i * n, (i + 1) * n)
            ahead = np.arange((i + 1) * n, (i + 2) * n)

            y[current] = yb[current] - self.forecast_adj(x[current], yb[ahead])

        return y

    def Pprod(self, x):
        """Product of four-dimensional covariance model with a vector.

        Argument:
            x: Vector of length window * n to multiply.

        Returns:
            y: Matrix-vector product.
        """
        n = self.model.n
        window = self.window

        y = np.empty(window * n)

        y[0 : n] = self.Bprod(x[0 : n])

        for i in xrange(1, window):
            current = np.arange(i * n, (i + 1) * n)
            
            y[current] = self.Qprod(x[current], i - 1)

        return y

    def sqrtPprod(self, x):
        """Product of P^(1 / 2) with a vector.

        Since the square root matrix for Q is a wide rectangle matrix
        of dimension n by n * Ne, the input vector x is a large
        vector of size (window - 1) * n * Ne + n. The output vector
        will be of the dimension of a four-dimensional vector, that
        is, of length window * n.

        Argument:
            x: Vector of multiply.

        Returns:
            y: Matrix-vector product.
        """
        n = self.model.n
        window = self.window
        Ne = len(self.Qdata[0])
        Ne_times_n = Ne * n
        
        y = np.zeros(window * n)

        y[0 : n] = self.sqrtBprod(x[0 : n])

        for i in xrange(window - 1):
            x_index = np.arange(n + i * Ne_times_n, n + (i + 1) * Ne_times_n)
            y_index = np.arange((i + 1) * n, (i + 2) * n)
            
            y[y_index] = self.sqrtQprod(x[x_index], i)

        return y

    def sqrtPprod_adj(self, x):
        """Product of P^(T / 2) with a vector.

        Since the adjoint of the square root matrix for Q is a tall
        rectangle matrix of dimension n * Ne by n, the input vector x
        is of the length of a four-dimensional vector of length
        window * n. The output vector will have a very large size of
        length (window - 1) * n * Ne + n.

        Argument:
            x: Four-dimensional vector to multiply.

        Returns:
            y: Matrix-vector product.
        """
        n = self.model.n
        window = self.window
        Ne = len(self.Qdata[0])
        Ne_times_n = Ne * n

        y = np.empty(n + (window - 1) * n * Ne)

        y[0 : n] = self.sqrtBprod_adj(x[0 : n])

        for i in xrange(window - 1):
            x_index = np.arange((i + 1) * n, (i + 2) * n)
            y_index = np.arange(n + i * Ne_times_n, n + (i + 1) * Ne_times_n)

            y[y_index] = self.sqrtQprod_adj(x[x_index], i)

        return y

    def blkRinvprod(self, y):
        """Inverse four-dimensional observation error covariance product.

        Argument:
            y: Vector of length window * nobs to multiply.

        Returns:
            output: Result of matrix multiplication.
        """
        nobs = self.nobs
        output = np.empty(len(y))

        for i in xrange(self.window):
            current = np.arange(i * nobs, (i + 1) * nobs)

            output[current] = self.Rinvprod(y[current])

        return output

    def blksqrtRinvprod(self, y):
        """Inverse square root 4D observation error covariance.

        Argument:
            y: Vector of length window * nobs to multiply.

        Returns:
            output: Result of matrix multiplication.
        """
        nobs = self.nobs
        output = np.empty(len(y))

        for i in xrange(self.window):
            current = np.arange(i * nobs, (i + 1) * nobs)

            output[current] = self.sqrtRinvprod(y[current])

        return output

    def blksqrtRinvprod_adj(self, y):
        """Inverse square root 4D observation error covariance adjoint.

        Argument:
            y: Vector of length window * nobs to multiply.

        Returns:
            output: Result of matrix multiplication.
        """
        nobs = self.nobs
        output = np.empty(len(y))

        for i in xrange(self.window):
            current = np.arange(i * nobs, (i + 1) * nobs)

            output[current] = self.sqrtRinvprod_adj(y[current])

        return output

    def blksqrtRprod(self, y):
        """Square root 4D observation error covariance vector product.

        Argument:
            y: Vector of length window * nobs to multiply.

        Returns:
            output: Result of matrix multiplication.
        """
        nobs = self.nobs
        output = np.empty(len(y))

        for i in xrange(self.window):
            current = np.arange(i * nobs, (i + 1) * nobs)

            output[current] = self.sqrtRprod(y[current])

        return output

    def blksqrtRprod_adj(self, y):
        """Square root 4D observation error covariance adjoint product.

        Argument:
            y: Vector of length window * nobs to multiply.

        Returns:
            output: Result of matrix multiplication.
        """
        nobs = self.nobs
        output = np.empty(len(y))

        for i in xrange(self.window):
            current = np.arange(i * nobs, (i + 1) * nobs)

            output[current] = self.sqrtRprod_adj(y[current])

        return output
    
    def Preconditioned_by_P_HessianProduct(self, x, xg):
        """The Hessian of the preconditioned cost functional product.

        The Hessian of the cost functional preconditioned by sqrtP
        I + sqrtP.T * blkF.T * blkH.T * Rinv * blkH * blkF * sqrtP
        is the coefficient matrix of the linear system for the
        preconditioned analysis.

        Arguments:
            x: Four-dimensional vector of size window * n to multiply
            xg: Four-dimensional guess state

        Returns:
            y: Result of Hessian-vector product.
        """
        y = self.sqrtPprod(x)
        y = self.blkFprod(xg, y)
        y = self.blkh_tlm(xg, y)
        y = self.blkRinvprod(y)
        y = self.blkh_adj(xg, y)
        y = self.blkFprod_adj(xg, y)
        y = self.sqrtPprod_adj(y)
        y += x

        return y

    def Preconditioned_by_R_HessianProduct(self, x, xg):
        """The Hessian of the preconditioned cost functional product.

        The Hessian of the preconditioned cost functional
        sqrtRinv * blkH * F * P * F.T * blkH.T * sqrtRinv.T + I
        is the coefficient matrix of the linear system for the
        preconditioned analysis.

        Arguments:
            x: Four-dimensional vector of size window * nobs
            xg: Four-dimensional guess state

        Returns:
            y: Result of matrix-vector product.
        """
        y = self.blksqrtRinvprod_adj(x)
        y = self.blkh_adj(xg, y)
        y = self.blkFprod_adj(xg, y)
        y = self.Pprod(y)
        y = self.blkFprod(xg, y)
        y = self.blkh_tlm(xg, y)
        y = self.blksqrtRinvprod(y)
        y += x

        return y

    def incremental(self, xb, xg_list, y_list, tol, option):
        """The incremental algorithm for solving w4D-Var.

        Two options for minimizing the cost functional are available, where
        the linear system can be solved in state space (option == "StateSpace")
        using sqrtP preconditioning or in observation space (option ==
        "ObsSpace") which uses sqrtR preconditioning.

        Inputs:
            xb: Background state.
            xg_list: List of guess states.
            y_list: List of observations.
            tol: Tolerance for conjugate gradient method.
            option: Solve in state space or in observation space, using
                preconditioning.

        Returns:
            xa_list: List of analysis states.
            k: Number of conjugate gradient iterations to convergence.
        """        
        window = self.window
        n = self.model.n
        nobs = self.nobs
        total_nobs = window * nobs
        
        xg = np.array(xg_list).flatten(0)           # Vectorized guess state
        q = np.array([np.zeros(n)] + \
                     [self.q[j] for j in xrange(window - 1)]).flatten(0) # Bias
        d = np.empty(total_nobs)                    # Observed minus guess
        g = np.empty(n * window)                    # Guess vector

        # Sets four-dimensional vector d = y - h(xg)
        for j in xrange(0, window): 
            temp = np.arange(j * nobs, (j + 1) * nobs)
        
            d[temp] = y_list[j] - self.h(xg_list[j], j)

        # Set guess vector
        g[0 : n] = xb - xg_list[0]

        for j in xrange(1, window):
            current = np.arange(j * n, (j + 1) * n)

            g[current] = self.forecast(xg_list[j - 1]) - xg_list[j]

        if option == "StateSpace":
            # Conjugate gradient method for solving the state space linear
            # system
            # [blkFinv.T * Pinv * blkFinv + blkH.T * blkRinv * blkH] * deltax
            # = blkFinv.T * Pinv * [g + q] + blkH.T * blkRinv * d
            # by preconditioning using sqrtP to get
            # [I + sqrtP.T * blkF.T * blkH.T * Rinv * blkH * blkF * sqrtP] *
            # chi = sqrtP.T * blkF.T * blkH.T * blkRinv * [d - blkH * blkF *
            # [g + q]], where deltax = blkF * [sqrtP * chi + g + q]
            b = g + q
            b = self.blkFprod(xg, b)
            b = self.blkh_tlm(xg, b)
            b = d - b
            b = self.blkRinvprod(b)
            b = self.blkh_adj(xg, b)
            b = self.blkFprod_adj(xg, b)
            b = self.sqrtPprod_adj(b)

            chi0 = np.zeros(b.shape) # Guess for chi
            (chi, k) = self._ConjugateGradient(chi0, \
                       self.Preconditioned_by_P_HessianProduct, b, tol, xg)

            deltax = self.blkFprod(xg, self.sqrtPprod(chi) + g + q)
        elif option == "ObsSpace":
            # Conjugate gradient method for solving the observation space
            # linear system
            # [blkH * F * P * F.T * blkH.T + R] * z = d - blkH * F * [g + q]
            # by preconditioning using sqrtR to get
            # [sqrtRinv * blkH * F * P * F.T * blkH.T * sqrtRinv.T + I] * chi =
            # sqrtRinv * (d - blkH * blkF * [g + q]), where z is
            # sqrtRinv.T * chi
            temp = self.blkFprod(xg, g + q)
            b = self.blkh_tlm(xg, temp)
            b = d - b
            b = self.blksqrtRinvprod(b)
        
            chi0 = np.zeros(b.shape)
            (chi, k) = self._ConjugateGradient(chi0, \
                       self.Preconditioned_by_R_HessianProduct, b, tol, xg)

            z = self.blksqrtRinvprod_adj(chi)
            deltax = self.blkh_adj(xg, z)
            deltax = self.blkFprod_adj(xg, deltax)
            deltax = self.Pprod(deltax)
            deltax = self.blkFprod(xg, deltax)
            deltax += temp
        else:
            raise ValueError("Choose a valid option.")

        xa = xg + deltax
        xa_list = [xa[j * n : (j + 1) * n] for j in xrange(0, window)]

        return (xa_list, k)

    def _ConjugateGradient(self, x0, matrix_vector_product, b, tol, xg):
        """Conjugate gradient method for use in incremental function.

        Conjugate gradient method to solve a linear system within tolerance.
        This code is written for use with the Hessian of the w4D-Var cost
        functional, which requires the four-dimensional guess state xg as an
        argument.

        Arguments:
            x0: Initial guess to initiate conjugate gradient method
            matrix_vector_product: Function reference to matrix-vector product
            b: Right-hand-side of linear system
            tol: Error tolerance
            xg: Four-dimensional guess state

        Returns:
            x: Solution to linear system
            k: Number of iterations to convergence within tolerance
        """
        x = x0.copy()
        r = b - matrix_vector_product(x, xg)
        p = r.copy()
        k = 0
        n = len(x0)
        r_old_scalar = np.dot(r, r)
        r_new_scalar = r_old_scalar
        
        while(k < n and math.sqrt(r_new_scalar) >= tol):
            A_times_p = matrix_vector_product(p, xg)
            alpha = r_old_scalar / np.dot(p, A_times_p)
            x += alpha * p
            r -= alpha * A_times_p
            r_new_scalar = np.dot(r, r)
            beta = r_new_scalar / r_old_scalar
            p *= beta
            p += r
            r_old_scalar = r_new_scalar
            
            k += 1

        return (x, k)

    def _symmetric_band_product(self, Lb, x):
        """Product of a symmetric band matrix with a vector.

        The matrix is assumed to be stored in lower form.

        """
        (p_plus_one, n) = Lb.shape
        p = p_plus_one - 1   # Bandwidth
        y = np.empty(n)

        for i in xrange(n):
            temp = 0.0

            for j in xrange(max(0, i - p), i):
                temp += Lb[i - j, j] * x[j]

            for j in xrange(i, min(i + p_plus_one, n)):
                temp += Lb[j - i, i] * x[j]

            y[i] = temp

        return y

    def _lower_triangular_band_product(self, Lb, x):
        """Computes the product L * x, where L is lower triangular.

        Arguments:
            Lb: Banded lower triangular matrix.
            x: Vector to multiply.

        Returns:
            y: Matrix-vector product L * x.
        """
        (p_plus_one, n) = Lb.shape
        p = p_plus_one - 1   # Bandwidth of the matrix
        y = np.empty(n)

        for i in xrange(p):
            temp = 0.0

            for j in xrange(i + 1):
                temp += Lb[i - j, j] * x[j]

            y[i] = temp

        for i in xrange(p, n):
            temp = 0.0

            for j in xrange(i - p, i + 1):
                temp += Lb[i - j, j] * x[j]

            y[i] = temp

        return y

    def _lower_triangular_band_product_adj(self, Lb, x):
        """Computes the product Lb.T * x, where Lb is lower triangular.

        Arguments:
            Lb: Banded lower triangular matrix.
            x: Vector to multiply.

        Returns:
            y: Matrix-vector product L.T * x.
        """
        (p_plus_one, n) = Lb.shape
        p = p_plus_one - 1   # Bandwidth of the matrix
        y = np.empty(n)

        for i in xrange(n - p):
            temp = 0.0

            for j in xrange(i, i + p_plus_one):
                temp += Lb[j - i, i] * x[j]

            y[i] = temp

        for i in xrange(n - p, n):
            temp = 0.0

            for j in xrange(i, n):
                temp += Lb[j - i, i] * x[j]

            y[i] = temp

        return y
