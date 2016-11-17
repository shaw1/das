"""Ensemble four-dimensional variational data assimilation.

Date Created:       1 February 2016
"""
import numpy as np
import scipy.linalg as la
import math

def ConjugateGradient(x0, matrix_vector_product, b, tol, xb):
    """Conjugate gradient method for use in incremental function.

    Conjugate gradient method to solve a linear system within
    tolerance. This code is written for use with the Hessian of
    the 4D-Var cost functional, which requires the background
    state xb as an argument.

    Arguments:
        x0: Initial guess to initiate conjugate gradient method
        matrix_vector_product: Function reference to matrix-vector
            product
        b: Right-hand-side of linear system
        tol: Error tolerance
        xb: Three-dimensional background state

    Returns:
        x: Solution to linear system
        k: Number of iterations to convergence within tolerance
    """
    x = x0.copy()
    r = b - matrix_vector_product(x, xb)
    p = r.copy()
    k = 0
    n = len(x0)
    r_old_scalar = np.dot(r, r)
    r_new_scalar = r_old_scalar

    while(k < n and math.sqrt(r_new_scalar) >= tol):
        A_times_p = matrix_vector_product(p, xb)
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


class en4dvar(object):
    """Ensemble four-dimensional variational data assimilation.

    This class contains the necessary functionality for implementing
    an ensemble strong-constraint four-dimensional variational
    (4D-Var) data assimilation system using the incremental method.
    The linear system from the incremental method is solved using the
    conjugate gradient method in a matrix-free way, as only the
    matrix-vector product is required.

    The hybrid 4D-Var class in hy4dvar.py is derived from this class,
    as all of the functionality can be extended to work for a hybrid
    background error covariance matrix.

    Attributes:
        model: A class specifying the underlying model dynamics that
            includes member functions forecast, forecast_tlm, and
            forecast_adj.
    
        Bdata: A list containing the data for the ensemble background
            error covariance. Specifically, the vectors b[j] in Bdata
            are such that the ensemble background error covariance is
            the sum(b[j] * b[j].T).
    
        sigo_squared: Observation error variance vector.
    
        window: The number of time levels in a data assimilation
            cycle.
    
        obsloc: List of arrays to indicate the states which are
            observed. If the state vector x == [x0, x1, x2, x3] and
            obsloc == np.array([1, 2]), then y = h(x) returns y ==
            [x1, x2].

        Cb: Symmetric band matrix representing the localization matrix
            for the ensemble background error covariance matrix.
    
        nobs: The number of observations, computed as the length of
            obsloc.

        sigo: Observation error standard deviation.

    Functionality:
        Bprod: Background error covariance-vector product.
    
        Cprod: Localization matrix-vector product.

        sqrtCprod: Cholesky factor of the localization matrix-vector
            product.

        sqrtCprod_adj: The transpose of the Cholesky factor of the
            localization matrix-vector product.
    
        sqrtBprod: B^(1/2) * x.
    
        sqrtBprod_adj: B^(T/2) * x.
    
        Rprod: Observation error covariance-vector product.
    
        Rinvprod: Inverse observation error covariance-vector product.

        sqrtRprod: The product of a square root matrix of R with a
            vector.

        sqrtRprod_adj: The product of the transpose of a square root
            matrix of R with a vector.

        sqrtRinvprod: The product of a square root of the inverse of
            the observation error covariance with a vector.

        sqrtRinvprod_adj: The product of the transpose a square root
            of the inverse observation error covariance with a vector.
    
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
    
        blkh_tlm: Tangent linear model of four-dimensional observation
            operator.
    
        blkh_adj: Adjoint of four-dimensional observation operator.
    
        blkRprod: Product of four-dimensional observation error
            covariance with a four-dimensional vector.
    
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
    
        Preconditioned_by_B_HessianProduct: Product of a state space
            vector with the Hessian of the cost functional
            preconditioned by sqrtB.

        Preconditioned_by_R_HessianProduct: Product of an observation
            space vector with the Hessian of the cost functional
            preconditioned by sqrtR.
    
        incremental: Incremental algorithm for 4D-Var.
    """
    def __init__(self, model, Bdata, sigo_squared, window, obsloc, Cb):
        """Initializes the class object to the specified inputs.

        Descriptions of each data member is provided in the comments
        above.

        """
        self.model = model
        self.Bdata = Bdata
        self.sigo_squared = sigo_squared
        self.window = window
        self.obsloc = obsloc
        self.Cb = Cb

        # Computation of data members derived from arguments
        self.sqrtCb = la.cholesky_banded(Cb, lower=True)
        self.nobs = len(obsloc[0])
        self.sigo = np.sqrt(sigo_squared)

    def Bprod(self, x):
        """Product of B with a vector x.

        The background error covariance is the sum of the outer
        product of the vectors stored in self.Bdata with localization
        applied. Utilizing the special structure of B, the computation
        is (C \circ Be) * x == sum((C * (x \circ b[j])) \circ b[j])
        where b[j] come from self.Bdata.

        Argument:
            x: Vector to multiply.

        Returns:
            Matrix-vector product.
        """
        return np.sum([self.Cprod(x * b) * b for b in self.Bdata], axis=0)
        
    def Cprod(self, x):
        """Localization matrix-vector product.

        The localization matrix is a symmetric band matrix stored in lower
        form.

        Argument:
            x: The vector to multiply.

        Returns:
            y: Product of localization matrix with x.
        """
        (p_plus_one, n) = self.Cb.shape
        p = p_plus_one - 1   # Bandwidth
        y = np.empty(n)

        for i in xrange(n):
            temp = 0.0

            for j in xrange(max(0, i - p), i):
                temp += self.Cb[i - j, j] * x[j]

            for j in xrange(i, min(i + p_plus_one, n)):
                temp += self.Cb[j - i, i] * x[j]

            y[i] = temp

        return y

    def sqrtCprod(self, x):
        """Cholesky factor of the localization matrix-vector product.
        
        Argument:
            x: The vector to multiply.

        Returns:
            Matrix-vector product C^(1/2) * x.
        """
        (p_plus_one, n) = self.sqrtCb.shape
        p = p_plus_one - 1   # Bandwidth of the matrix
        y = np.empty(n)

        for i in xrange(p):
            temp = 0.0

            for j in xrange(i + 1):
                temp += self.sqrtCb[i - j, j] * x[j]

            y[i] = temp

        for i in xrange(p, n):
            temp = 0.0

            for j in xrange(i - p, i + 1):
                temp += self.sqrtCb[i - j, j] * x[j]

            y[i] = temp

        return y

    def sqrtCprod_adj(self, x):
        """The product C^(T/2) * x.

        The matrix C^(T/2) is the transpose of the Cholesky factor of
        the banded localization matrix C.

        Argument:
            x: The vector to multiply.

        Returns:
            Matrix-vector product.
        """
        (p_plus_one, n) = self.sqrtCb.shape
        p = p_plus_one - 1   # Bandwidth of the matrix
        y = np.empty(n)

        for i in xrange(n - p):
            temp = 0.0

            for j in xrange(i, i + p_plus_one):
                temp += self.sqrtCb[j - i, i] * x[j]

            y[i] = temp

        for i in xrange(n - p, n):
            temp = 0.0

            for j in xrange(i, n):
                temp += self.sqrtCb[j - i, i] * x[j]

            y[i] = temp

        return y

    def sqrtBprod(self, x):
        """Computes the product B^(1/2) * x, using localization.

        The matrix is a rectangular matrix of size n by (n * Ne), and is
        (C * Be)^(1 / 2) = [diag(x_1) C^(1 / 2), ..., diag(x_Ne) C^(1 / 2)].

        Argument:
            x: Vector to multiply, length of n * Ne.

        Returns:
            y: Product with x, length of n.
        """
        Ne_times_n = len(x)
        Ne = len(self.Bdata)
        
        n = Ne_times_n / len(self.Bdata)
        y = np.zeros(n)

        for j in xrange(Ne):
            y += self.sqrtCprod(x[j * n : (j + 1) * n]) * self.Bdata[j]

        return y

    def sqrtBprod_adj(self, x):
        """Computes the product B^(T/2) * x.

        The matrix is a rectangular matrix of size (n * Ne) by n, and is
        (C * Be)^(1 / 2) = [C^(T / 2) diag(x_1); ...; C^(T / 2) diag(x_Ne)].

        Argument:
            x: Vector to multiply of length n.

        Returns:
            y: Product with x, length of n * Ne.
        """
        n = len(x)
        Ne = len(self.Bdata)

        Ne_times_n = n * Ne
        y = np.zeros(Ne_times_n)

        for j in xrange(Ne):
            y[j * n : (j + 1) * n] = self.sqrtCprod_adj(self.Bdata[j] * x)

        return y

    def Rprod(self, x):
        """Product of R with a vector x.

        The observation error covariance matrix R is assumed to be
        diagonal.

        Argument:
            x: The vector to multiply of length nobs.

        Returns:
            Product of R with x.
        """
        return x * self.sigo_squared

    def Rinvprod(self, x):
        """Product of R^(-1) with a vector x.

        The observation error covariance matrix R is assumed to be
        diagonal.

        Argument:
            x: The vector to multiply of length nobs.

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

    def forecast(self, x, ndt=1):
        """Nonlinear forecast model.

        Essentially a wrapper to call the forecast function stored in
        the data member self.model.

        Arguments:
            x: Vector to forecast.
            ndt: Number of time-steps to forecast, default is 1

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
            i: Index indicating which observation operator to use.

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
            i: Index indicating which observation operator to use.

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
            i: Index indicating which observation operator to use.

        Returns:
            Adjoint observation operator in the direction of yb.
        """
        xb = np.zeros(x.shape)
        xb[self.obsloc[i]] = yb
        
        return xb
    
    def blkh(self, x):
        """Four-dimensional observation operator.

        Since the three-dimensional observation operator is linear, this
        four-dimensional operator is linear for each block of x, where
        H = [H_0; H_1 * M_1; ...; H_N * M_{0 to N}]

        Argument:
            x: Three-dimensional state vector.

        Returns:
            y: Four-dimensional observation space equivalent of x.
        """
        nobs = self.nobs
        window = self.window
        n = self.model.n
        
        y = np.empty(window * nobs)
        y[0 : nobs] = self.h(x)

        for j in xrange(1, window):
            x = self.forecast(x)
            y[j * nobs : (j + 1) * nobs] = self.h(x)

        return y
                
    def blkh_tlm(self, xb, xd):
        """Four-dimensional tangent linear observation model.

        The ith block of this matrix is H_i * M_{0 to i}.
        
        Arguments:
            xb: Background state to set the tangent linear model.
            xd: Direction vector of length n.

        Returns:
            yd: Four-dimensional tangent linear observation model in
                the direction of xd of length window * nobs.
        """
        nobs = self.nobs
        window = self.window
        n = self.model.n
        
        yd = np.empty(window * nobs)
        yd[0 : nobs] = self.h_tlm(xb, xd, 0)
        z = xd.copy()

        for j in xrange(1, window):
            z = self.forecast_tlm(xb, z)
            xb = self.forecast(xb)
            yd[j * nobs : (j + 1) * nobs] = self.h_tlm(xb, z, j)

        return yd

    def blkh_adj(self, xb, yb):
        """Four-dimensional adjoint observation model.

        Arguments:
            xb: Background state to set the adjoint model.
            yb: Direction vector of length window * nobs.

        Returns:
            output: Four-dimensional adjoint observation model in the
                direction of yb of length n.
        """
        nobs = self.nobs
        window = self.window
        n = self.model.n
        
        xb_list = [[]] * window
        xb_list[0] = xb

        output = self.h_adj(xb, yb[0 : nobs], 0)

        # Forward sweep
        for i in xrange(1, window):
            xb_list[i] = self.forecast(xb_list[i - 1])

        for i in xrange(1, window):
            z = self.h_adj(xb_list[i], yb[i * nobs : (i + 1) * nobs], i)

            # Backward sweep 
            for j in xrange(i - 1, -1, -1):
                z = self.forecast_adj(xb_list[j], z)
                
            output += z

        return output

    def blkRprod(self, y):
        """Product with four-dimensional observation error covariance.

        Argument:
            y: Vector of length window * nobs to multiply.

        Returns:
            output: Result of matrix multiplication.
        """
        nobs = self.nobs
        output = np.empty(len(y))

        for i in xrange(self.window):
            current = np.arange(i * nobs, (i + 1) * nobs)

            output[current] = self.Rprod(y[current])

        return output

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
    
    def Preconditioned_by_B_HessianProduct(self, x, xb):
        """The Hessian of the preconditioned cost functional product.

        The Hessian of the cost functional preconditioned by sqrtB
        I + sqrtB.T * blkH.T * Rinv * blkH * sqrtB
        is the coefficient matrix of the linear system for the
        preconditioned analysis.

        Arguments:
            x: Three-dimensional vector of size n to multiply.
            xb: Three-dimensional background state.

        Returns:
            y: Result of Hessian-vector product.
        """
        y = self.sqrtBprod(x)
        y = self.blkh_tlm(xb, y)
        y = self.blkRinvprod(y)
        y = self.blkh_adj(xb, y)
        y = self.sqrtBprod_adj(y)
        y += x

        return y

    def Preconditioned_by_R_HessianProduct(self, x, xb):
        """The Hessian of the preconditioned cost functional product.

        The Hessian of the preconditioned cost functional
        sqrtRinv * blkH * B * blkH.T * sqrtRinv.T + I
        is the coefficient matrix of the linear system for the
        preconditioned analysis.

        Arguments:
            x: Vector to multiply of size window * nobs.
            xb: Three-dimensional guess state.

        Returns:
            y: Result of matrix-vector product.
        """
        y = self.blksqrtRinvprod_adj(x)
        y = self.blkh_adj(xb, y)
        y = self.Bprod(y)
        y = self.blkh_tlm(xb, y)
        y = self.blksqrtRinvprod(y)
        y += x

        return y

    def incremental(self, xb, y_list, tol, option):
        """The incremental algorithm for solving 4D-Var.

        Two options for minimizing the cost functional are available,
        where the linear system can be solved in state space (option
        == "StateSpace") using sqrtB preconditioning or in observation
        space (option == "ObsSpace") which uses sqrtR preconditioning.

        Inputs:
            xb: Background state.
            y_list: List of observations.
            tol: Tolerance for conjugate gradient method.
            option: Solve in state space (preconditioned) or in
                observation space, using preconditioning.

        Returns:
            xa: Analysis state at the beginning of the time window.
            k: Number of conjugate gradient iterations to convergence.
        """        
        window = self.window
        n = self.model.n
        nobs = self.nobs
        total_nobs = window * nobs
        d = np.empty(total_nobs)  # Observed minus background

        # Sets four-dimensional vector d = y - h(xb)
        xb_forecast = xb
        d[0 : nobs] = y_list[0] - self.h(xb, 0)
        
        for j in xrange(1, window): 
            temp = np.arange(j * nobs, (j + 1) * nobs)

            xb_forecast = self.forecast(xb_forecast)
            d[temp] = y_list[j] - self.h(xb_forecast, j)

        if option == "StateSpace":
            # Conjugate gradient method for solving the state space
            # linear system
            # [Binv + blkH.T * blkRinv * blkH] * deltax = blkH.T * blkRinv * d
            # by preconditioning using sqrtB to get
            # [I + sqrtB.T * blkH.T * Rinv * blkH * sqrtB] * chi =
            # sqrtB.T * blkH.T * Rinv * d
            b = self.blkRinvprod(d)
            b = self.blkh_adj(xb, b)
            b = self.sqrtBprod_adj(b)

            chi0 = np.zeros(b.shape) # Guess for chi
            (chi, k) = ConjugateGradient(chi0, \
                       self.Preconditioned_by_B_HessianProduct, b, tol, xb)

            deltax = self.sqrtBprod(chi)
        elif option == "ObsSpace":
            # Conjugate gradient method for solving the observation space
            # linear system
            # [blkH * B * blkH.T + R] * z = d
            # by preconditioning using sqrtR to get
            # [sqrtRinv * blkH * B * blkH.T * sqrtRinv.T + I] * chi =
            # sqrtRinv * d, where z = sqrtRinv.T * chi
            b = self.blksqrtRinvprod(d)
        
            chi0 = np.zeros(b.shape)
            (chi, k) = ConjugateGradient(chi0, \
                       self.Preconditioned_by_R_HessianProduct, b, tol, xb)

            z = self.blksqrtRinvprod_adj(chi)
            deltax = self.blkh_adj(xb, z)
            deltax = self.Bprod(deltax)
        else:
            raise ValueError("Choose a valid option.")

        xa = xb + deltax

        return (xa, k)
