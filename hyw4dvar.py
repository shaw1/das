"""Hybrid weak-constraint 4D-Var.

Date Created:       23 November 2015
"""
import numpy as np
import math

from enw4dvar import enw4dvar

class hyw4dvar(enw4dvar):
    """Hybrid weak-constraint 4D-Var.

    This class is an extension of the ensemble weak-constraint
    four-dimensional variational data assimilation (hybrid w4D-Var).
    To implement a hybrid w4D-Var requires defining functions for
    multiplication with a hybrid model error covariance and its
    square root and adjoint. The functions defined in this class
    override those of enw4dvar and no additional functions are
    included. Two new attributes which define the hybrid model error
    covariance are included.

    Attributes:
        alpha: Hybrid model error covariance scalar weight. So
            Q = (1 - alpha) * Qe + alpha * Qc, where Qe is the
            ensemble model error covariance and Qc is the static
            diagonal model error covariance prescribed.

        sigq_squared_c: Variance for static model error covariance Qc,
            which is assumed to be a diagonal matrix.

    Functionality:
        Qprod: Model error covariance-vector product, where the model
            error covariance is assumed to be an ensemble covariance
            matrix with localization plus a static diagonal covariance
            together weighted by alpha.

        sqrtQprod: Q^(1/2) * x.

        sqrtQprod_adj: Q^(T/2) * x.

        sqrtPprod: Product of the square root of the four-dimensional
            covariance model with a vector.
    
        sqrtPprod_adj: Product of the square root of the
            four-dimensional covariance model transpose with a vector.
    """
    def __init__(self, model, sqrtBdata, sigo_squared, Qdata, q, window, obsloc,
                 Cb, sigq_squared_c, alpha):
        """Initializes the class object to the specified inputs.

        Descriptions of each data member is provided in the comments
        above. The __init__ of the super class enw4dvar is called to
        initialize the components of the parent class.
        """
        super(hyw4dvar, self).__init__(model, sqrtBdata, sigo_squared, Qdata, q, \
                                       window, obsloc, Cb)

        self.alpha = alpha                   # Hybrid scalar weight
        self.sigq_squared_c = sigq_squared_c # Variance for diagonal Qc

    def Qprod(self, x, i):
        """Product of Q[i] with a vector x.

        Since the hybrid covariance is specified as
        Q = (1 - alpha) * Qe + alpha * Qc for each i = 0, 1, ...,
        window - 1, the product of Q with x is
        Q * x = (1 - alpha) * (Qe * x) + alpha * (Qc * x).
        Since Qc is diagonal with diagonal sigq_squared_c, Qc * x can
        be computed as the Hadamard product of x with sigq_squared_c.
        The product Qe * x will be computed by calling the super
        class's Qprod function.

        Arguments:
            x: Vector to multiply.
            i: Time index.

        Returns:
            Matrix-vector product.
        """
        return self.alpha * self.sigq_squared_c * x + \
            (1.0 - self.alpha) * super(hyw4dvar, self).Qprod(x, i)

    def sqrtQprod(self, x, i):
        """Product of Q^(1 / 2) with x, using covariance localization.

        The matrix is a rectangular matrix of size n by n * (Ne + 1), and is
        Q = [sqrt(1 - alpha) * (C * Qe)^(1 / 2)   sqrt(alpha) * Qc^(1 / 2)].
        
        Arguments:
            x: The vector to multiply of length n.
            i: Index indicating which model error covariance to use.

        Returns:
            y: Product with x, length of n * (Ne + 1).
        """
        Ne_plus_one_times_n = len(x)
        Ne_times_n = Ne_plus_one_times_n - self.model.n
        
        y = math.sqrt(1.0 - self.alpha) * \
            super(hyw4dvar, self).sqrtQprod(x[0 : Ne_times_n], i)

        y += np.sqrt(self.alpha * self.sigq_squared_c) * x[Ne_times_n :]

        return y

    def sqrtQprod_adj(self, x, i):
        """Product of Q^(T / 2) with x, using covariance localization.

        The matrix is a rectangular matrix of size n by n * (Ne + 1), and is
        Q = [sqrt(1 - alpha) * (C * Qe)^(1 / 2)   sqrt(alpha) * Qc^(1 / 2)].
        
        Arguments:
            x: The vector to multiply of length n * (Ne + 1).
            i: Index indicating which model error covariance to use.

        Returns:
            y: Product with x, length of n.
        """
        Ne_plus_one_times_n = (len(self.Qdata[0]) + 1) * self.model.n
        Ne_times_n = Ne_plus_one_times_n - self.model.n

        y = np.empty(Ne_plus_one_times_n)

        y[0 : Ne_times_n] = math.sqrt(1.0 - self.alpha) * \
                            super(hyw4dvar, self).sqrtQprod_adj(x, i)

        y[Ne_times_n :] = np.sqrt(self.alpha * self.sigq_squared_c) * x

        return y

    def sqrtPprod(self, x):
        """Product of P^(1 / 2) with a vector.

        Since the square root matrix for Q is a wide rectangle matrix
        of dimension n by n * (Ne + 1), the input vector x is a large
        vector of size (window - 1) * n * (Ne + 1) + n. The output vector
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
        Ne_plus_one_times_n = (Ne + 1) * n
        
        y = np.zeros(window * n)

        y[0 : n] = self.sqrtBprod(x[0 : n])

        for i in xrange(window - 1):
            x_index = np.arange(n + i * Ne_plus_one_times_n, \
                                n + (i + 1) * Ne_plus_one_times_n)
            y_index = np.arange((i + 1) * n, (i + 2) * n)
            
            y[y_index] = self.sqrtQprod(x[x_index], i)

        return y

    def sqrtPprod_adj(self, x):
        """Product of P^(T / 2) with a vector.

        Since the adjoint of the square root matrix for Q is a tall
        rectangle matrix of dimension n * (Ne + 1) by n, the input vector x
        is of the length of a four-dimensional vector of length
        window * n. The output vector will have a very large size of
        length (window - 1) * n * (Ne + 1) + n.

        Argument:
            x: Four-dimensional vector to multiply.

        Returns:
            y: Matrix-vector product.
        """
        n = self.model.n
        window = self.window
        Ne = len(self.Qdata[0])
        Ne_plus_one_times_n = (Ne + 1) * n

        y = np.empty(n + (window - 1) * (Ne + 1) * n)

        y[0 : n] = self.sqrtBprod_adj(x[0 : n])

        for i in xrange(window - 1):
            x_index = np.arange((i + 1) * n, (i + 2) * n)
            y_index = np.arange(n + i * Ne_plus_one_times_n, \
                                n + (i + 1) * Ne_plus_one_times_n)

            y[y_index] = self.sqrtQprod_adj(x[x_index], i)

        return y
