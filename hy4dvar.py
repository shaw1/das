"""Hybrid ensemble strong-constraint 4D-Var.

Date Created:       2 February 2016
"""
import numpy as np
import math

from en4dvar import en4dvar

class hy4dvar(en4dvar):
    """Hybrid strong-constraint 4D-Var.

    This class is an extension of the ensemble strong-constraint
    four-dimensional variational data assimilation from en4dvar.py.
    To implement a hybrid 4D-Var requires defining functions for
    multiplication with a hybrid background error covariance and its
    square root and adjoint. The functions defined in this class
    override those of en4dvar and no additional functions are
    included. Two new attributes which define the hybrid background
    error covariance are included.

    Attributes:
        alpha: Hybrid model error covariance scalar weight. So
            B = (1 - alpha) * Be + alpha * Bc, where Be is the
            ensemble background error covariance and Bc is the static
            diagonal background error covariance prescribed.

        sigb_squared_c: Variance for static background error
            covariance Bc, which is assumed to be a diagonal matrix.

    Functionality:
        Bprod: Background error covariance-vector product, where the
            background error covariance is assumed to be an ensemble
            covariance matrix with localization plus a static diagonal
            covariance together weighted by alpha.

        sqrtBprod: B^(1/2) * x.

        sqrtBprod_adj: B^(T/2) * x.
    """
    def __init__(self, model, Bdata, sigo_squared, window, obsloc, \
                 Cb, alpha, sigb_squared_c):
        """Initializes the class object to the specified inputs.

        Descriptions of each data member is provided in the comments
        above.
        """
        super(hy4dvar, self).__init__(model, Bdata, sigo_squared, \
                                      window, obsloc, Cb)

        self.alpha = alpha
        self.sigb_squared_c = sigb_squared_c

    def Bprod(self, x):
        """Product of B with a vector x.
        
        Since the hybrid covariance is specified as
        B = (1 - alpha) * Be + alpha * Bc, the product of B with x is
        B * x = (1 - alpha) * (Be * x) + alpha * (Bc * x).
        Since Bc is diagonal with diagonal sigb_squared_c, Bc * x can
        be computed as the Hadamard product of x with sigb_squared_c.
        The product Be * x will be computed by calling the super
        class's Bprod function.

        Argument:
            x: Vector to multiply.

        Returns:
            Matrix-vector product.
        """
        return self.alpha * self.sigb_squared_c * x + \
            (1.0 - self.alpha) * super(hy4dvar, self).Bprod(x)

    def sqrtBprod(self, x):
        """Product of B^(1 / 2) with x, using covariance localization.

        The matrix is a rectangular matrix of size n by n * (Ne + 1),
        and is B = [sqrt(1 - alpha) * (C * Be)^(1 / 2) sqrt(alpha) *
        Bc^(1 / 2)].
        
        Argument:
            x: The vector to multiply of length n.

        Returns:
            y: Product with x, length of n * (Ne + 1).
        """
        Ne_plus_one_times_n = len(x)
        Ne_times_n = Ne_plus_one_times_n - self.model.n
        
        y = math.sqrt(1.0 - self.alpha) * \
            super(hy4dvar, self).sqrtBprod(x[0 : Ne_times_n])

        y += np.sqrt(self.alpha * self.sigb_squared_c) * x[Ne_times_n :]

        return y

    def sqrtBprod_adj(self, x):
        """Product of B^(T / 2) with x, using covariance localization.

        The matrix is a rectangular matrix of size n by n * (Ne + 1),
        and is B = [sqrt(1 - alpha) * (C * Be)^(1 / 2) sqrt(alpha) *
        Bc^(1 / 2)].
        
        Argument:
            x: The vector to multiply of length n * (Ne + 1).

        Returns:
            y: Product with x, length of n.
        """
        Ne_plus_one_times_n = (len(self.Bdata) + 1) * self.model.n
        Ne_times_n = Ne_plus_one_times_n - self.model.n

        y = np.empty(Ne_plus_one_times_n)

        y[0 : Ne_times_n] = math.sqrt(1.0 - self.alpha) * \
                            super(hy4dvar, self).sqrtBprod_adj(x)

        y[Ne_times_n :] = np.sqrt(self.alpha * self.sigb_squared_c) * x

        return y
