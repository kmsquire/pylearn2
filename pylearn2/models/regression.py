"""
.. todo::

    WRITEME
"""
__authors__ = "Kevin Squire"
__copyright__ = "Copyright 2014, Kevin Squire"
__credits__ = ["Kevin Squire"]
__license__ = "3-clause BSD"
__maintainer__ = "Kevin Squire"
__email__ = "kevin.squire@gmail.com"
from sklearn.linear_model import LogisticRegression as SKLR
import numpy as np

class LogisticRegression(Softmax):
    """
    A logistic regression layer of an MLP.
    """

    def __init__(self, layer_name,
                 irange = None,
                 istdev = None,
                 sparse_init = None,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 max_row_norm = None,
                 max_col_norm = None):
        """
        .. todo::

            WRITEME
        """

        if isinstance(W_lr_scale, str):
            W_lr_scale = float(W_lr_scale)

        self.__dict__.update(locals())
        del self.self
        del self.init_bias_target_marginals
        
        self.output_space = VectorSpace(1)

    def get_monitoring_channels(self):
        """
        .. todo::

            WRITEME
        """

        W = self.W

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ])

    def cost(self, Y, Y_hat):

        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
            assert len(owner.inputs) == 1
            Y_hat, = owner.inputs
            owner = Y_hat.owner
            op = owner.op
        assert isinstance(op, T.nnet.Softmax)
        z1 ,= owner.inputs
        assert z1.ndim == 1

        z0 = T.log(T.ones_like(z1)-T.exp(z1))
        log_prob = (Y * z1) + (T.ones_like(Y)-Y)*z2

        rval = log_prob.mean()

        return -rval
