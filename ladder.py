import logging
import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from blocks.bricks.cost import SquaredError
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.roles import PARAMETER, WEIGHT, BIAS
from utils import shared_param, AttributeDict, apply_act
from blocks.bricks import Linear
from blocks.initialization import IsotropicGaussian
logger = logging.getLogger('main.model')
floatX = theano.config.floatX


class LadderAE():
    def __init__(self):
        self.input_dim = 784
        self.denoising_cost_x = (500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.noise_std = (0.3,) * 7
        self.default_lr = 0.002
        self.shareds = OrderedDict()
        self.rstream = RandomStreams(seed=1)
        self.rng = np.random.RandomState(seed=1)
        self.layers = [(0, (('fc', 784), 'relu')),
                       (1, (('fc', 1000), 'relu')),
                       (2, (('fc', 500), 'relu')),
                       (3, (('fc', 250), 'relu')),
                       (4, (('fc', 250), 'relu')),
                       (5, (('fc', 250), 'relu')),
                       (6, (('fc', 10), 'softmax'))]

    def shared(self, init, name, cast_float32=True, role=PARAMETER, **kwargs):
        p = self.shareds.get(name)
        if p is None:
            p = shared_param(init, name, cast_float32, role, **kwargs)
            self.shareds[name] = p
        return p

    def new_activation_dict(self):
        return AttributeDict({'z': {}, 'h': {}, 's': {}, 'm': {}})

    def encoder(self, input_, noise_std):
        h = input_
        h = h + (self.rstream.normal(size=h.shape).astype(floatX) *
                 noise_std[0])

        d = self.new_activation_dict()
        d.z[0] = h
        d.h[0] = h
        prev_dim = self.input_dim
        for i, (spec, act_f) in self.layers[1:]:
            layer_type, dim = spec
            noise = noise_std[i] if i < len(noise_std) else 0.
            z, m, s, h = self.f(h, prev_dim, layer_type, dim,
                                i, act_f, noise)
            self.layer_dims[i] = dim
            d.z[i] = z
            d.s[i] = s
            d.m[i] = m
            d.h[i] = h
            prev_dim = dim

        return d

    def decoder(self, clean, corr):
        est = self.new_activation_dict()
        costs = AttributeDict()
        costs.denois = AttributeDict()
        for i, ((_, spec), act_f) in self.layers[::-1]:
            z_corr = corr.z[i]
            z_clean = clean.z[i]
            z_clean_s = clean.s.get(i)
            z_clean_m = clean.m.get(i)

            # It's the last layer
            if i == len(self.layers) - 1:
                fspec = (None, None)
                ver = corr.h[i]
                ver_dim = self.layer_dims[i]
                top_g = True
            else:
                fspec = self.layers[i + 1][1][0]
                ver = est.z.get(i + 1)
                ver_dim = self.layer_dims.get(i + 1)
                top_g = False

            z_est = self.g(z_lat=z_corr,
                           z_ver=ver,
                           in_dims=ver_dim,
                           out_dims=self.layer_dims[i],
                           num=i,
                           fspec=fspec,
                           top_g=top_g)

            # The first layer
            if z_clean_s:
                z_est_norm = (z_est - z_clean_m) / z_clean_s
            else:
                z_est_norm = z_est

            se = SquaredError('denois' + str(i))
            costs.denois[i] = se.apply(z_est_norm.flatten(2),
                                       z_clean.flatten(2)) \
                / np.prod(self.layer_dims[i], dtype=floatX)
            costs.denois[i].name = 'denois' + str(i)

            # Store references for later use
            est.z[i] = z_est
            est.h[i] = apply_act(z_est, act_f)
            est.s[i] = None
            est.m[i] = None
        return est, costs

    def apply(self, input, target):
        self.layer_counter = 0
        self.layer_dims = {0: self.input_dim}
        self.lr = self.shared(self.default_lr, 'learning_rate', role=None)
        top = len(self.layers) - 1

        clean = self.encoder(input, noise_std=[0])
        corr = self.encoder(input, noise_std=self.noise_std)

        est, costs = self.decoder(clean, corr)

        # Costs
        y = target.flatten()

        costs.class_clean = CategoricalCrossEntropy().apply(
            y, clean.h[top])
        costs.class_clean.name = 'CE_clean'

        costs.class_corr = CategoricalCrossEntropy().apply(
            y, corr.h[top])
        costs.class_corr.name = 'CE_corr'

        costs.total = costs.class_corr * 1.0
        for i in range(len(self.layers)):
            costs.total += costs.denois[i] * self.denoising_cost_x[i]
        costs.total.name = 'Total_cost'

        self.costs = costs

        # Classification error
        mr = MisclassificationRate()
        self.error = mr.apply(y, clean.h[top]) * np.float32(100.)
        self.error.name = 'Error_rate'

    def rand_init(self, in_dim, out_dim):
        return self.rng.randn(in_dim, out_dim) / np.sqrt(in_dim)

    def apply_layer(self, layer_type, input_, in_dim, out_dim, layer_name):
        # Since we pass this path twice (clean and corr encoder),we
        # want to make sure that parameters of both layers are shared.
        layer = self.shareds.get(layer_name)
        if layer_type == "fc":
            W = self.shared(self.rand_init(in_dim, out_dim), layer_name + 'W',
                            role=WEIGHT)
            return T.dot(input_, W)

        if layer is not None:
            return layer
        else:
            if layer_type == 'fc':
                linear = Linear(use_bias=False,
                                name=layer_name,
                                input_dim=in_dim,
                                output_dim=out_dim,
                                seed=1)
                layer = linear.apply(input_)
                linear.weights_init = IsotropicGaussian(1.0 / np.sqrt(in_dim))
                linear.initialize()
                self.shareds[layer_name] = layer

            return layer

    def f(self, h, in_dim, layer_type, dim, num, act_f, noise_std):
        layer_name = 'f_' + str(num) + '_'

        z = self.apply_layer(layer_type, h, in_dim, dim, layer_name)

        m = s = None
        m = z.mean(0, keepdims=True)
        s = z.var(0, keepdims=True)

        z = (z - m) / T.sqrt(s + np.float32(1e-10))

        if noise_std > 0:
            z += self.rstream.normal(size=z.shape).astype(floatX) * noise_std

        # z for lateral connection
        z_lat = z
        b_init, c_init = 0.0, 1.0
        b_c_size = dim

        # Add bias
        if act_f != 'linear':
            z += self.shared(b_init * np.ones(b_c_size), layer_name + 'b',
                             role=BIAS)

        # Add Gamma parameter if necessary. (Not needed for all act_f)
        if (act_f in ['sigmoid', 'tanh', 'softmax']):
            c = self.shared(c_init * np.ones(b_c_size), layer_name + 'c',
                            role=WEIGHT)
            z *= c

        h = apply_act(z, act_f)

        return z_lat, m, s, h

    def g(self, z_lat, z_ver, in_dims, out_dims, num, fspec, top_g):
        f_layer_type, dims = fspec
        layer_name = 'g_' + str(num) + '_'

        in_dim = np.prod(dtype=floatX, a=in_dims)
        out_dim = np.prod(dtype=floatX, a=out_dims)

        if top_g:
            u = z_ver
        else:
            u = self.apply_layer(f_layer_type, z_ver,
                                 in_dim, out_dim, layer_name)

        # Batch-normalize u
        u -= u.mean(0, keepdims=True)
        u /= T.sqrt(u.var(0, keepdims=True) + np.float32(1e-10))

        # Define the g function
        z_lat = z_lat.flatten(2)
        bi = lambda inits, name: self.shared(inits * np.ones(out_dim),
                                             layer_name + name, role=BIAS)
        wi = lambda inits, name: self.shared(inits * np.ones(out_dim),
                                             layer_name + name, role=WEIGHT)

        sigval = bi(0., 'c1') + wi(1., 'c2') * z_lat
        sigval += wi(0., 'c3') * u + wi(0., 'c4') * z_lat * u
        sigval = T.nnet.sigmoid(sigval)
        z_est = bi(0., 'a1') + wi(1., 'a2') * z_lat + wi(1., 'b1') * sigval
        z_est += wi(0., 'a3') * u + wi(0., 'a4') * z_lat * u

        if (type(out_dims) == tuple and
                len(out_dims) > 1.0 and z_est.ndim < 4):
            z_est = z_est.reshape((z_est.shape[0],) + out_dims)

        return z_est
