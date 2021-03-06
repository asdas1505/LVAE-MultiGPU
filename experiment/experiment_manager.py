import argparse
import os
import boilr.data
import torch
# from boilr import VAEExperimentManager
from experiment.experiments import VAEExperimentManager
from boilr.nn.init import data_dependent_init
from boilr.utils import linear_anneal
from torch import optim
from torch.optim.optimizer import Optimizer
from typing import Optional
from models.lvae import LadderVAE
from .data import DatasetLoader

boilr.set_options(model_print_depth=2)



class LVAEExperiment(VAEExperimentManager):
    """
    Experiment manager.

    Data attributes:
    - 'args': argparse.Namespace containing all config parameters. When
      initializing this object, if 'args' is not given, all config
      parameters are set based on experiment defaults and user input, using
      argparse.
    - 'run_description': string description of the run that includes a timestamp
      and can be used e.g. as folder name for logging.
    - 'model': the model.
    - 'device': torch.device that is being used
    - 'dataloaders': DataLoaders, with attributes 'train' and 'test'
    - 'optimizer': the optimizer
    """

    def _make_datamanager(self, args_eval=None) -> boilr.data.BaseDatasetManager:
        cuda = self.device == 'cuda'
        return DatasetLoader(self.args if args_eval is None else args_eval, cuda)

    def _make_model(self, args_eval=None, dataloader=None) -> torch.nn.Module:
        args = self.args if args_eval is None else args_eval


        # torch.distributed.init_process_group(backend = 'nccl', init_method='env://')

        model = torch.nn.DataParallel(LadderVAE(
            self.dataloaders.color_ch if dataloader is None else dataloader.color_ch,
            z_dims=args.z_dims,
            blocks_per_layer=args.blocks_per_layer,
            downsample=args.downsample,
            merge_type=args.merge_layers,
            batchnorm=args.batch_norm,
            nonlin=args.nonlin,
            stochastic_skip=args.skip_connections,
            n_filters=args.n_filters,
            dropout=args.dropout,
            res_block_type=args.residual_type,
            free_bits=args.free_bits,
            learn_top_prior=args.learn_top_prior,
            img_shape=self.dataloaders.img_size if dataloader is None else dataloader.img_size,
            likelihood_form=args.likelihood,
            gated=args.gated,
            no_initial_downscaling=args.no_initial_downscaling,
            analytical_kl=args.analytical_kl,
        ), device_ids=[0,1,2,5])
        model.to(f'cuda:{model.device_ids[0]}')

        # Weight initialization
        if args.simple_data_dependent_init:

            # Get batch
            t = [
                self.dataloaders.train.dataset[i] if dataloader is None else dataloader.train.dataset[i]
                for i in range(args.batch_size)
            ]
            t = torch.stack(tuple(t[i][0] for i in range(len(t))))

            # Use batch for data dependent init
            data_dependent_init(model, {'x': t.to(self.device)})

        return model

    def _make_optimizer(self) -> Optimizer:
        args = self.args
        optimizer = optim.Adamax(self.model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
        return optimizer

    @classmethod
    def _define_args_defaults(cls) -> dict:
        defaults = super(LVAEExperiment, cls)._define_args_defaults()

        # Override boilr defaults
        defaults.update(

            # General
            batch_size=64,
            test_batch_size=64,
            lr=3e-4,
            train_log_every=10000,
            test_log_every=10000,
            checkpoint_every=100000,
            keep_checkpoint_max=2,
            resume="",

            # VI-specific
            loglikelihood_every=50000,
            loglikelihood_samples=100,
        )

        return defaults

    def _add_args(self, parser: argparse.ArgumentParser) -> None:

        super(LVAEExperiment, self)._add_args(parser)

        def list_options(lst):
            if lst:
                return "'" + "' | '".join(lst) + "'"
            return ""

        legal_merge_layers = ['linear', 'residual']
        legal_nonlin = ['relu', 'leakyrelu', 'elu', 'selu']
        legal_resblock = ['cabdcabd', 'bacdbac', 'bacdbacd']
        legal_datasets = [
            'static_mnist', 'cifar10', 'celeba', 'svhn',
            'multi_dsprites_binary_rgb', 'multi_mnist_binary'
        ]
        legal_likelihoods = [
            'bernoulli', 'gaussian', 'discr_log', 'discr_log_mix'
        ]

        parser.add_argument('-d',
                            '--dataset',
                            type=str,
                            choices=legal_datasets,
                            default='static_mnist',
                            metavar='NAME',
                            dest='dataset_name',
                            help="dataset: " + list_options(legal_datasets))

        parser.add_argument('--likelihood',
                            type=str,
                            choices=legal_likelihoods,
                            metavar='NAME',
                            dest='likelihood',
                            help="likelihood: {}; the default depends on the "
                            "dataset".format(list_options(legal_likelihoods)))

        parser.add_argument('--zdims',
                            nargs='+',
                            type=int,
                            default=[32, 32, 32],
                            metavar='DIM',
                            dest='z_dims',
                            help='list of dimensions (number of channels) for '
                            'each stochastic layer')

        parser.add_argument('--blocks-per-layer',
                            type=int,
                            default=2,
                            metavar='N',
                            help='residual blocks between stochastic layers')

        parser.add_argument('--nfilters',
                            type=int,
                            default=64,
                            metavar='N',
                            dest='n_filters',
                            help='number of channels in all residual blocks')

        parser.add_argument('--no-bn',
                            action='store_true',
                            dest='no_batch_norm',
                            help='do not use batch normalization')

        parser.add_argument('--skip',
                            action='store_true',
                            dest='skip_connections',
                            help='skip connections in generative model')

        parser.add_argument('--gated',
                            action='store_true',
                            dest='gated',
                            help='use gated layers in residual blocks')

        parser.add_argument('--downsample',
                            nargs='+',
                            type=int,
                            default=[1, 1, 1],
                            metavar='N',
                            help='list of integers, each int is the number of '
                            'downsampling steps (by a factor of 2) before each '
                            'stochastic layer')

        parser.add_argument('--learn-top-prior',
                            action='store_true',
                            help="learn the top-layer prior")

        parser.add_argument('--residual-type',
                            type=str,
                            choices=legal_resblock,
                            default='bacdbacd',
                            metavar='TYPE',
                            help="type of residual blocks: " +
                            list_options(legal_resblock))

        parser.add_argument('--merge-layers',
                            type=str,
                            choices=legal_merge_layers,
                            default='residual',
                            metavar='TYPE',
                            help="type of merge layers: " +
                            list_options(legal_merge_layers))

        parser.add_argument('--beta-anneal',
                            type=int,
                            default=0,
                            metavar='B',
                            help='steps for annealing beta from 0 to 1')

        parser.add_argument('--data-dep-init',
                            action='store_true',
                            dest='simple_data_dependent_init',
                            help='use simple data-dependent initialization to '
                            'normalize outputs of affine layers')

        parser.add_argument('--wd',
                            type=float,
                            default=0.0,
                            dest='weight_decay',
                            help='weight decay')

        parser.add_argument('--nonlin',
                            type=str,
                            choices=legal_nonlin,
                            default='elu',
                            metavar='F',
                            help="nonlinear activation: " +
                            list_options(legal_nonlin))

        parser.add_argument('--dropout',
                            type=float,
                            default=0.2,
                            metavar='D',
                            help='dropout probability (in deterministic '
                            'layers)')

        parser.add_argument('--freebits',
                            type=float,
                            default=0.0,
                            metavar='N',
                            dest='free_bits',
                            help='free bits (nats)')

        parser.add_argument('--analytical-kl',
                            action='store_true',
                            dest='analytical_kl',
                            help='use analytical KL')

        parser.add_argument('--no-initial-downscaling',
                            action='store_true',
                            dest='no_initial_downscaling',
                            help='do not downscale as first inference step (and'
                            'upscale as last generation step)')

    @classmethod
    def _check_args(cls, args: argparse.Namespace) -> argparse.Namespace:

        args = super(LVAEExperiment, cls)._check_args(args)

        if len(args.z_dims) != len(args.downsample):
            msg = ("length of list of latent dimensions ({}) does not match "
                   "length of list of downsampling factors ({})").format(
                       len(args.z_dims), len(args.downsample))
            raise RuntimeError(msg)

        assert args.weight_decay >= 0.0
        assert 0.0 <= args.dropout <= 1.0
        if args.dropout < 1e-5:
            args.dropout = None
        assert args.free_bits >= 0.0
        args.batch_norm = not args.no_batch_norm

        likelihood_map = {
            'static_mnist': 'bernoulli',
            'multi_dsprites_binary_rgb': 'bernoulli',
            'multi_mnist_binary': 'bernoulli',
            'cifar10': 'discr_log_mix',
            'celeba': 'discr_log_mix',
            'svhn': 'discr_log_mix',
        }
        if args.likelihood is None:  # default
            args.likelihood = likelihood_map[args.dataset_name]

        return args

    @staticmethod
    def _make_run_description(args: argparse.Namespace) -> str:
        s = ''
        s += args.dataset_name
        s += ',{}ly'.format(len(args.z_dims))
        # s += ',z=' + str(args.z_dims).replace(" ", "")
        # s += ',dwn=' + str(args.downsample).replace(" ", "")
        s += ',{}bpl'.format(args.blocks_per_layer)
        s += ',{}ch'.format(args.n_filters)
        if args.skip_connections:
            s += ',skip'
        if args.gated:
            s += ',gate'
        s += ',block=' + args.residual_type
        if args.beta_anneal != 0:
            s += ',b{}'.format(args.beta_anneal)
        s += ',{}'.format(args.nonlin)
        if args.free_bits > 0:
            s += ',freeb={}'.format(args.free_bits)
        if args.dropout is not None:
            s += ',drop={}'.format(args.dropout)
        if args.learn_top_prior:
            s += ',learnp'
        if args.weight_decay > 0.0:
            s += ',wd={}'.format(args.weight_decay)
        s += ',seed{}'.format(args.seed)
        if len(args.additional_descr) > 0:
            s += ',' + args.additional_descr
        return s


    def forward_pass(self,
                     x: torch.Tensor,
                     y: Optional[torch.Tensor] = None) -> dict:

        # Forward pass

        x = x.to(f'cuda:{self.model.device_ids[0]}', non_blocking=True)
        model_out = self.model(x)

        # L2
        l2 = 0.0
        for p in self.model.parameters():
            l2 = l2 + torch.sum(p**2)
        l2 = l2.sqrt()

        output = {
            'loss': model_out['loss'],
            'elbo': model_out['elbo'],
            'elbo_sep': model_out['elbo_sep'],
            'kl': model_out['kl'],
            'l2': l2,
            'recons': model_out['recons'],
            'out_mean': model_out['out_mean'],
            'out_mode': model_out['out_mode'],
            'out_sample': model_out['out_sample'],
            'likelihood_params': model_out['likelihood_params'],
        }
        if 'kl_avg_layerwise' in model_out:
            output['kl_avg_layerwise'] = model_out['kl_avg_layerwise']

        return output

    @classmethod
    def train_log_str(cls,
                      summaries: dict,
                      step: int,
                      epoch: Optional[int] = None) -> str:
        s = "       [step {}]   loss: {:.5g}   ELBO: {:.5g}   recons: {:.3g}   KL: {:.3g}"
        s = s.format(step, summaries['loss/loss'], summaries['elbo/elbo'],
                     summaries['elbo/recons'], summaries['elbo/kl'])
        return s

    @classmethod
    def test_log_str(cls,
                     summaries: dict,
                     step: int,
                     epoch: Optional[int] = None) -> str:
        s = "       "
        if epoch is not None:
            s += "[step {}, epoch {}]   ".format(step, epoch)
        s += "ELBO {:.5g}   recons: {:.3g}   KL: {:.3g}".format(
            summaries['elbo/elbo'], summaries['elbo/recons'],
            summaries['elbo/kl'])
        ll_key = None
        for k in summaries.keys():
            if k.find('elbo_IW') > -1:
                ll_key = k
                iw_samples = k.split('_')[-1]
                break
        if ll_key is not None:
            s += "   marginal log-likelihood ({}) {:.5g}".format(
                iw_samples, summaries[ll_key])

        return s

    @classmethod
    def get_metrics_dict(cls, results: dict) -> dict:
        metrics_dict = {
            'loss/loss': results['loss'].mean().item(),
            'elbo/elbo': results['elbo'].mean().item(),
            'elbo/recons': results['recons'].mean().item(),
            'elbo/kl': results['kl'].mean().item(),
            'l2/l2': results['l2'].mean().item(),
        }
        if 'kl_avg_layerwise' in results:
            for i in range(len(results['kl_avg_layerwise'])):
                key = 'kl_layers/kl_layer_{}'.format(i)
                metrics_dict[key] = results['kl_avg_layerwise'][i].mean().item()
        return metrics_dict
