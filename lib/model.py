import torch
from torch import nn

from .membership_functions import *
from .t_norms import *


class FirstTSK(nn.Module):
    """
    First-order TSK fuzzy system
    """

    def __init__(self, in_dim, out_dim, num_fuzzy_set, mf='Gaussian', frb_type='CoCo-FRB', tnorm='prod'):
        """

        :param in_dim: input dimension
        :param out_dim: output dimension
        :param num_fuzzy_set: No. of fuzzy sets defined on each feature
        :param mf: membership function,
            {'Gaussian' (default), 'Gaussian_DMF_sig (for HDFIS-prod)', 'Gaussian_HTSK (for HTSK)'}
        :param frb_type: fuzzy rule base, {'CoCo-FRB' (default), 'FuCo-FRB'}
        :param tnorm: for computing firing strength, {'prod' (default), 'min', 'softmin', 'adasoftmin', 'adasoftmin2'}
        """
        super(FirstTSK, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_fuzzy_set = num_fuzzy_set
        self.mf = mf
        self.frb_type = frb_type
        self.tnorm = tnorm

        # CoCo-FRB or FuCo-FRB
        self.FRB = self._init_frb(in_dim, num_fuzzy_set, frb_type=frb_type)
        self.num_rule = self.FRB.size(0)

        # antecedent initialization
        partition = torch.arange(num_fuzzy_set, dtype=torch.float64) / (num_fuzzy_set - 1)
        self.center = nn.Parameter(partition.repeat([in_dim, 1]).T)  # [num_fuzzy_set, in_dim]
        self.spread = nn.Parameter(
            torch.ones([num_fuzzy_set, in_dim], dtype=torch.float64))  # [num_fuzzy_set, in_dim]
        self.ini_center = self.center.data.clone()
        self.ini_spread = self.spread.data.clone()

        # consequent initialization
        self.con_param = nn.Parameter(torch.zeros([self.out_dim, self.num_rule, self.in_dim + 1],
                                                  dtype=torch.float64))  # [out_dim,num_rule,in_dim+1]
        self.ini_con_param = self.con_param.data.clone()

    def _init_frb(self, in_dim, num_fuzzy_set, frb_type):
        """
        generate the index of FRB
        :param in_dim: input dimension
        :param num_fuzzy_set: No. of fuzzy sets defined on each feature
        :param frb_type: fuzzy rule base, {'CoCo-FRB' (default), 'FuCo-FRB'}
        :return: the index of the fuzzy set for computing the firing strength
        """
        if frb_type == 'CoCo-FRB':
            fs_ind = torch.tensor(range(num_fuzzy_set)).unsqueeze(1).repeat_interleave(in_dim, dim=1)
            return fs_ind.long()
        elif frb_type == 'FuCo-FRB':
            fs_ind = torch.zeros([num_fuzzy_set ** in_dim, in_dim])
            for i, ii in enumerate(reversed(range(in_dim))):
                # i: positive order subscript; ii: negative order subscript
                fs_ind[:, ii] = torch.tensor(range(num_fuzzy_set)).repeat_interleave(num_fuzzy_set ** i).repeat(
                    num_fuzzy_set ** ii)
            return fs_ind.long()
        else:
            raise ValueError(
                "Invalid value for frb: '{}', expected 'CoCo-FRB', 'FuCo-FRB'".format(self.frb))

    def est_con_param(self, model_input, target_output, rcond=1e-15):
        """
        estimate the consequent parameters using least square estimation (LSE)
        :param model_input: [num_sam, in_dim]
        :param target_output: [num_sam, out_dim]
        :param rcond: A coefficient in pseudo-inverse
        :return:
        """
        model_input = model_input.double()
        target_output = target_output.double()

        # calculate the normalized firing strengths, [num_sam, num_rule]
        _, fir_str_bar = self.forward(model_input)

        # combine the firing strengths and input samples
        model_input_plus = torch.cat([torch.ones(model_input.size(0), 1), model_input], 1)  # [num_sam,in_dim+1]
        fir_str_bar_input = fir_str_bar.repeat_interleave(model_input_plus.shape[1], dim=1) * model_input_plus.repeat(
            [1, fir_str_bar.size(1)])  # [num_sam,num_rule*(in_dim+1)]
        con_param_temp = fir_str_bar_input.pinverse(rcond=rcond) @ target_output  # [num_rule*(in_dim+1),out_dim]
        self.con_param.data = con_param_temp.permute(1, 0).reshape(self.con_param.shape)  # [out_dim,num_rule,in_dim+1]

    def trained_param(self, tra_param='all'):
        """
        which parameters are going to be trained
        :param tra_param: {'IF', 'THEN', 'IF_THEN', 'all'(default), 'None'}
            IF: antecedent parameters;
            THEN: consequent parameters;
            IF_THEN: antecedent and consequent parameters;
            all: all the parameters
            None: no parameters need gradients
        :return:
        """
        for each in self.parameters():
            each.requires_grad = False

        # which parameters need gradients
        if tra_param == 'None':
            return
        elif tra_param == 'IF':
            self.center = nn.Parameter(self.center)
            self.spread = nn.Parameter(self.spread)
        elif tra_param == 'THEN':
            self.con_param = nn.Parameter(self.con_param)
        elif tra_param == 'IF_THEN':
            self.center = nn.Parameter(self.center)
            self.spread = nn.Parameter(self.spread)
            self.con_param = nn.Parameter(self.con_param)
        elif tra_param == 'all':
            for each in self.parameters():
                each.requires_grad = True
        else:
            raise ValueError("Invalid value for tra_param: '{}'".format(tra_param))

    def forward(self, model_input):
        """
        系统正向传播计算
        :param model_input: [num_sam, in_dim]
        :return: model outputs, [num_sam, out_dim]
        """
        model_input = model_input.double()

        # calculate membership values
        if self.mf == 'Gaussian':
            membership_value = gauss(model_input.unsqueeze(1), self.center, self.spread)
        elif self.mf == 'Gaussian_DMF_sig':
            membership_value = gauss_dmf_sig(model_input.unsqueeze(1), self.center, self.spread, self.in_dim)
        elif self.mf == 'Gaussian_HTSK':
            membership_value = gauss_htsk(model_input.unsqueeze(1), self.center, self.spread, self.in_dim)
        else:
            raise ValueError("Invalid value for mf: '{}'".format(self.mf))

        # calculate firing strengths, [num_sam,num_rule]
        in_dim, fs_ind = self.in_dim, self.FRB
        if self.tnorm == 'prod':  # ProdTSK
            fir_str = membership_value[:, fs_ind, range(in_dim)].prod(dim=2)
        elif self.tnorm == 'min':  # HDFIS-min
            fir_str = membership_value[:, fs_ind, range(in_dim)].min(dim=2).values
        elif self.tnorm == 'softmin':
            fir_str = softmin(membership_value[:, fs_ind, range(in_dim)], q=-12, dim=2)
        elif self.tnorm == 'adasoftmin':  # AdaTSK
            fir_str = adasoftmin(membership_value[:, fs_ind, range(in_dim)], dim=2)
        elif self.tnorm == 'adasoftmin2':  # ALETSK
            fir_str = adasoftmin2(membership_value[:, fs_ind, range(in_dim)], dim=2)
        else:
            raise ValueError("Invalid value for tnorm: '{}'".format(self.tnorm))

        # check numeric underflow
        if fir_str.eq(0).any():
            print('Numeric underflow happens on the firing strength.')

        # calculate rule outputs, [num_sam,num_rule,out_dim]
        rule_output = (self.con_param[:, :, 1:] @ model_input.T).T + self.con_param[:, :, 0].T

        # de-fuzzy for computing the model outputs
        fir_str_bar = fir_str / torch.sum(fir_str, 1).unsqueeze(1)  # [num_sam,num_rule]
        model_output = torch.einsum('NRC,NR->NC', rule_output, fir_str_bar)  # [num_sam,out_dim]

        return model_output, fir_str_bar
