from itertools import chain
from itertools import compress
from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import copy
import torch.nn.functional as F
from self_distillation.print_color import pc, ps
from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier, MR, minADE, minAHE, minFDE, minFHE
from torch_geometric.data import Batch
try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object

from self_distillation.mask import mask_scenario
from self_distillation.viz import *
import self_distillation.augment as augment
        
class TeacherStudent(pl.LightningModule):
    def __init__(self, backbone_model:pl.LightningModule, args):
        super(TeacherStudent, self).__init__()
        pl.seed_everything(2024, workers=True)
        torch.cuda.empty_cache()
        # torch.set_float32_matmul_precision('medium' | 'high')

        self.student = backbone_model
        self.teacher = copy.deepcopy(self.student)
        self.teacher.load_state_dict(self.student.state_dict())     # same init weights

        # Disable gradient computation for the teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.mask_token = nn.Parameter(torch.randn(args['input_dim']))         # Learnable mask token

        self.set_args(args)

        # Define loss functions
        self.reg_loss = NLLLoss(component_distribution=['laplace'] * self.output_dim + ['von_mises'] * self.output_head, reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * self.output_dim + ['von_mises'] * self.output_head, reduction='none')

        # Metrics
        self.Brier = Brier(max_guesses=6)
        self.minADE = minADE(max_guesses=6)
        self.minAHE = minAHE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.minFHE = minFHE(max_guesses=6)
        self.MR = MR(max_guesses=6)

        self.test_predictions = dict()

    def set_args(self, args):
        self.t_history = args['num_historical_steps']
        self.t_future = args['num_future_steps']
        self.max_epochs = args['max_epochs']
        self.weight_decay = args['weight_decay']
        self.T_max = args['T_max']
        self.lr = args['lr']
        self.output_dim = args['output_dim']
        self.output_head = args['output_head']
        self.submission_dir = args['submission_dir']
        self.submission_file_name = args['submission_file_name']
        self.dataset = args['dataset']
        self.num_modes = args['num_modes']

    def update_ema(self, alpha: float = 0.999):
        """
        Update the weights of the teacher to be an exponential moving average of the weights of the student.
        
        Args:
        alpha (float): Smoothing coefficient for EMA. Alpha --> 1: very smooth, alpha --> 0: no smoothing. 
        """
        with torch.no_grad():
            for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
                # teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data

    def forward(self, traj, road):
        return self.student(traj, road)
    
    
    def calc_loss(self, data):
        plot_traj_from_tensor(data)
        
        #* 2 augmentations
        data1 = data
        data2 = augment.flip(data)

        plot_traj_from_tensor(data2, flip=True)

        exit()

        #* MASKING
        masked_data_1 = mask_scenario(data1)
        masked_data_2 = mask_scenario(data2)

        # Student forward
        student_output_1 = self.get_output(masked_data_1, network_type='student')
        student_output_2 = self.get_output(masked_data_2, network_type='student')

        # Teacher forward
        with torch.no_grad():
            teacher_output_1 = self.get_output(data1, network_type='teacher')
            teacher_output_2 = self.get_output(data2, network_type='teacher')

        # Pick best teacher mode and regard it as psudo GT
        gt1 = torch.cat([data1['agent']['target'][..., :self.output_dim], data1['agent']['target'][..., -1:]], dim=-1)
        psudo_gt1 = self.get_best_mode(teacher_output_1, gt1)
        gt2 = torch.cat([data2['agent']['target'][..., :self.output_dim], data2['agent']['target'][..., -1:]], dim=-1)
        psudo_gt2 = self.get_best_mode(teacher_output_2, gt2)

        #* Reg loss 
        reg_loss_1 = self.calc_reg_loss(traj_propose = student_output_1['traj_propose'],
                                        traj_refine = student_output_1['traj_refine'],
                                        target = psudo_gt1,
                                        reg_mask = student_output_1['reg_mask'])
        
        reg_loss_2 = self.calc_reg_loss(traj_propose = student_output_2['traj_propose'],
                                traj_refine = student_output_2['traj_refine'],
                                target = psudo_gt2,
                                reg_mask = student_output_2['reg_mask'])
        # reg_loss = torch.mean(torch.cat((reg_loss_1.unsqueeze(0), reg_loss_2.unsqueeze(0)), dim = 0))

        #* Cls loss - cross-view 
        cls_loss_1 = self.calc_cls_loss(traj_refine= student_output_1['traj_refine'],
                                        target = psudo_gt2,
                                        pi = student_output_1['pi'],
                                        reg_mask = student_output_1['reg_mask'],
                                        cls_mask = student_output_1['cls_mask'])

        cls_loss_2 = self.calc_cls_loss(traj_refine= student_output_2['traj_refine'],
                                                target = psudo_gt1,
                                                pi = student_output_2['pi'],
                                                reg_mask = student_output_2['reg_mask'],
                                                cls_mask = student_output_2['cls_mask'])

        loss = reg_loss_1 + reg_loss_2 + cls_loss_1  + cls_loss_2
        
        
        return {
            "loss": loss,
            "cls_loss": cls_loss_1.item(),
            "stu_tea_loss": reg_loss_1.item(),
        }
    
    def get_output(self, data, network_type = "student"):

        network = self.student if network_type == "student" else self.teacher 

        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        reg_mask = data['agent']['predict_mask'][:, self.t_history:]
        cls_mask = data['agent']['predict_mask'][:, -1]

        pred = network(data)     # Forward

        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']

        return {
            'traj_propose': traj_propose,
            'traj_refine': traj_refine,
            'pi': pi,
            'reg_mask': reg_mask,
            'cls_mask': cls_mask,
        }
    
    def calc_reg_loss(self, traj_propose, traj_refine, target, reg_mask):
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              target[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        
        # Best traj in terms of l2 norm
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        
        # Reg loss propose 
        reg_loss_propose = self.reg_loss(traj_propose_best, target[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()

        # Reg loss refine
        reg_loss_refine = self.reg_loss(traj_refine_best, target[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()

        return reg_loss_propose + reg_loss_refine 


    def calc_cls_loss(self, traj_refine, target, pi, reg_mask, cls_mask):
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                            target=target[:, -1:, :self.output_dim + self.output_head],
                            prob=pi,
                            mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        return cls_loss

    def get_best_mode(self, output, gt):
        traj_refine = output['traj_refined']
        reg_mask = output['reg_mask']

        l2_norm = (torch.norm(traj_refine[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        
        best_mode = l2_norm.argmin(dim=-1)
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        return traj_refine_best
    
    def training_step(self, batch, batch_idx):

        loss = self.calc_loss(batch)

        self.log('train_loss', loss['loss'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_refine', loss['reg_loss'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', loss['cls_loss'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)

        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.update_ema()
    
    def validation_step(self, data, batch_idx):

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        
        output = self.get_output(data, network_type="student")
        reg_mask = output['reg_mask']
        traj_refine = output['traj_refine']
        pi = output['pi']

        valid_mask_eval = reg_mask[eval_mask]
        traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
        if not self.output_head:
            traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                     traj_eval[..., :2]], dim=-2)
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
            head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
            traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        gt_eval = gt[eval_mask]

        # Update metrics
        self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, valid_mask=valid_mask_eval)
        self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, valid_mask=valid_mask_eval)
        self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval, valid_mask=valid_mask_eval)
        self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,valid_mask=valid_mask_eval)

        self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minAHE', self.minAHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFHE', self.minFHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

    def test_step(self, data, batch_idx):
        if self.dataset == 'argoverse_v2':
                eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        output = self.get_output(data, network_type="student")
        traj_refine = output['traj_refine']
        pi = output['pi']
        
        origin_eval = data['agent']['position'][eval_mask, self.t_history - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.t_history - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2], rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)

        traj_eval = traj_eval.cpu().numpy()
        pi_eval = pi_eval.cpu().numpy()
        if self.dataset == 'argoverse_v2':
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
            if isinstance(data, Batch):
                for i in range(data.num_graphs):
                    self.test_predictions[data['scenario_id'][i]] = (pi_eval[i], {eval_id[i]: traj_eval[i]})
            else:
                self.test_predictions[data['scenario_id']] = (pi_eval[0], {eval_id[0]: traj_eval[0]})
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def on_test_end(self):
        if self.dataset == 'argoverse_v2':
            ChallengeSubmission(self.test_predictions).to_parquet(
                Path(self.submission_dir) / f'{self.submission_file_name}.parquet')
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]


# class SaveEveryNEpochs(Callback):
#     def __init__(self, n_epochs):
#         super().__init__()
#         self.n_epochs = n_epochs

#     def on_validation_epoch_end(self, trainer:pl.LightningModule, pl_module):
#         if SAVE_WEIGHTS and (trainer.current_epoch + 1) % self.n_epochs == 0:
#             trainer.save_checkpoint(f"{WEIGHTS_PATH}/{date}_{RUN_NAME}/epoch={trainer.current_epoch+1:02d}.ckpt")
