# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from predictors import QCNet
import wandb
from pytorch_lightning.loggers import WandbLogger
from self_distillation.teacher_student import TeacherStudent

from self_distillation.utils import get_date
from self_distillation.print_color import *

import self_distillation.augment as augment
import self_distillation.mask as mask
from self_distillation.viz import *


if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--n', type=str, default=get_date())        # Run name


    QCNet.add_model_specific_args(parser)
    args = parser.parse_args()

    # backbone_model = QCNet(**vars(args))
    
    # # WANDB
    # if args.wandb:
    #     wandb.login()
    #     wandb_logger = WandbLogger(project='QCNet', name=args.n)
    # else:
    #     wandb_logger = None

    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[args.dataset](**vars(args))

    # #! Only during debugging
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    input_data = batch[0]

    # plot_traj_from_tensor(input_data)

    # input_data = augment.flip(input_data)
    masked_input = mask.mask_scenario(input_data)
    plot_scenario(masked_input, name="mask_test")

    # plot_traj_from_tensor(input_data, name="mask")


    # pc(input_data['agent']['predict_mask'][:,:50])
    # pc(input_data['agent']['predict_mask'][:,50:], c = "b")

    # pc("#####", c="g")
    # pc(input_data['agent']['valid_mask'][:,:50])
    # pc(input_data['agent']['valid_mask'][:,50:], c = "b")
    

    exit()
    # #! -----------

    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    teacher_student_model = TeacherStudent(backbone_model, vars(args))
    
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs, logger=wandb_logger)

    trainer.fit(teacher_student_model, datamodule)

    wandb.finish() if args.wandb else None

