import matplotlib as mpl
mpl.use('Agg') # for linux
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import math
from random import randint
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary
from torch.utils.data import DataLoader
from demo.data_preprocessing import Feature_Selector_Dataset
from demo.SeversonDataset_preprocess import train_val_split
import demo.discharge_model as discharge_model
from utils import *


def get_args_parser():
    parser = argparse.ArgumentParser('Discharge Model Feature Selector training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--seed', default=41, type=int)
    parser.add_argument('--detail_step', default=50, type=int)

    # Model parameters
    parser.add_argument('--model_name', default='Dim_Reduction_2', type=str) 
    parser.add_argument('--pred_target', default='chargetime', type=str) 
    parser.add_argument('--finetune', default=True, type=bool)   
    parser.add_argument('--load_checkpoint', default='Dim_Reduction_2_best_seed41.pth', type=str)                  

    # Hyperparameters
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--lr_schedule', type=bool, default=False, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR')
    parser.add_argument('--warm_up', type=int, default=10, metavar='LR')
    parser.add_argument('--delta', type=int, default=1)

    return parser

 
def main(args):
    if torch.cuda.is_available():
        print(" -- GPU is available -- ")

    # 根據random seed，隨機分割訓練及測試集
    train_val_split(seed=args.seed)

    # pytorch_dataset_preprocessing(seed=args.seed, folder='Severson_Dataset/')
    trn_set = Feature_Selector_Dataset(train=True, pred_target=args.pred_target, part='discharge')
    val_set = Feature_Selector_Dataset(train=False, pred_target=args.pred_target, part='discharge')
    trn_loader = DataLoader(trn_set, batch_size=args.batch_size, num_workers=0, drop_last=False, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0, drop_last=False, shuffle=False)

    model = discharge_model.__dict__[args.model_name](4, 1, 0.2).apply(init_weights).cuda()
    summary(model, (4, 500))

    if args.finetune:
        model.load_state_dict(torch.load(args.load_checkpoint))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.anneal_period, eta_min=args.min_lr)
    # criterion = nn.MSELoss()
    criterion = nn.HuberLoss(delta=args.delta)

    best_mape = 1000
    trn_loss_record, val_loss_record = [], []
    for epoch in range(args.epochs):
        model.train()
        step = 0
        n_minibatch = math.ceil(len(trn_set)/args.batch_size)
        if args.lr_schedule:
            adjust_learning_rate(optimizer, args.epochs, epoch+1, args.warm_up, args.lr, args.min_lr)
        for inputs, targets in trn_loader:
            step += 1
            optimizer.zero_grad()
            outputs = model(inputs.cuda().float())
            loss = criterion(outputs, targets.reshape(-1, 1).cuda().float())
            loss.backward()
            optimizer.step()
            if step%args.detail_step==0:
                print('epoch:[%d / %d] batch:[%d / %d] loss: %.3f lr: %e' % (epoch+1, args.epochs, step, n_minibatch, loss, optimizer.param_groups[0]["lr"]))

        # model evaluation per epoch
        model.eval()
        trn_rmse, trn_mape = real_RMSE_and_MAPE(model, trn_loader, args.pred_target)
        val_rmse, val_mape = real_RMSE_and_MAPE(model, val_loader, args.pred_target)
        trn_loss_record.append(trn_rmse)
        val_loss_record.append(val_rmse)
        print('real trn RMSE: %.2f, MAPE: %.2f' % (trn_rmse, trn_mape))
        print('real val RMSE: %.2f, MAPE: %.2f' % (val_rmse, val_mape))
        if val_mape < best_mape:
            best_rmse, best_mape = val_rmse, val_mape
            pred_result(model, trn_set, 'trn', args.pred_target)        
            pred_result(model, val_set, 'val', args.pred_target)
            torch.save(model.state_dict(), args.model_name+'_seed'+str(args.seed)+'.pth')

    # training finished
    loss_profile(trn_loss_record, val_loss_record)
    print('best RMSE: %.2f, MAPE: %.2f' % (best_rmse, best_mape))


if __name__=='__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args) 