import argparse
import torch
from auction import ContextualAffineMaximizerAuction, VCG
from net import Payment_Cor
from tqdm import tqdm
import numpy as np
from gen_values import *
from IPython import embed
from logger import load_model
import os

def str2bool(v):
    return v.lower() in ('true', '1') 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=2)
    parser.add_argument('--m_items', type=int, default=2)
    parser.add_argument('--dx', type=int, default=10)
    parser.add_argument('--dy', type=int, default=10)
    parser.add_argument('--menu_size', type=int, default=32)
    parser.add_argument('--deterministic', type=str2bool, default=False)
    parser.add_argument('--continuous_context', type=str2bool, default=False)
    parser.add_argument('--const_bidder_weights', type=str2bool, default=True) # 

    parser.add_argument('--d_emb', type=int, default=10)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_hidden', type=int, default=64)

    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--alloc_softmax_temperature', type=int, default=10, help='tau_A')
    parser.add_argument('--ama_load_path_ckpt', type=str, default=None)
    parser.add_argument('--cor_load_path_ckpt', type=str, default=None)
    parser.add_argument('--VCG', type=str2bool, default=False)
    parser.add_argument('--ablation', type=int, default=0)
    parser.add_argument('--data', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--residual', type=float, default=0.0)
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    DEVICE = args.device
    VCG_test= args.VCG
    
    func_name = f"generate_data_{args.data}"
    my_generate_sample = globals()[func_name]
    if args.data == 8:
        means = np.load('./data/data_8_means.npy')
        covs = np.load('./data/data_8_covs.npy')
        means = means[args.seed, :, :args.n_agents]
        covs = covs[args.seed, :, :args.n_agents, :args.n_agents]
    
    if args.deterministic:
        args.menu_size = (args.n_agents + 1) ** args.m_items - 1
        
    torch.manual_seed(2002)
    if not VCG_test:
        model_path = f"./results/{args.data}_{args.n_agents}_{args.m_items}_{args.menu_size}_{args.seed}_{args.alpha}/model_{args.ama_load_path_ckpt}.pt"
        model, _ = load_model(ContextualAffineMaximizerAuction, model_path, device=DEVICE)
        model.eval()
        
        model_path = f"./results/{args.data}_{args.n_agents}_{args.m_items}_{args.menu_size}_{args.seed}_{args.alpha}/model_cor_{args.cor_load_path_ckpt}.pt"
        print(model_path)
        payment_model, _ = load_model(Payment_Cor, model_path, device=DEVICE)   
    
    
    bs = args.batch_size
    if args.data == 8:
        data_path = f"./data/{args.n_agents}_{args.m_items}_data_{args.data}_test_{args.seed}.pt"
        if os.path.exists(data_path):
            print("Test data file exists!")
        else:
            print("Test data file does not exist! New test data created.")
            x = my_generate_sample(20000, args.n_agents, args.m_items, args.alpha, args.dx, args.dy, device='cpu', means=means, covs=covs)
            torch.save(x, data_path)
    else:
        data_path = f"./data/{args.n_agents}_{args.m_items}_data_{args.data}_test_{args.alpha}.pt"
        if os.path.exists(data_path):
            print("Test data file exists!")
        else:
            print("Test data file does not exist! New test data created.")
            x = my_generate_sample(20000, args.n_agents, args.m_items, args.alpha, args.dx, args.dy, device='cpu')
            torch.save(x, data_path)

    test_values, test_X, test_Y = torch.load(data_path)
    test_values, test_X, test_Y = test_values.to(DEVICE), test_X.to(DEVICE), test_Y.to(DEVICE)

    test_num = test_values.shape[0] 
    alloc_list, revenue_list = [], [] # added
    choice_id_record = torch.zeros(args.menu_size + 1)
    removed_choice_id_record = torch.zeros(args.menu_size + 1)
    randomize_cnt = 0
    tot_cnt = 0
    thre = 0
    with torch.no_grad():
        revenue = torch.zeros(1).to(DEVICE)
        valid_revenue = torch.zeros(1).to(DEVICE)
        ir_regret = torch.zeros(1).to(DEVICE)
        for num in tqdm(range(int(test_num / bs))):
            if not VCG_test:
                choice_id, _, payment, allocs, w, b, removed_choice_id, valuation = model.test_time_forward(test_values[num*bs:(num+1)*bs], test_X[num*bs:(num+1)*bs], test_Y[num*bs:(num+1)*bs])
                
                payment_cor = payment_model(test_values[num*bs:(num+1)*bs], test_values[num*bs:(num+1)*bs], test_values[num*bs:(num+1)*bs])
                payment_cor = payment_cor - args.residual
                payment_cor[payment_cor < 0] = 0
                
                revenue += (payment+payment_cor).sum()
                utility_ama = valuation - payment
                utility = utility_ama - payment_cor
                ir_regret += torch.clamp(- utility, min=0).sum()
                valid_revenue += torch.where(utility >= - thre, payment+payment_cor, torch.zeros_like(payment+payment_cor)).sum()
                alloc_list.append(choice_id.cpu().numpy()) # added 
                revenue_list.append((payment+payment_cor).sum(0).cpu().numpy()) # added
            else:
                choice_id, _, payment, allocs = VCG(test_values[num*bs:(num+1)*bs], DEVICE)
                revenue += payment.sum()
        valid_revenue /= test_num
        revenue /= test_num
        ir_regret /= test_num
    # print(revenue, valid_revenue)
    with open('result.txt', 'a') as f:
        f.write(f"setting: {args.n_agents}_{args.m_items}_{args.data}_{args.alpha}_{args.seed}, revenue: {revenue}, valid revenue: {valid_revenue}, ir_regret: {ir_regret}\n")
    # embed()
    # np.save(f'./results/{args.data}_{args.n_agents}_{args.m_items}_{args.menu_size}_1_{args.alpha}/model_{args.cor_load_path_ckpt}_alloc.npy', alloc_list)
    # np.save(f'./results/{args.data}_{args.n_agents}_{args.m_items}_{args.menu_size}_1_{args.alpha}/model_{args.cor_load_path_ckpt}_revenue.npy', revenue_list)  
    # np.save(f'./results/{args.data}_{args.n_agents}_{args.m_items}_{args.menu_size}_1_{args.alpha}/model_{args.cor_load_path_ckpt}_a.npy', allocs[0])    
  
    # torch.save((choice_id_record, removed_choice_id_record, allocs, w, b), 
    #            f'{args.n_agents}_{args.m_items}_{args.menu_size}_record.pt')
