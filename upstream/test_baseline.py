import argparse
import torch
from auction import ContextualAffineMaximizerAuction, VCG
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
    # parser.add_argument('--cor_load_path_ckpt', type=str, default=None)
    parser.add_argument('--VCG', type=str2bool, default=False)
    parser.add_argument('--ablation', type=int, default=0)
    parser.add_argument('--data', type=int, default=9)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--seed', type=int, default=1)

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
        model_path = f"./results_baseline/{args.data}_{args.n_agents}_{args.m_items}_{args.menu_size}_{args.seed}_{args.alpha}/model_{args.ama_load_path_ckpt}.pt"
        model, _ = load_model(ContextualAffineMaximizerAuction, model_path, device=DEVICE)
        model.eval()
            
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
    with torch.no_grad():
        revenue = torch.zeros(1).to(DEVICE)
        for num in tqdm(range(int(test_num / bs))):
            if not VCG_test:
                choice_id, _, payment, allocs, w, b, removed_choice_id, valuations = model.test_time_forward(test_values[num*bs:(num+1)*bs], test_X[num*bs:(num+1)*bs], test_Y[num*bs:(num+1)*bs])
                # print(allocs[0], w[0], b[0])
            else:
                choice_id, _, payment, allocs = VCG(test_values[num*bs:(num+1)*bs], DEVICE)
            revenue += payment.sum()
            alloc_list.append(choice_id.cpu().numpy()) # added 
            revenue_list.append(payment.sum(0).cpu().numpy()) # added
            # for i in range(choice_id.shape[0]):
            #     flag = 0
            #     x = allocs[i, choice_id[i]]
            #     x = x[x>0.01]
            #     x = x[x<0.99]
            #     if x.shape[0] > 0:
            #         randomize_cnt += 1
            #         flag = 1
            #     choice_id_record[choice_id[i]] += 1
            #     for j in range(args.n_agents):
            #         x = allocs[i, removed_choice_id[j][i]]
            #         x = x[x>0.01]
            #         x = x[x<0.99]
            #         if x.shape[0] > 0:
            #             randomize_cnt += 1
            #             flag = 1
            #         removed_choice_id_record[removed_choice_id[j][i]] += 1
            #     if flag == 1:
            #         tot_cnt += 1
        revenue /= test_num
    # print(revenue)
    with open('result.txt', 'a') as f:
        f.write(f"setting: {args.n_agents}_{args.m_items}_{args.data}_{args.alpha}_{args.seed}, revenue: {revenue}\n")
    # print(randomize_cnt)

    # embed()
    # np.save(f'./results_baseline/{args.data}_{args.n_agents}_{args.m_items}_{args.menu_size}_1_{args.alpha}/model_{args.ama_load_path_ckpt}_alloc.npy', alloc_list)
    # np.save(f'./results_baseline/{args.data}_{args.n_agents}_{args.m_items}_{args.menu_size}_1_{args.alpha}/model_{args.ama_load_path_ckpt}_revenue.npy', revenue_list)  
    # np.save(f'./results_baseline/{args.data}_{args.n_agents}_{args.m_items}_{args.menu_size}_1_{args.alpha}/model_{args.ama_load_path_ckpt}_a.npy', allocs[0])    

    # torch.save((choice_id_record, removed_choice_id_record, allocs, w, b), 
    #            f'{args.n_agents}_{args.m_items}_{args.menu_size}_record.pt')
