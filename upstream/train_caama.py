import argparse
import numpy as np
import torch
from auction import ContextualAffineMaximizerAuction
from net import Payment_Cor, Payment_Cor_max_min
from tqdm import tqdm
from logger import get_logger, save_model
import os
from gen_values import *

def str2bool(v):
    return v.lower() in ('true', '1') 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=2)
    parser.add_argument('--m_items', type=int, default=2)
    parser.add_argument('--dx', type=int, default=10)
    parser.add_argument('--dy', type=int, default=10)
    parser.add_argument('--menu_size', type=int, default=16)
    parser.add_argument('--deterministic', type=str2bool, default=False)
    parser.add_argument('--continuous_context', type=str2bool, default=False)
    parser.add_argument('--const_bidder_weights', type=str2bool, default=True) # 

    parser.add_argument('--d_emb', type=int, default=10)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_hidden', type=int, default=64)
    parser.add_argument('--init_softmax_temperature', type=int, default=500)
    parser.add_argument('--alloc_softmax_temperature', type=int, default=10)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--train_steps', type=int, default=2000)
    parser.add_argument('--train_sample_num', type=int, default = 32768)
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--eval_sample_num', type=int, default = 32768)
    parser.add_argument('--batch_size', type=int, default = 2048)
    parser.add_argument('--device', type=str, default='cuda:4')

    parser.add_argument('--lr', type=float, default = 3e-4)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--name', type=str, default='./results')
    parser.add_argument('--data', type=int, default=9)
    parser.add_argument('--alpha', type=float, default=1) # for data in {20, 21}
    parser.add_argument('--test_batch_size', type=int, default = 1000)
    parser.add_argument('--ablation', type=int, default=0, help='1 for w, 2 for b, 3 for w and b, 4 for deterministic')
    
    ## About the IR regret term. 
    parser.add_argument('--gamma', type=float, default = 3)
    parser.add_argument('--gamma_max', type=float, default = 20)
    parser.add_argument('--gamma_min', type=float, default = 1)
    parser.add_argument('--target_ir_regret', type=float, default = 0.001)
    parser.add_argument('--gamma_lr', type=float, default = 0.1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    if args.deterministic:
        args.menu_size = (args.n_agents + 1) ** args.m_items - 1
        
    file_path = f"{args.name}/{args.data}_{args.n_agents}_{args.m_items}_{args.menu_size}_{args.seed}_{args.alpha}"
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    log_path = f"{file_path}/record.log"
    logger = get_logger(log_path)
    logger.info(args)
    DEVICE = args.device
    
    func_name = f"generate_data_{args.data}"
    my_generate_sample = globals()[func_name]
    if args.data == 8:
        means = np.load('./data/data_8_means.npy')
        covs = np.load('./data/data_8_covs.npy')
        means = means[args.seed, :, :args.n_agents]
        covs = covs[args.seed, :, :args.n_agents, :args.n_agents]
    
    model = ContextualAffineMaximizerAuction(args).to(DEVICE)
    
    if args.load_path != None:
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.load_path).items()})

    payment_model = Payment_Cor_max_min(args).to(DEVICE)
    cur_softmax_temperature = args.init_softmax_temperature
    warm_up_iters = 100
    warm_up_init = 1e-8
    warm_up_end = args.lr
    warm_up_anneal_increase = (warm_up_end - warm_up_init) / warm_up_iters

    optimizer = torch.optim.Adam([
        {'params': model.citransnet.parameters(), 'lr': warm_up_init},
        {'params': payment_model.parameters(), 'lr': warm_up_init}
    ])
    
    current_gamma = args.gamma
    bs = args.batch_size
    num_per_train = int(args.train_sample_num / bs)
    for i in tqdm(range(1, args.train_steps+1)):
        if i % args.eval_freq == 0: # eval
            if i == args.train_steps:
                save_model(model, f"{file_path}/model", i, args)
                save_model(payment_model, f"{file_path}/model_cor", i, args)

            with torch.no_grad():
                if args.data == 8:
                    test_values, test_X, test_Y = my_generate_sample(args.eval_sample_num, 
                                                                 args.n_agents, args.m_items, args.alpha, 
                                                                 args.dx, args.dy, device=DEVICE, means=means, covs=covs)
                else:
                    test_values, test_X, test_Y = my_generate_sample(args.eval_sample_num, 
                                                                    args.n_agents, args.m_items, args.alpha, args.dx, args.dy, DEVICE)
                revenue = torch.zeros(1).to(DEVICE)
                ir_regret = torch.zeros(1).to(DEVICE)
                for num in range(int(test_values.shape[0] / bs)):
                    choice_id, _, payment, allocs, _, _, _, valuation = model.test_time_forward(test_values[num*bs:(num+1)*bs],
                                                                            test_X[num*bs:(num+1)*bs], 
                                                                            test_Y[num*bs:(num+1)*bs])
                    payment_cor = payment_model(test_values[num*bs:(num+1)*bs], 
                                          test_values[num*bs:(num+1)*bs], 
                                          test_values[num*bs:(num+1)*bs])
                    revenue += (payment+payment_cor).sum()
                    utility_ama = valuation - payment
                    utility = utility_ama - payment_cor
                    ir_regret += torch.clamp(- utility, min=0).sum()
                    
                    if num == 0:
                        logger.info(f"value {test_values[0].data}, valuation_ama {valuation[:, 0].data}, payment_ama {payment[:, 0].data} payment_cor {payment_cor[:, 0].data}")
                revenue /= test_values.shape[0]
                ir_regret /= test_values.shape[0]
                
                logger.info(f"step {i}: revenue: {revenue}, ir_regret: {ir_regret}")

        if args.data == 8:
            train_values, train_X, train_Y = my_generate_sample(args.train_sample_num, 
                                                            args.n_agents, args.m_items, args.alpha,
                                                            args.dx, args.dy, device=DEVICE, covs=covs, means=means)
        else:  
            train_values, train_X, train_Y = my_generate_sample(args.train_sample_num, 
                                                            args.n_agents, args.m_items, args.alpha,
                                                            args.dx, args.dy, DEVICE)
        ama_rev, cor_rev, ir_regret = 0, 0, 0
        for num in range(num_per_train): # train
            optimizer.zero_grad()
            _, _, payment, allocs, valuation = model(train_values[num*bs:(num+1)*bs], 
                                          train_X[num*bs:(num+1)*bs], 
                                          train_Y[num*bs:(num+1)*bs], 
                                          cur_softmax_temperature)
            payment_cor = payment_model(train_values[num*bs:(num+1)*bs], 
                                          train_X[num*bs:(num+1)*bs], 
                                          train_Y[num*bs:(num+1)*bs])

            loss_revenue = - (payment + payment_cor).sum(0).mean()
            utility_ama = (valuation - payment)
            utility = utility_ama - payment_cor
            loss_ir = torch.clamp(- utility, min=0).sum(0).mean()
            loss_aug = utility_ama.sum(0).mean()
            loss = loss_revenue + current_gamma * loss_ir
            
            loss.backward()
            ama_rev += payment.sum(0).mean().data
            cor_rev += payment_cor.sum(0).mean().data
            ir_regret += loss_ir.data
            optimizer.step()
            
            
        if i % 5 == 0:
            logger.info(f"step {i}: ama_rev: {(ama_rev / num_per_train):.4f}, cor_rev {(cor_rev / num_per_train):.4f}, ir_regret {(ir_regret / num_per_train):.4f}, current gamma {current_gamma}.")

        if i > warm_up_iters:
            current_gamma += args.gamma_lr * (torch.log(ir_regret / num_per_train) - torch.log(torch.tensor(args.target_ir_regret)))
            current_gamma = current_gamma + (current_gamma >= args.gamma_max) * (args.gamma_max - current_gamma) + (current_gamma <= args.gamma_min) * (args.gamma_min - current_gamma)
            
        if i <= warm_up_iters: # warm up
            for p in optimizer.param_groups:
                p['lr'] += warm_up_anneal_increase

    # test 
    # logger.info("------------Final test------------")
    # bs = args.test_batch_size
    # DEVICE = 'cpu'
    # test_values, test_X, test_Y = torch.load(f'./data/{args.n_agents}_{args.m_items}_data_{args.data}_test.pt')
    # test_values, test_X, test_Y = test_values.to(DEVICE), test_X.to(DEVICE), test_Y.to(DEVICE)
    # test_num = test_values.shape[0]
    # model = model.to('cpu')
    # model.device = 'cpu'
    # with torch.no_grad():
    #     revenue = torch.zeros(1).to(DEVICE)
    #     for num in tqdm(range(int(test_num / bs))):
    #         choice_id, _, payment, allocs, w, b, removed_choice_id = model.test_time_forward(test_values[num*bs:(num+1)*bs], 
    #                                                                                          test_X[num*bs:(num+1)*bs], 
    #                                                                                          test_Y[num*bs:(num+1)*bs])
    #         revenue += payment.sum()
    #     revenue /= test_num
    # logger.info(f"Final test revenue: {revenue}")