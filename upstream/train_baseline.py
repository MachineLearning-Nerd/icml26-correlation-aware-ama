import argparse
import torch
from auction import ContextualAffineMaximizerAuction
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
    parser.add_argument('--device', type=str, default='cuda:6')

    parser.add_argument('--lr', type=float, default = 3e-4)
    parser.add_argument('--decay_round_one', type=int, default = 3000) #
    parser.add_argument('--one_lr', type=float, default = 5e-5) #
    parser.add_argument('--decay_round_two', type=int, default = 6000) #
    parser.add_argument('--two_lr', type=float, default = 1e-5) #
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--name', type=str, default='./results_baseline')
    parser.add_argument('--data', type=int, default=9)
    parser.add_argument('--alpha', type=float, default=1) # for data in {20, 21}

    # parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--test_batch_size', type=int, default = 1000)
    parser.add_argument('--ablation', type=int, default=0, help='1 for w, 2 for b, 3 for w and b, 4 for deterministic')
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

    cur_softmax_temperature = args.init_softmax_temperature
    warm_up_iters = 100
    warm_up_init = 1e-8
    warm_up_end = args.lr
    warm_up_anneal_increase = (warm_up_end - warm_up_init) / warm_up_iters
    optimizer = torch.optim.Adam(model.citransnet.parameters(), lr=warm_up_init)

    bs = args.batch_size
    num_per_train = int(args.train_sample_num / bs)
    for i in tqdm(range(1, args.train_steps+1)):
        if i % args.eval_freq == 0: # eval
            if i == args.train_steps:
                save_model(model, f"{file_path}/model", i, args)
            with torch.no_grad():
                if args.data == 8:
                    test_values, test_X, test_Y = my_generate_sample(args.eval_sample_num, 
                                                                 args.n_agents, args.m_items, args.alpha,
                                                                 args.dx, args.dy, device=DEVICE, means=means, covs=covs)
                else:
                    test_values, test_X, test_Y = my_generate_sample(args.eval_sample_num, 
                                                                 args.n_agents, args.m_items, args.alpha,
                                                                 args.dx, args.dy, DEVICE)
                revenue = torch.zeros(1).to(DEVICE)
                for num in range(int(test_values.shape[0] / bs)):
                    choice_id, _, payment, allocs, _, _, _, valuation = model.test_time_forward(test_values[num*bs:(num+1)*bs],
                                                                            test_X[num*bs:(num+1)*bs], 
                                                                            test_Y[num*bs:(num+1)*bs])
                    revenue += payment.sum()
                revenue /= test_values.shape[0]
                logger.info(f"step {i}: revenue: {revenue}")
        if args.data == 8:
            train_values, train_X, train_Y = my_generate_sample(args.train_sample_num, 
                                                            args.n_agents, args.m_items, args.alpha,
                                                            args.dx, args.dy, covs=covs, means=means, device=DEVICE)
        else: 
            train_values, train_X, train_Y = my_generate_sample(args.train_sample_num, 
                                                            args.n_agents, args.m_items, args.alpha,
                                                            args.dx, args.dy, DEVICE)
        reportrev = 0
        for num in range(num_per_train): # train
            optimizer.zero_grad()
            _, _, payment, allocs, valuation = model(train_values[num*bs:(num+1)*bs], 
                                          train_X[num*bs:(num+1)*bs], 
                                          train_Y[num*bs:(num+1)*bs], 
                                          cur_softmax_temperature)
            loss_revenue = - payment.sum(0).mean() 
            reportrev += payment.sum(0).mean().data 
            loss_revenue.backward()
            optimizer.step()
            
        if i % 5 == 0:
            logger.info(f"step {i}: payment: {(reportrev / num_per_train):.4f}.")

        if i <= warm_up_iters: # warm up
            for p in optimizer.param_groups:
                p['lr'] += warm_up_anneal_increase
                
        # if i == args.decay_round_one:
        #     for p in optimizer.param_groups:
        #         p['lr'] = args.one_lr 
                    
        # if i == args.decay_round_two:
        #     for p in optimizer.param_groups:
        #         p['lr'] = args.two_lr

    # test 
    logger.info("------------Final test------------")
    bs = args.test_batch_size
    DEVICE = 'cpu'
    test_values, test_X, test_Y = torch.load(f'./data/{args.n_agents}_{args.m_items}_data_{args.data}_test.pt')
    test_values, test_X, test_Y = test_values.to(DEVICE), test_X.to(DEVICE), test_Y.to(DEVICE)
    test_num = test_values.shape[0]
    model = model.to('cpu')
    model.device = 'cpu'
    with torch.no_grad():
        revenue = torch.zeros(1).to(DEVICE)
        for num in tqdm(range(int(test_num / bs))):
            choice_id, _, payment, allocs, w, b, removed_choice_id = model.test_time_forward(test_values[num*bs:(num+1)*bs], 
                                                                                             test_X[num*bs:(num+1)*bs], 
                                                                                             test_Y[num*bs:(num+1)*bs])
            revenue += payment.sum()
        revenue /= test_num
    logger.info(f"Final test revenue: {revenue}")