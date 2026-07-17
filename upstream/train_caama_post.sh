##########
## Normal Distribution:
## --data 8

# device="cuda:4"
# for seed in 1 2 3
# do
#     python3  train_caama_post.py --menu_size 32 --n_agents 2 --m_items 2 --device $device --name './results' --const_bidder_weights False  --gamma 3  --data 8 --seed $seed --load_ckpt 2000
#     python3  train_caama_post.py --menu_size 256 --n_agents 10 --m_items 2 --device $device --name './results' --const_bidder_weights False  --gamma 8  --data 8 --seed $seed --load_ckpt 2000
#     python3  train_caama_post.py --menu_size 64 --n_agents 5 --m_items 2 --device $device --name './results' --const_bidder_weights False  --gamma 6  --data 8 --seed $seed --load_ckpt 2000
#     python3  train_caama_post.py --menu_size 128 --n_agents 8 --m_items 2 --device $device --name './results' --const_bidder_weights False  --gamma 6  --data 8 --seed $seed --load_ckpt 2000
# done

# device="cuda:5"
# for seed in 1 2 3
# do
#     python3  train_caama_post.py --menu_size 64 --n_agents 2 --m_items 3 --device $device --name './results' --const_bidder_weights False  --gamma 5  --data 8 --seed $seed --load_ckpt 2000
#     python3  train_caama_post.py --menu_size 2048 --n_agents 10 --m_items 3 --device $device --name './results' --const_bidder_weights False  --gamma 8  --data 8 --seed $seed --load_ckpt 2000
# done

# device="cuda:6"
# for seed in 1 2 3
# do
#     python3  train_caama_post.py --menu_size 1024 --n_agents 5 --m_items 3 --device $device --name './results' --const_bidder_weights False  --gamma 6  --data 8 --seed $seed --load_ckpt 2000
#     python3  train_caama_post.py --menu_size 2048 --n_agents 8 --m_items 3 --device $device --name './results' --const_bidder_weights False  --gamma 8  --data 8 --seed $seed --load_ckpt 2000
# done

# device="cuda:7"
# for seed in 1 2 3
# do
#     python3  train_caama_post.py --menu_size 256 --n_agents 2 --m_items 5 --device $device --name './results' --const_bidder_weights False  --gamma 3  --data 8 --seed $seed --load_ckpt 2000
#     python3  train_caama_post.py --menu_size 2048 --n_agents 5 --m_items 5 --device $device --name './results' --const_bidder_weights False  --gamma 10  --data 8 --seed $seed --load_ckpt 2000
# done

##########
## Symmetric Positive:
## --data 20 
## Symmetric Negative:
## --data 21 
## Asymmetric Negative:
## --data 22

# device="cuda:0"
# for alpha in 1.0 0.8 0.6 0.4 0.2 0.0
# do
#     python3  train_caama_post.py --menu_size 256 --n_agents 2 --m_items 5 --device $device --name './results' --const_bidder_weights False  --gamma 6  --data 20 --alpha $alpha --load_ckpt 2000
#     python3  train_caama_post.py --menu_size 256 --n_agents 2 --m_items 5 --device $device --name './results' --const_bidder_weights False  --gamma 6  --data 21 --alpha $alpha --load_ckpt 2000
#     python3  train_caama_post.py --menu_size 256 --n_agents 2 --m_items 5 --device $device --name './results' --const_bidder_weights False  --gamma 6  --data 22 --alpha $alpha --load_ckpt 2000
# done

##########
## Asymmetric Equal Revenue Distribution
## --data 23

# device=cuda:7

# for alpha in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10
# do
#     python3  train_caama_post.py --menu_size 8 --n_agents 2 --m_items 1 --device $device --alpha $alpha --name './results' --const_bidder_weights False  --data 23 --seed 1 --gamma 5 --gamma_lr 0 --load_ckpt 2000
# done

