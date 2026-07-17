# # bidder=2

# # for alpha in 1 0.8 0.6 0.4 0.2 0.0
# # do
# # python3 test_caama.py --n_agents $bidder --m_items 1 --menu_size 8 --data 21 --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_baseline.py --n_agents $bidder --m_items 1 --menu_size 8 --data 21 --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_caama.py --n_agents $bidder --m_items 2 --menu_size 32 --data 21 --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_baseline.py --n_agents $bidder --m_items 2 --menu_size 32 --data 21 --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_caama.py --n_agents $bidder --m_items 5 --menu_size 256 --data 21 --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_baseline.py --n_agents $bidder --m_items 5 --menu_size 256 --data 21 --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # done

# # bidder=2
# # data=20
# # for alpha in 1 0.8 0.6 0.4 0.2 0.0
# # do
# # python3 test_caama.py --n_agents $bidder --m_items 1 --menu_size 8 --data $data --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_baseline.py --n_agents $bidder --m_items 1 --menu_size 8 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_caama.py --n_agents $bidder --m_items 2 --menu_size 32 --data $data --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_baseline.py --n_agents $bidder --m_items 2 --menu_size 32 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_caama.py --n_agents $bidder --m_items 5 --menu_size 256 --data $data --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_baseline.py --n_agents $bidder --m_items 5 --menu_size 256 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # done

# # data=20
# # bidder=3
# # for alpha in 1 0.8 0.6 0.4 0.2 0.0
# # do
# # python3 test_caama.py --n_agents $bidder --m_items 1 --menu_size 8 --data $data --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_baseline.py --n_agents $bidder --m_items 1 --menu_size 8 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_caama.py --n_agents $bidder --m_items 2 --menu_size 32 --data $data --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_baseline.py --n_agents $bidder --m_items 2 --menu_size 32 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_caama.py --n_agents $bidder --m_items 3 --menu_size 128 --data $data --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_baseline.py --n_agents $bidder --m_items 3 --menu_size 128 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # done

# # data=20
# # bidder=5
# # for alpha in 1 0.8 0.6 0.4 0.2 0.0
# # do
# # python3 test_caama.py --n_agents $bidder --m_items 1 --menu_size 8 --data $data --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_baseline.py --n_agents $bidder --m_items 1 --menu_size 8 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_caama.py --n_agents $bidder --m_items 2 --menu_size 32 --data $data --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_baseline.py --n_agents $bidder --m_items 2 --menu_size 32 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_caama.py --n_agents $bidder --m_items 3 --menu_size 512 --data $data --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # python3 test_baseline.py --n_agents $bidder --m_items 3 --menu_size 512 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False
# # done


# bidder=2

# for alpha in 1 0.8 0.6 0.4 0.2 0.0
# do
# python3 test_baseline.py --n_agents $bidder --m_items 1 --menu_size 8 --data 21 --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --VCG True
# python3 test_baseline.py --n_agents $bidder --m_items 2 --menu_size 32 --data 21 --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --VCG True
# python3 test_baseline.py --n_agents $bidder --m_items 5 --menu_size 256 --data 21 --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --VCG True
# done

# bidder=2
# data=20
# for alpha in 1 0.8 0.6 0.4 0.2 0.0
# do
# python3 test_baseline.py --n_agents $bidder --m_items 1 --menu_size 8 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --VCG True
# python3 test_baseline.py --n_agents $bidder --m_items 2 --menu_size 32 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --VCG True
# python3 test_baseline.py --n_agents $bidder --m_items 5 --menu_size 256 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --VCG True
# done

# data=20
# bidder=3
# for alpha in 1 0.8 0.6 0.4 0.2 0.0
# do
# python3 test_baseline.py --n_agents $bidder --m_items 1 --menu_size 8 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --VCG True
# python3 test_baseline.py --n_agents $bidder --m_items 2 --menu_size 32 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --VCG True
# python3 test_baseline.py --n_agents $bidder --m_items 3 --menu_size 128 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --VCG True
# done

# data=20
# bidder=5
# for alpha in 1 0.8 0.6 0.4 0.2 0.0
# do
# python3 test_baseline.py --n_agents $bidder --m_items 1 --menu_size 8 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --VCG True
# python3 test_baseline.py --n_agents $bidder --m_items 2 --menu_size 32 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --VCG True
# python3 test_baseline.py --n_agents $bidder --m_items 3 --menu_size 512 --data $data --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --VCG True
# done


# bidder=2

# for alpha in 1 0.8 0.6 0.4 0.2 0.0
# do
# python3 test_caama.py --n_agents $bidder --m_items 5 --menu_size 256 --data 22 --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --alpha $alpha --continuous_context False --const_bidder_weights False
# python3 test_baseline.py --n_agents $bidder --m_items 5 --menu_size 256 --data 22 --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --const_bidder_weights False
# python3 test_baseline.py --n_agents $bidder --m_items 5 --menu_size 256 --data 22 --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --const_bidder_weights False --VCG True
# done


############# data 8 #############

#!/bin/bash
# for tuple in "2 2 32" "5 2 64" "8 2 128" "10 2 256" "2 3 64" "5 3 1024" "8 3 2048" "10 3 2048" "2 5 256" "5 5 2048"
# do
# for seed in 1 2 3 4 5
# do
#     read bidder item size <<< "$tuple"
#     python3 test_caama.py --n_agents $bidder --m_items $item --menu_size $size --data 8 --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --seed $seed --continuous_context False --const_bidder_weights False
#     # python3 test_baseline.py --n_agents $bidder --m_items $item --menu_size $size --data 8 --ama_load_path_ckpt 2000 --seed $seed --continuous_context False --const_bidder_weights False
#     # python3 test_baseline.py --n_agents $bidder --m_items $item --menu_size $size --data 8 --ama_load_path_ckpt 2000 --seed $seed --continuous_context False --const_bidder_weights False --VCG True
# done
# done

######### data 23
# for alpha in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1
# for alpha in 0.1
# do
# python3 test_caama.py --n_agents 2 --m_items 1 --menu_size 8 --data 23 --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --alpha $alpha --continuous_context False --const_bidder_weights False
# python3 test_baseline.py --n_agents 2 --m_items 1 --menu_size 8 --data 23 --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --const_bidder_weights False
# python3 test_baseline.py --n_agents 2 --m_items 1 --menu_size 8 --data 23 --ama_load_path_ckpt 2000 --alpha $alpha --continuous_context False --const_bidder_weights False --VCG True
# done



# !/bin/bash
for tuple in "2 2 32" "5 2 64" "8 2 128" "10 2 256" "2 3 64" "5 3 1024" "8 3 2048" "10 3 2048" "2 5 256" "5 5 2048"
do
for seed in 1 2 3 4 5
do
    read bidder item size <<< "$tuple"
    python3 test_caama.py --n_agents $bidder --m_items $item --menu_size $size --data 9 --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --seed $seed --continuous_context False --const_bidder_weights False --residual 0.01
    python3 test_baseline.py --n_agents $bidder --m_items $item --menu_size $size --data 9 --ama_load_path_ckpt 2000 --seed $seed --continuous_context False --const_bidder_weights False 
    python3 test_baseline.py --n_agents $bidder --m_items $item --menu_size $size --data 9 --ama_load_path_ckpt 2000 --seed $seed --continuous_context False --const_bidder_weights False --VCG True
done
done



# for residual in 0.05 0.02 0.01 0.005 0.002 0.001 
# do
#     python3 test_caama.py --n_agents 2 --m_items 2 --menu_size 32 --data 9 --ama_load_path_ckpt 2000 --cor_load_path_ckpt 2000 --seed 0 --continuous_context False --const_bidder_weights False --residual $residual
# done
