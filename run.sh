
# source model training
python -u image_source_backdoor.py --trte val --output 'ckps/source/' --da uda --gpu_id 0 --dset office --max_epoch 100 --s 0 --seed 2020 --backdoor 'Blended'
python -u image_source_backdoor.py --trte val --output 'ckps/source/' --da uda --gpu_id 0 --dset office-home --max_epoch 50 --s 0 --seed 2020 --backdoor 'Blended'
python -u image_source_backdoor.py --trte val --output 'ckps/source/' --da uda --gpu_id 0 --dset DoaminNet126 --max_epoch 50 --s 0 --seed 2020 --backdoor 'Blended'

# direct model adaptation (vulnerable to universal attacks)
python -u image_target.py --cls_par 0.3 --da uda --dset office --gpu_id 0 --s 0 --t 1 --output_src 'ckps/source/' --output 'ckps/target/' --seed 2020 --backdoor 'Blended'
python -u image_target.py --cls_par 0.3 --da uda --dset office-home --gpu_id 0 --s 0 --t 1 --output_src 'ckps/source/' --output 'ckps/target/' --seed 2020 --backdoor 'Blended'
python -u image_target.py --cls_par 0.3 --da uda --dset DoaminNet126 --gpu_id 0 --s 0 --t 1 --output_src 'ckps/source/' --output 'ckps/target/' --seed 2020 --backdoor 'Blended'

# AdaptGuard
python -u image_ft_adaptguard.py --dset office --gpu_id 0  --da uda --s 0 --t 1 --kd_max_epoch 50 --output_src 'ckps/source/' --output 'ckps/source_adaptguard/' --seed 2020 --backdoor 'Blended'
python -u image_ft_adaptguard.py --dset office-home --gpu_id 0  --da uda --s 0 --t 1 --kd_max_epoch 50 --output_src 'ckps/source/' --output 'ckps/source_adaptguard/' --seed 2020 --backdoor 'Blended'
python -u image_ft_adaptguard.py --dset DoaminNet126 --gpu_id 0  --da uda --s 0 --t 1 --kd_max_epoch 50 --output_src 'ckps/source/' --output 'ckps/source_adaptguard/' --seed 2020 --backdoor 'Blended'

# then model adaptation (defend against universal attacks)
python -u image_target.py --cls_par 0.3 --da uda --dset office --gpu_id 0 --s 0 --t 1 --output_src 'ckps/source_adaptguard/' --output 'ckps/target_adaptguard/' --seed 2020 --backdoor 'Blended'
python -u image_target.py --cls_par 0.3 --da uda --dset office-home --gpu_id 0 --s 0 --t 1 --output_src 'ckps/source_adaptguard/' --output 'ckps/target_adaptguard/' --seed 2020 --backdoor 'Blended'
python -u image_target.py --cls_par 0.3 --da uda --dset DoaminNet126 --gpu_id 0 --s 0 --t 1 --output_src 'ckps/source_adaptguard/' --output 'ckps/target_adaptguard/' --seed 2020 --backdoor 'Blended'

