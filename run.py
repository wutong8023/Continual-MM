'''
fine tuning

Author: Tong
Time: 10-02-2020
'''

import os
import time
from run_exp import \
    exp_1_1_vanilla, exp_1_0_joint, exp_1_2_er, exp_1_3_ewc, exp_1_4_derpp, exp_1_5_hat, \
    exp_2_0_joint, exp_2_2_size_er, exp_2_3_ewc_lmd, exp_2_4_size_derpp, \
    exp_4_0_prob_task_joint, exp_4_1_prob_task_vanilla, \
    exp_4_2_prob_task_er, exp_4_3_prob_task_ewc, exp_4_4_prob_task_derpp, exp_4_5_prob_task_hat, \
    exp_5_0_prob_joint, exp_5_1_prob_vanilla, exp_5_2_prob_er, exp_5_3_prob_ewc, exp_5_4_prob_derpp, \
    exp_5_visual_prob, \
    exp_6_0_comput, \
    exp_7_0_dataset, \
    exp_8_visual_main


def batch_submit_multi_jobs(cmd_list, info_list, platform: str, split_num: int = 4, partition="g"):
    assert len(cmd_list) == len(info_list)
    
    content = []
    file_name = "./job_base_{pltf}.sh".format(pltf=platform)
    file_out = "./job_{pltf}.sh".format(pltf=platform)
    
    cmd_list_frac = []
    info_list_frac = []
    
    flag_idx = 0
    while flag_idx < len(cmd_list):
        if (flag_idx + split_num) <= len(cmd_list):
            next_flag_idx = flag_idx + split_num
        else:
            next_flag_idx = len(cmd_list)
        
        sub_cmd_list = cmd_list[flag_idx:next_flag_idx:]
        sub_info_list = info_list[flag_idx:next_flag_idx:]
        
        cmd_list_frac.append(sub_cmd_list)
        info_list_frac.append(sub_info_list)
        
        flag_idx = next_flag_idx
    
    with open(file_name) as in_file:
        for line in in_file:
            content.append(line)
    for i, sub_cmd_list in enumerate(cmd_list_frac):
        with open(file_out, "w") as out_file:
            
            # job_name
            job_name = "__".join(info_list_frac[i])
            print("- JOB NAME: ", job_name)
            if platform == "group":
                _info = "#SBATCH -J {job_name}\n".format(job_name=job_name)
                content[21] = _info
                # SBATCH -o log/fs2s-iwslt-%J.out
                # SBATCH -e log/fs2s-iwslt-%J.err
                _out_file = "#SBATCH -o log/%J-{job_name}.out\n".format(job_name=job_name)
                content[15] = _out_file
                _err_file = "#SBATCH -e log/%J-{job_name}.err\n".format(job_name=job_name)
                content[16] = _err_file
            
            else:
                _partition = "#SBATCH --partition={var}\n".format(var=partition)
                content[2] = _partition
                _info = "#SBATCH --job-name={job_name}\n".format(job_name=job_name)
                content[3] = _info
                
                # SBATCH --output=log/fs2s-iwslt-%j.out
                # SBATCH --error=log/fs2s-iwslt-%j.err
                _out_file = "#SBATCH --output=log/%j-{job_name}.out\n".format(job_name=job_name)
                content[4] = _out_file
                _err_file = "#SBATCH --error=log/%j-{job_name}.err\n".format(job_name=job_name)
                content[5] = _err_file
            
            for line in content:
                out_file.write(line)
            
            # command
            if platform == "m3":
                pltf = "  --pltf {pltf}".format(pltf="m")
            else:
                pltf = "  --pltf {pltf}".format(pltf="gp")
            for cmd in sub_cmd_list:
                cmd = cmd + pltf
                out_file.write(cmd)
                out_file.write("\n\n")
        cmd = "sbatch job_{pltf}.sh".format(pltf=platform)
        os.system(cmd)


def batch_run_interactive(cmd_list: [str], order=1):
    # print(cmd_list)
    for i in cmd_list[::order]:
        print(i)
    for i in cmd_list[::order]:
        try:
            i = i + "  --pltf m"
            os.system(i)
            time.sleep(10)
        except:
            print(i, " failed!")


# cancel slurm jobs
def batch_cancel(job_start: int, num: int, platform: str):
    for i in range(job_start, job_start + num):
        if platform == "group":
            cmd = "scancel -v {i}".format(i=i)
        else:
            cmd = "scancel {i}".format(i=i)
        os.system(cmd)


if __name__ == '__main__':
    # for test

    # main experiment
    set1_0 = [exp_1_0_joint]
    set1_1 = [exp_1_1_vanilla]
    set1_2 = [exp_1_2_er]
    set1_3 = [exp_1_3_ewc]
    set1_4 = [exp_1_4_derpp]
    set1_5 = [exp_1_5_hat]
    # set1_5 = [exp_1_5_hat]
    set1 = [exp_1_0_joint, exp_1_1_vanilla, exp_1_2_er, exp_1_3_ewc]
    
    
    # hyper-parameter search
    set2 = [exp_2_0_joint, exp_2_2_size_er, exp_2_3_ewc_lmd]
    set2_0 = [exp_2_0_joint]
    set2_2 = [exp_2_2_size_er]
    set2_3 = [exp_2_3_ewc_lmd]
    set2_4 = [exp_2_4_size_derpp]
    
    # probing task experiment
    set4 = [exp_4_0_prob_task_joint, exp_4_5_prob_task_hat]
    set4_all = [exp_4_3_prob_task_ewc, exp_4_0_prob_task_joint, exp_4_1_prob_task_vanilla, exp_4_2_prob_task_er, exp_4_4_prob_task_derpp]
    set4_0 = [exp_4_0_prob_task_joint]
    set4_1 = [exp_4_1_prob_task_vanilla]
    set4_2 = [exp_4_2_prob_task_er]
    set4_4 = [exp_4_4_prob_task_derpp]
    set4_3 = [exp_4_3_prob_task_ewc]
    set4_5 = [exp_4_5_prob_task_hat]
    
    # probing experiment
    set5_v = [exp_5_visual_prob]
    set5 = [exp_5_0_prob_joint, exp_5_1_prob_vanilla, exp_5_2_prob_er, exp_5_3_prob_ewc, exp_5_4_prob_derpp]
    set5_0 = [exp_5_0_prob_joint]
    set5_1 = [exp_5_1_prob_vanilla]
    set5_2 = [exp_5_3_prob_ewc]
    set5_3 = [exp_5_4_prob_derpp]
    
    # computation analysis
    set6 = [exp_6_0_comput]
    
    # dataset analysis
    set7 = [exp_7_0_dataset]
    
    # main result analysis
    set8 = [exp_8_visual_main]
    
    for exp in set8:
        # select experiment to do
        cmd_list, info_list = exp.generate_cmd()
        
        # cmd: select the running platform and the corresponding shell template
        platform = {0: "m3", 1: "group"}
        pltf = 1
        
        # cmd: running in an interactive session
        batch_run_interactive(cmd_list, order=1)
        
        # cmd: submit batch jobs for multi jobs
        # optional partition for m3: dgx , m3g, m3h, m3e
        # batch_submit_multi_jobs(cmd_list, info_list, platform[pltf], split_num=1, partition="m3g")
    
    # cmd: cancel jobs
    # batch_cancel(18595657, 10, platform=platform[pltf])
    




# def generate_cmd():
#     cmd_list = []
#     info_list = []
#
#     # for epsilon in [0.01, 0.1, 0.3, 1]:
#     #     cmd = 'python -m utils.main --info finetune_epsilon ' \
#     #           '--model per4 --dataset seq-mnist --buffer_size 200 ' \
#     #           '--csv_log --tensorboard --lr 0.03 --minibatch_size 128 ' \
#     #           '--alpha 0.2 --beta 1.0 --batch_size 10 --n_epochs 1 --lmd 0.2 ' \
#     #           '--pseudo_size 2 --epsilon={epsilon}'.format(epsilon=epsilon)
#     #     info_list.append("")
#     #     cmd_list.append(cmd)
#
#     # ablation study on mnist
#     # for model in ['per3', 'per4', 'per401', 'per402', 'per403']:
#     #     cmd = 'python -m utils.main --info ablation_study_seq-mnist ' \
#     #           '--model {model} --dataset seq-mnist --buffer_size 200 ' \
#     #           '--csv_log --tensorboard --lr 0.03 --minibatch_size 128 ' \
#     #           '--alpha 0.2 --beta 1.0 --batch_size 10 --n_epochs 1 --lmd 0.2 ' \
#     #           '--pseudo_size 2 --epsilon=0.1'.format(model=model)
#     #
#     #     cmd_list.append(cmd)
#     #
#     # for model in ['der', 'derpp']:
#     #     cmd = 'python -m utils.main --info baseline_seq-cifar10 ' \
#     #           '--model {model} --dataset seq-mnist --buffer_size 200 ' \
#     #           '--csv_log --tensorboard --load_best_args'.format(model=model)
#     #
#     # ablation study on seq-cifar10
#     # for model in ['per3', 'per4', 'per401', 'per402', 'per403']:
#     #     cmd = 'python -m utils.main --info ablation_study_seq-cifar10 ' \
#     #           '--model {model} --dataset seq-cifar10 --buffer_size 200 ' \
#     #           '--csv_log --tensorboard --lr 0.03 --minibatch_size 32 ' \
#     #           '--alpha 0.1 --beta 0.5 --batch_size 32 --n_epochs 50 --lmd 0.2 ' \
#     #           '--pseudo_size 2 --epsilon=0.1'.format(model=model)
#     #     info_list.append("")
#     #     cmd_list.append(cmd)
#
#     # for model in ['der', 'derpp']:
#     #     cmd = 'python -m utils.main --info baseline_seq-cifar10 ' \
#     #           '--model {model} --dataset seq-cifar10 --buffer_size 200 ' \
#     #           '--csv_log --tensorboard --load_best_args'.format(model=model)
#     #
#     # # ablation study on seq-tinyimg
#     # for model in ['per3', 'per4', 'per401', 'per402', 'per403']:
#     #     cmd = 'python -m utils.main --info ablation_study_seq-tinyimg ' \
#     #           '--model {model} --dataset seq-tinyimg --buffer_size 200 ' \
#     #           '--csv_log --tensorboard --lr 0.03 --minibatch_size 32 ' \
#     #           '--alpha 0.1 --beta 1.0 --batch_size 32 --n_epochs 100 --lmd 0.2 ' \
#     #           '--pseudo_size 2 --epsilon=0.1'.format(model=model)
#     #     info_list.append("")
#     #     cmd_list.append(cmd)
#     #
#     # for model in ['der', 'derpp']:
#     #     for dataset in ["seq-tinyimg", "seq-cifar10"]:
#     #         cmd = 'python -m utils.main --info baseline_{dataset} ' \
#     #               '--model {model} --dataset {dataset} --buffer_size 200 ' \
#     #               '--csv_log --tensorboard --load_best_args'.format(dataset=dataset,model=model)
#     #         info_list.append("")
#     #         cmd_list.append(cmd)
#
#     # # ablation study on seq-cifar10
#     # for model in ["per407", "per408", ]:
#     #     for epsilon in [1.0, 2.0, 3.0]:
#     #         for lmd in [0.1, 0.3, 0.5, 1.0]:
#     #             # for alpha in [0.1, 0.3, 0.5, 1.0]:
#     #             #     for beta in [0.1, 0.3, 0.5, 1.0]:
#     #             cmd = 'python3 -m utils.main --info ft_epsilon_lmd_seq-cifar10 ' \
#     #                   '--model {model} --dataset seq-cifar10 --buffer_size 200 ' \
#     #                   '--csv_log --tensorboard --lr 0.03 --minibatch_size 32 ' \
#     #                   '--alpha 0.1 --beta 0.5 --batch_size 32 --n_epochs 50 --lmd {lmd} ' \
#     #                   '--pseudo_size 2 --epsilon={epsilon}'.format(model=model, epsilon=epsilon, lmd=lmd)
#     #             info_list.append("")
#     #             cmd_list.append(cmd)
#
#     # ablation study on seq-tinyimg
#     # for model in ['per404g']:
#     #     for epsilon in [2.0, 3.0]:
#     #         for lmd in [0.3, 0.5]:
#     #             cmd = 'python -m utils.main --info ft_epsilon_lmd_seq-tinyimg ' \
#     #                   '--model {model} --dataset seq-tinyimg --buffer_size 200 ' \
#     #                   '--csv_log --tensorboard --lr 0.03 --minibatch_size 32 ' \
#     #                   '--alpha 0.1 --beta 1.0 --batch_size 32 --n_epochs 100 --lmd {lmd} ' \
#     #                   '--pseudo_size 2 --epsilon={epsilon}'.format(model=model, epsilon=epsilon, lmd=lmd)
#     #             info_list.append("")
#     #             cmd_list.append(cmd)
#
#     # # # ablation study on seq-clinc150
#     # for model in ["per408_nlp_g"]:
#     #     for epsilon in [1.0]:
#     #         for lmd in [0.2]:
#     #             cmd = 'python3 -m utils.main --info ft_epsilon_lmd_webred ' \
#     #                   '--model {model} --area NLP --dataset seq-webred --buffer_size 200 ' \
#     #                   '--csv_log --tensorboard --lr 0.03 --minibatch_size 32 ' \
#     #                   '--alpha 0.1 --beta 0.5 --batch_size 32 --n_epochs 20 --lmd {lmd} ' \
#     #                   '--pseudo_size 2 --epsilon={epsilon}'.format(model=model, epsilon=epsilon, lmd=lmd)
#     #             info_list.append("")
#     #             cmd_list.append(cmd)
#
#     # PER408
#     # for model in ["per408_nlp"]:
#     #     for dataset in ["seq-clinc150"]:
#     #         for epsilon in [1.0]:
#     #             for lmd in [0.5]:
#     #                 for seed in [500]:
#     #                     for lr in [0.00001]:
#     #                         for ptm in ["bert", "roberta", "albert", "distilbert", "xlnet"]:
#     #                             cmd = 'python3 -m utils.main --info bad_case_analysis ' \
#     #                                   '--model {model} --area NLP --dataset {dataset} --buffer_size 200 ' \
#     #                                   '--csv_log --tensorboard --lr {lr} --minibatch_size 32 --seed {seed} ' \
#     #                                   '--alpha 0.1 --beta 0.5 --batch_size 32 --n_epochs 20 --lmd {lmd} --ptm {ptm} ' \
#     #                                   '--pseudo_size 2 --epsilon={epsilon}'.format(model=model, epsilon=epsilon,
#     #                                                                                lmd=lmd, ptm=ptm,
#     #                                                                                dataset=dataset, seed=seed, lr=lr)
#     #                             info_list.append("")
#     #                             cmd_list.append(cmd)
#
#     # for model in ["der_nlp"]:
#     #     for alpha in [1.0]:
#     #         for dataset in ["seq-clinc150"]:
#     #             for seed in range(1):
#     #                 seed = seed * 100
#     #                 for ptm in ["bert", "roberta", "albert", "distilbert", "xlnet"]:
#     #                     cmd = 'python3 -m utils.main --info debugging_test_{dataset} --seed {seed} ' \
#     #                           '--model {model} --area NLP --dataset {dataset} --buffer_size 200 ' \
#     #                           '--csv_log --tensorboard --lr 0.0001 --minibatch_size 32 --ptm {ptm} ' \
#     #                           '--alpha {alpha} --batch_size 32 --n_epochs 100'.format(model=model, alpha=alpha,
#     #                                                                                   dataset=dataset, seed=seed,
#     #                                                                                   ptm=ptm)
#     #                     info_list.append("")
#     #                     cmd_list.append(cmd)
#
#     # for model in ["derpp_nlp"]:
#     #     for alpha in [1.0]:
#     #         for beta in [0.5]:
#     #             for dataset in ["seq-clinc150"]:
#     #                 for ptm in ["bert", "roberta", "albert", "distilbert", "xlnet"]:
#     #                     info = "ptm_analysis"
#     #                     cmd = 'python3 -m utils.main --info {info} ' \
#     #                           '--model {model} --area NLP --dataset {dataset} --buffer_size 200 ' \
#     #                           '--csv_log --tensorboard --lr 0.0001 --minibatch_size 32 --ptm {ptm} ' \
#     #                           '--alpha {alpha} --beta {beta} --batch_size 32 --n_epochs 20'.format(model=model,
#     #                                                                                                alpha=alpha,
#     #                                                                                                beta=beta,
#     #                                                                                                dataset=dataset,
#     #                                                                                                ptm=ptm,
#     #                                                                                                info=info)
#     #                     info_list.append("")
#     #                     cmd_list.append(cmd)
#
#     '''ER analysis dataset / setting'''
#     # for model in ["er_nlp"]:
#     #     for beta in [1.0]:
#     #         for dataset in ["seq-clinc150", "seq-webred", "seq-maven"]:
#     #             for seed in range(1):
#     #                 seed = seed * 100
#     #                 for ptm in ["bert"]:
#     #                     for filter_rate in [1]:
#     #                         for prob_l in [-1]:
#     #                             info = "{description}{var}".format(description="er_", var=dataset)
#     #                             cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
#     #                                   '--model {model} --area NLP --dataset {dataset} --buffer_size 200 ' \
#     #                                   '--csv_log --tensorboard --lr 0.0001 --minibatch_size 32 --ptm {ptm} ' \
#     #                                   '--eval_freq 1 --prob_l {prob_l} ' \
#     #                                   '--beta {beta} --batch_size 32 --n_epochs 20'.format(model=model,
#     #                                                                                        beta=beta,
#     #                                                                                        dataset=dataset,
#     #                                                                                        seed=seed,
#     #                                                                                        ptm=ptm,
#     #                                                                                        info=info,
#     #                                                                                        prob_l=prob_l)
#     #                             info_list.append(info)
#     #                             cmd_list.append(cmd)
#
#     # '''ER analysis dataset / setting'''
#     # for model in ["er_nlp"]:
#     #     for beta in [1.0]:
#     #         for dataset in ["seq-clinc150"]:
#     #             for seed in range(1):
#     #                 seed = seed * 100
#     #                 for ptm in ["bert"]:
#     #                     for filter_rate in [1]:
#     #                         for prob_l in [-1]:
#     #                             info = "{description}{var}".format(description="er_", var=dataset)
#     #                             cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
#     #                                   '--model {model} --area NLP --dataset {dataset} --buffer_size 200 ' \
#     #                                   '--csv_log --tensorboard --lr 0.0001 --minibatch_size 32 --ptm {ptm} ' \
#     #                                   '--eval_freq 1 --prob_l {prob_l} ' \
#     #                                   '--beta {beta} --batch_size 16 --n_epochs 30'.format(model=model,
#     #                                                                                        beta=beta,
#     #                                                                                        dataset=dataset,
#     #                                                                                        seed=seed,
#     #                                                                                        ptm=ptm,
#     #                                                                                        info=info,
#     #                                                                                        prob_l=prob_l)
#     #                             info_list.append(info)
#     #                             cmd_list.append(cmd)
#     #
#     # """joint analysis"""
#     # for model in ["joint_nlp"]:
#     #     for dataset in ["seq-clinc150"]:
#     #         for seed in range(1):
#     #             seed = seed * 100
#     #             for ptm in ["bert", "roberta", "albert", "gpt2"]:
#     #                 for filter_rate in [1]:
#     #                     for prob_l in [12]:
#     #                         for lr in [0.00001]:
#     #                             for epoch in [20, 50, 100]:
#     #                                 info = "{description}{var}_{var3}".format(description="joint_", var=ptm,
#     #                                                                           var3=epoch)
#     #                                 cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
#     #                                       '--model {model} --area NLP --dataset {dataset} --increment_joint ' \
#     #                                       '--csv_log --tensorboard --lr {lr} --ptm {ptm} ' \
#     #                                       '--eval_freq 1 --prob_l {prob_l} --filter_rate {filter_rate} ' \
#     #                                       ' --batch_size 32 --n_epochs {epoch}'.format(model=model,
#     #                                                                                    dataset=dataset,
#     #                                                                                    seed=seed,
#     #                                                                                    ptm=ptm,
#     #                                                                                    info=info,
#     #                                                                                    filter_rate=filter_rate,
#     #                                                                                    prob_l=prob_l,
#     #                                                                                    lr=lr,
#     #                                                                                    epoch=epoch)
#     #                                 info_list.append(info)
#     #                                 cmd_list.append(cmd)
#
#     """vanilla analysis"""
#     # for model in ["vanilla_nlp"]:
#     #     for dataset in ["seq-clinc150"]:
#     #         for seed in range(1):
#     #             seed = seed * 100
#     #             for ptm in ["bert", "roberta", "albert", "gpt2"]:
#     #                 for filter_rate in [1]:
#     #                     for prob_l in [12]:
#     #                         for epoch in [30, 50, 100]:
#     #                             for lr in [0.00001]:
#     #                                 info = "{description}{var}_{var3}".format(description="joint_",
#     #                                                                           var=ptm,
#     #                                                                           var3=epoch)
#     #                                 cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
#     #                                       '--model {model} --area NLP --dataset {dataset} ' \
#     #                                       '--csv_log --tensorboard --lr {lr} --ptm {ptm} ' \
#     #                                       '--eval_freq 1 --prob_l {prob_l} --filter_rate {filter_rate} ' \
#     #                                       ' --batch_size 32 --n_epochs {epoch}'.format(model=model,
#     #                                                                                    dataset=dataset,
#     #                                                                                    seed=seed,
#     #                                                                                    ptm=ptm,
#     #                                                                                    info=info,
#     #                                                                                    filter_rate=filter_rate,
#     #                                                                                    prob_l=prob_l,
#     #                                                                                    epoch=epoch,
#     #                                                                                    lr=lr)
#     #                                 info_list.append(info)
#     #                                 cmd_list.append(cmd)
#
#     '''ER analysis_prob'''
#     # for model in ["er_nlp"]:
#     #     for beta in [1.0, 0.5, 0.7]:
#     #         for dataset in ["seq-clinc150"]:
#     #             for seed in range(1):
#     #                 seed = seed * 100
#     #                 for ptm in ["bert", "roberta", "albert", "gpt2"]:
#     #                     for filter_rate in [1]:
#     #                         for prob_l in [12]:
#     #                             for epoch in [50]:
#     #                                 for lr in [0.00001]:
#     #                                     info = "{description}{var}_{var1}".format(description="er_", var=ptm, var1=beta)
#     #                                     cmd = 'python3 -m utils.main --info {info} --seed {seed} ' \
#     #                                           '--model {model} --area NLP --dataset {dataset} --buffer_size 200 ' \
#     #                                           '--csv_log --tensorboard --minibatch_size 32 --ptm {ptm} ' \
#     #                                           '--eval_freq 1 --prob_l {prob_l} --filter_rate {filter_rate} ' \
#     #                                           '--beta {beta} --batch_size 32 --n_epochs {epoch} ' \
#     #                                           '--lr {lr}'.format(model=model,
#     #                                                              beta=beta,
#     #                                                              dataset=dataset,
#     #                                                              seed=seed,
#     #                                                              ptm=ptm,
#     #                                                              info=info,
#     #                                                              prob_l=prob_l,
#     #                                                              filter_rate=filter_rate,
#     #                                                              lr=lr,
#     #                                                              epoch=epoch)
#     #                                     info_list.append(info)
#     #                                     cmd_list.append(cmd)
#
#     return cmd_list, info_list
