pipeline {
    agent {
        docker { 
            image 'cuda:10.1-base-ubuntu18.04-custom' 
            args '--gpus all -no-cache'
        }
    }
    stages {
        stage('Setup') {
            steps {
                sh '''
                export LC_ALL=C.UTF-8
                export LANG=C.UTF-8
                export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                mkdir -p /var/jenkins_home/workspace/.cache
                cd self-ensemble
                python3 -m venv venv
                venv/bin/pip install -r requirements.txt
                '''
            }
        }
        stage('utom') {
            steps {
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                sh '''
                cd self-ensemble
                venv/bin/python trainDigitModel.py --gpu_id 0 --category utom --batch_size 128 --n_epoch 1
                '''
                }
            }
        }
        stage('mtou') {
            steps {                
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                sh '''
                export LC_ALL=C.UTF-8
                export LANG=C.UTF-8                
                cd self-ensemble
                venv/bin/python trainDigitModel.py --gpu_id 0 --category mtou --batch_size 128 --n_epoch 1
                '''
                }
            }
        }
        stage('stom') {
            steps {
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                sh '''
                export LC_ALL=C.UTF-8
                export LANG=C.UTF-8                
                cd self-ensemble
                venv/bin/python trainDigitModel.py --gpu_id 0 --category stom --batch_size 128 --n_epoch 1
                '''
                }
            }
        }
        stage('office atod') {
            steps {
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                sh '''
                export LC_ALL=C.UTF-8
                export LANG=C.UTF-8
                export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                mkdir -p export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                cd self-ensemble      
                mkdir -p results_office
                venv/bin/python trainResnet50.py --category=atod --log_file=results_office/res_office_atod_resnet50_run${2}.txt --result_file=results_office/history_office_atod_resnet50_run${2}.h5 --model_file=results_office/model_office_atod_resnet50_run${2}.pkl --gpu_id=0 --num_epochs=1
                '''
                }
            }
        }
        stage('office atow') {
            steps {
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                sh '''
                export LC_ALL=C.UTF-8
                export LANG=C.UTF-8
                export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                mkdir -p export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                cd self-ensemble
                mkdir -p results_office
                venv/bin/python trainResnet50.py --category=atow --log_file=results_office/res_office_atow_resnet50_run${2}.txt --result_file=results_office/history_office_atow_resnet50_run${2}.h5 --model_file=results_office/model_office_atow_resnet50_run${2}.pkl --gpu_id=0 --num_epochs=1
                '''
                }
            }
        }
        stage('office dtoa') {
            steps {
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                sh '''
                export LC_ALL=C.UTF-8
                export LANG=C.UTF-8                
                export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                mkdir -p export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                cd self-ensemble
                mkdir -p results_office
                venv/bin/python trainResnet50.py --category=dtoa --log_file=results_office/res_office_dtoa_resnet50_run${2}.txt --result_file=results_office/history_office_dtoa_resnet50_run${2}.h5 --model_file=results_office/model_office_dtoa_resnet50_run${2}.pkl --gpu_id=0 --num_epochs=1 
                '''
                }
            }
        }
        stage('office dtow') {
            steps {
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                sh '''
                export LC_ALL=C.UTF-8
                export LANG=C.UTF-8
                export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                mkdir -p export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                cd self-ensemble
                mkdir -p results_office
                venv/bin/python trainResnet50.py --category=dtow --log_file=results_office/res_office_dtow_resnet50_run${2}.txt --result_file=results_office/history_office_dtow_resnet50_run${2}.h5 --model_file=results_office/model_office_dtow_resnet50_run${2}.pkl --gpu_id=0 --num_epochs=1 
                '''
                }
            }
        }
        stage('office wtoa') {
            steps {
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                sh '''
                export LC_ALL=C.UTF-8
                export LANG=C.UTF-8
                export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                mkdir -p export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                cd self-ensemble
                mkdir -p results_office
                venv/bin/python trainResnet50.py --category=wtoa --log_file=results_office/res_office_wtoa_resnet50_run${2}.txt --result_file=results_office/history_office_wtoa_resnet50_run${2}.h5 --model_file=results_office/model_office_wtoa_resnet50_run${2}.pkl --gpu_id=0 --num_epochs=1 
                '''
                }
            }
        }
        stage('office wtod') {
            steps {
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                sh '''
                export LC_ALL=C.UTF-8
                export LANG=C.UTF-8
                export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                mkdir -p export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                cd self-ensemble
                mkdir -p results_office
                venv/bin/python trainResnet50.py --category=wtod --log_file=results_office/res_office_wtod_resnet50_run${2}.txt --result_file=results_office/history_office_wtod_resnet50_run${2}.h5 --model_file=results_office/model_office_wtod_resnet50_run${2}.pkl --gpu_id=0 --num_epochs=1 
                '''
                }
            }
        }
        stage('office atow adamix') {
            steps {
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                sh '''
                export LC_ALL=C.UTF-8
                export LANG=C.UTF-8
                export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                mkdir -p export XDG_CACHE_HOME=/var/jenkins_home/workspace/.cache
                cd self-ensemble
                mkdir -p results_office
                venv/bin/python trainResnet50.py --category=atow --log_file=results_office/res_office_atow_resnet50_run${2}.txt --result_file=results_office/history_office_atow_resnet50_run${2}.h5 --model_file=results_office/model_office_atow_resnet50_run${2}.pkl --adamix --img_size=224 --batch_size=32 --img_pad_width=0 --constrain_crop=-1 --src_hflip --tgt_hflip --epoch_size=target --unsup_weight=10.0 --cls_balance=0.0 --confidence_thresh=0.5 --num_epochs=100 --learning_rate=5e-4 --hide_progress_bar --num_threads=4 --gpu_id=0 
                '''
                }
            }
        }
stage('cleanup') {
            steps {
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                sh '''
                exit
                #  -- --model_file=results_office/model_office_atow_resnet50_run${2}.pkl --adamix -
                '''
                }
            }
        }
        
    }
}

