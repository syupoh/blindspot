#!/bin/sh
#SOURCE=$1
#TARGET=$2
#ARCH1=$3
#ARCH2=$4
#ARCH3=$5

# MYVAR="pre/users/joebloggs/domain.com" 

# # Remove the path leaving file name (all characters up to a slash):
# echo ${MYVAR##*/}
# # domain.com
# echo ${MYVAR#*/}
# # users/joebloggs/domain.com

# # Remove the file name, leaving the path (delete shortest match after last /):
# echo ${MYVAR%/*}
# # pre/users/joebloggs
# echo ${MYVAR%%/*}
# # pre

# # Get just the file extension (remove all before last period):
# echo ${MYVAR##*.}
# # com

# # NOTE: To do two operations, you can't combine them, but have to assign to an intermediate variable. So to get the file name without path or extension:

# NAME=${MYVAR##*/}      # remove part before last slash
# echo ${NAME%.*}        # from the new var remove the part after the last period
# domain

# ${MYVAR:3}   # Remove the first three chars (leaving 4..end)
# ${MYVAR::3}  # Return the first three characters
# ${MYVAR:3:5} # The next five characters after removing the first 3 (chars 4-9)
#

if [ -z "$1" ]
  then
    gpu="0"
  else
    gpu="${1}"
fi

myvar=$(pwd)
NAME=${myvar%/syupoh/*}/syupoh/ # retain the part before the /python/
# NAME=${NAME##*/}  # retain the part after the last slash
  
root=${NAME} # /data/syupoh


#  conda update -n base -c defaults conda
if [ ${gpu} -eq "0" ]
  then
    


    echo
  elif [ ${gpu} -eq "1" ]
  then
    
    data_set='
    d1.16
    '
    model_set='
    '${root}'dataset/endoscopy/JFD-03K/ckpt/Xception/timm3/211102-1547/best_cnn_timm3_211102-154718.h5
    '
    # '${root}'/data/syupoh/dataset/endoscopy/JFD-03K/ckpt/Xception/timm3/211102-1547/best_cnn_timm3_211102-154718.h5
    
    
    ## new version
    for dataname in ${data_set}
    do
      val_data=${root}/dataset/endoscopy/JFD-03K/jfd_dataset/${dataname}
      for model_path in ${model_set}
      do

        python jh_evaluate.py --gpu=${gpu} --scale  \
          --val_data=${val_data} \
          --save_dir=${root}/dataset/endoscopy/JFD-03K/grad_cam/ \
          --ckpt_path=${model_path} --cam
      done
    done

    


    # python jh_train.py --gpu=${gpu} \
    #   --train_path=${root}/dataset/endoscopy/JFD-03K/jfd_dataset/timm3/train \
    #   --val_path=${root}/dataset/endoscopy/JFD-03K/jfd_dataset/timm3/validation \
    #   --save_path=${root}/dataset/endoscopy/JFD-03K/ckpt/Xception/ -cv

#########################
    echo
  elif [ ${gpu} -eq "2" ]
  then
    data_set='
    d1.16
    video_LSS_10
    video_MSN_09
    video_YSK_11
    '
    model_set='
    '${root}'/dataset/endoscopy/JFD-03K/ckpt/Xception/_prev/fold_1_*.h5
    '

    
    # '${root}'/dataset/endoscopy/JFD-03K/ckpt/Xception/timm3/211102-1547/*_cnn_timm3_211102-154718.h5
    ## new version
    for dataname in ${data_set}
    do
      val_data=${root}/dataset/endoscopy/JFD-03K/jfd_dataset/${dataname}
      for model_path in ${model_set}
      do

        python jh_evaluate.py --gpu=${gpu} --scale  \
          --val_data=${val_data} \
          --save_dir=${root}/dataset/endoscopy/JFD-03K/grad_cam/ \
          --ckpt_path=${model_path} 
      done
    done
    # dataname='video_LEESANGSOON_10'
    # model_path=${root}/dataset/endoscopy/JFD-03K/ckpt/Xception/_prev/best_cnn_20210830-164755.h5
    # myvar=${model_path}
    # model_name=${myvar##*/}  # retain the part after the last slash
    # model_name=${model_name%.h5*}
    # python jh_evaluate.py --gpu=${gpu} --scale  \
    #   --val_data=${root}/dataset/endoscopy/JFD-03K/jfd_dataset/${dataname}/ \
    #   --save_dir=${root}/dataset/endoscopy/JFD-03K/grad_cam/${dataname}/${model_name} \
    #   --ckpt_path=${model_path} 
  




#########################
    echo
  elif [ ${gpu} -eq "3" ]
  then
    python jh_train.py --gpu=${gpu} \
      --train_path=${root}/dataset/endoscopy/JFD-03K/jfd_dataset/d1.16_1 \
      --save_path=${root}/dataset/endoscopy/JFD-03K/ckpt/Xception/ -cv
      # --val_path=${root}/dataset/endoscopy/JFD-03K/jfd_dataset/timm3/validation \





#########################
    echo
  elif [ ${gpu} -eq "4" ]
  then
  
  



############
############

#########################
    echo
  elif [ ${gpu} -eq "5" ]
  then
  


#########################
    echo
  elif [ ${gpu} -eq "6" ]
  then
  


#########################
    echo
  elif [ ${gpu} -eq "7" ]
  then



#########################
    echo
fi


### Train
    # python jh_train.py --gpu=${gpu} \
    # --train_path=${root}'/dataset/endoscopy/JFD-03K/jfd_dataset/timm3/train' \
    # --val_path=${root}'/dataset/endoscopy/JFD-03K/jfd_dataset/timm3/validation' 

    # python jh_train.py --gpu=${gpu} \
    #   --train_path=${root}/dataset/endoscopy/JFD-03K/jfd_dataset/timm3/train \
    #   --val_path=${root}/dataset/endoscopy/JFD-03K/jfd_dataset/timm3/validation \
    #   --save_path=${root}/dataset/endoscopy/JFD-03K/ckpt/Xception/ -cv

### Evaluation
    # dataname='d1.16'
    # model_path=${root}/dataset/endoscopy/JFD-03K/ckpt/Xception/_prev/best_cnn_20210830-164755.h5
    # myvar=${model_path}
    # model_name=${myvar##*/}  # retain the part after the last slash
    # model_name=${model_name%.h5*}
    # python jh_evaluate.py --gpu=${gpu} --scale  \
    #   --val_data=${root}/dataset/endoscopy/JFD-03K/jfd_dataset/${dataname}/ \
    #   --save_dir=${root}/dataset/endoscopy/JFD-03K/grad_cam/${dataname}/${model_name} \
    #   --ckpt_path=${model_path} 
     # # --cam  --plot

    ## new version
    # dataname='d1.16'
    # val_data=${root}/dataset/endoscopy/JFD-03K/jfd_dataset/${dataname}
    # model_path=${root}/dataset/endoscopy/JFD-03K/ckpt/Xception/timm3/211102_1622/timm3_211102_1622.h5
    # python jh_evaluate.py --gpu=${gpu} --scale  \
    #   --val_data=${val_data} \
    #   --save_dir=${root}/dataset/endoscopy/JFD-03K/grad_cam/ \
    #   --ckpt_path=${model_path} 
    #  # # --cam  --plot

### all dataset validation
    # model_set=${root}'/dataset/endoscopy/JFD-03K/ckpt/_prev/*/*.h5'
    # validation_set=${root}'/dataset/endoscopy/JFD-03K/jfd_dataset/*/validation/'
    # # dataname='v2.1'
    # # model='/dataset/endoscopy/JFD-03K/ckpt/xception_extractor/cv_d1.4_scale/best_cnn_20210314.h5 '
    
    # for model in ${model_set}
    # do
    #   myvar=${model}
    #   model_name=${myvar##*/}  # retain the part after the last slash
    #   model_name=${model_name%.h5*}
    # #   echo ${model_name}

    #   for validation in ${validation_set}
    #   do  
    #     myvar=${validation}
    #     dataname=${myvar%/validation*}
    #     dataname=${dataname##*/}
    #     # echo ${dataname}

    #     python jh_evaluate.py --gpu=${gpu} \
    #       --val_data=${validation} \
    #       --save_dir=${root}/dataset/endoscopy/JFD-03K/grad_cam/${dataname}/${model_name} \
    #       --ckpt_path=${model}
    #   done
    # done  

# Timm
# CUDA_VISIBLE_DEVICES=${gpu} python train.py /Data/jhshin/jfd_dataset/timm2/ --model xception --epochs 1000  --num-classes 26 --amp -j 40 --aa 'v0'

# -8월 24일-
# python jh_train.py -g ${gpu} --train_path '/Data/jhshin/jfd_dataset/timm3/train' --val_path '/Data/jhshin/jfd_dataset/timm3/validation'
# -8월 31일-
# python jh_evaluate.py -g ${gpu} \
  # --val_data '/Data/jhshin/jfd_dataset/d1.16/' \
  # --ckpt_path  '/Data/jhshin/ckpt/Xception/best_cnn_20210830-164755.h5' \
  # --save_dir '/Data/jhshin/incorrects/210831/' \
  # --cam --scale --plot

    