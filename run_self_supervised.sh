
##################  3D UNet   #######################
# python basicsr/train.py -opt options/in_vivo_brain/in_vivo_brain_UNet3D_self_lines_train_bi_scan.yml

# python basicsr/train.py -opt options/in_vivo_brain/in_vivo_brain_UNet3D_self_lines_train.yml
 python basicsr/test.py -opt options/in_vivo_brain/in_vivo_brain_UNet3D_self_lines_test.yml


# python basicsr/train.py -opt options/in_vivo_brain/in_vivo_brain_UNet_self_frames_train.yml
# python basicsr/test.py -opt options/in_vivo_brain/in_vivo_brain_UNet_self_frames_test.yml


##################  Synthetic_Ca   #######################

# python basicsr/train.py -opt options/in_vivo_brain/Synthetic_Ca_UNet3D_self_lines_train.yml

# python basicsr/train.py -opt options/in_vivo_brain/Synthetic_Ca_UNet_self_frames_train.yml

# python basicsr/test.py -opt options/in_vivo_brain/Synthetic_Ca_UNet3D_test.yml




##################  Res 3D   #######################

# python basicsr/train.py -opt options/in_vivo_brain/in_vivo_brain_Res3D_self_frames_train.yml

##################  3D RCAN   #######################

# python basicsr/train.py -opt options/in_vivo_brain/in_vivo_brain_RCAN3D_self_lines_train.yml
# python basicsr/test.py -opt options/in_vivo_brain/in_vivo_brain_RCAN3D_self_lines_test.yml


# python basicsr/train.py -opt options/in_vivo_brain/in_vivo_brain_RCAN_self_frames_train.yml
# python basicsr/test.py -opt options/in_vivo_brain/in_vivo_brain_RCAN_self_frames_test.yml