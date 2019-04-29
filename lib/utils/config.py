import numpy as np

class ConfigSiamFC:

    """
    sampling related
    """
    VID_used = True                        # whether to use VID dataset
    GOT10K_used = False                    # whether to use GOT10K dataset
    YTB_used = False                       # whether to use YTB dataset
    frame_range_vid = 100                  # frame range of choosing the instance
    frame_range_got10k = 100               # frame range of choosing the instance
    frame_range_ytb = 1                    # frame range of choosing the instance
    exemplar_size = 127                    # exemplar size
    instance_size = 255                    # instance size
    context_amount = 0.5                   # context amount
    total_stride = 8                       # total stride of backbone
    score_size = 17                        # response size
    seed = 1234                            # seed to sample training videos
    sample_type = 'uniform'                # sample strategy of VID dataset
    gray_ratio = 0.25
    blur_ratio = 0.15

    """
    training related
    """
    load_imagenet = True                   # whether to use the imagenet pretrained model
    fix_former_layers = True               # whether to fix the first layers
    epoch = 50                             # total epoch for training
    lr = 1e-2                              # learning rate of SGD
    momentum = 0.9                         # momentum of SGD
    weight_decay = 1e-4                    # weight decay of optimizator
    # step_size = 25                       # step size of LR_Schedular
    # gamma = 0.1                          # decay rate of LR_Schedular
    gamma = 0.8709635899560806             # decay rate of EP_Schedular
    num_per_epoch = 53200                  # num of samples per epoch
    train_ratio = 0.9                      # training ratio of VID dataset
    max_translate = 3                      # max translation of random shift
    train_batch_size = 16                  # training batch size
    valid_batch_size = 16                  # validation batch size
    train_num_workers = 8                  # number of workers of train dataloader
    valid_num_workers = 8                  # number of workers of validation dataloader
    log_dir = './models/logs'              # log dirs
    data_dir = 'dataset/data_curated'      # VID trian dir
    save_interval = 1                      # model save interval
    show_interval = 100                    # info print interval

    radius = 16                            # radius of positive label
    response_scale = 1e-3                  # normalize of response, adjust the scale of response


    """
    tracking related
    """
    num_scale = 3                          # number of scales
    scale_step = 1.0375                    # scale step of instance image
    scale_penalty = 0.9745                 # scale penalty
    scale_lr = 0.59                        # scale learning rate
    response_up = 16                       # response upsample stride

    windowing = 'cosine'
    window_influence = 0.350               # window influence #0.176

    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1

class ConfigSiamRPN:

    """
    sampling related
    """
    VID_used = True                        # whether to use VID dataset
    GOT10K_used = False                    # whether to use GOT10K dataset
    YTB_used = True                        # whether to use YTB dataset
    frame_range_vid = 100                  # frame range of choosing the instance
    frame_range_got10k = 100               # frame range of choosing the instance
    frame_range_ytb = 1                    # frame range of choosing the instance
    exemplar_size = 127                    # exemplar size
    instance_size = 271                    # instance size
    context_amount = 0.5                   # context amount
    total_stride = 8                       # total stride of backbone
    score_size = int((instance_size - exemplar_size) / 8 + 1)
    seed = 6666                            # seed to sample training videos
    sample_type = 'uniform'                # sample strategy of VID dataset
    gray_ratio = 0.25
    blur_ratio = 0.15

    """
    training related
    """
    load_imagenet = True                   # whether to use the imagenet pretrained model
    fix_former_layers = True               # whether to fix the first layers
    epoch = 50                             # total epoch for training
    start_lr = 3e-2                        # start learning rate
    end_lr = 1e-5                          # end learning rate
    lr = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
    gamma = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[1] / \
            np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
                                           # decay rate of LR_Schedular
    step_size = 1                          # step size of LR_Schedular
    momentum = 0.9                         # momentum of SGD
    weight_decay = 0.0005                  # weight decay of optimizator
    clip = 10  # grad clip
    pairs_per_video_per_epoch = 12         # pairs per video(12 * 4417 = 53000)
    train_ratio = 0.99                     # training ratio of VID dataset
    max_translate = 12                     # max translation of random shift
    train_batch_size = 32                  # training batch size
    valid_batch_size = 16                  # validation batch size
    train_num_workers = 8                  # number of workers of train dataloader
    valid_num_workers = 8                  # number of workers of validation dataloader
    log_dir = './data/logs'                # log dirs
    data_dir = 'dataset/data_curated'      # VID train dir
    save_interval = 1                      # model save interval
    show_interval = 100                    # info print interval


    ohem_pos = False
    ohem_neg = False
    ohem_reg = False
    scale_range = (0.001, 0.7)
    ratio_range = (0.1, 10)
    scale_resize = 0.15                    # scale step of instance image
    valid_scope = int((instance_size - exemplar_size) / total_stride / 2)
    anchor_scales = np.array([8, ])
    anchor_ratios = np.array([0.33, 0.5, 1, 2, 3])
    anchor_num = len(anchor_scales) * len(anchor_ratios)
    anchor_base_size = 8
    pos_threshold = 0.6
    neg_threshold = 0.3
    num_pos = 16
    num_neg = 48
    lamb = 5
    show_topK = 3

    """
    tracking related
    """
    penalty_k = 0.22
    lr_box = 0.30
    min_scale = 0.1
    max_scale = 10

    windowing = 'cosine'
    window_influence = 0.40

    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1

configSiamFC = ConfigSiamFC()
configSiamRPN = ConfigSiamRPN()