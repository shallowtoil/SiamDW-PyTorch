
class Config:
    # dataset related
    exemplar_size = 127                    # exemplar size
    instance_size = 255                    # instance size
    context_amount = 0.5                   # context amount

    # training related
    num_per_epoch = 53200                # num of samples per epoch
    train_ratio = 0.9                      # training ratio of VID dataset
    frame_range = 100                      # frame range of choosing the instance
    train_batch_size = 8                  # training batch size
    valid_batch_size = 8                  # validation batch size
    train_num_workers = 8                  # number of workers of train dataloader
    valid_num_workers = 8                  # number of workers of validation dataloader
    lr = 1e-2                              # learning rate of SGD
    momentum = 0.9                         # momentum of SGD
    weight_decay = 1e-4                    # weight decay of optimizator
    # step_size = 25                        # step size of LR_Schedular
    # gamma = 0.1                           # decay rate of LR_Schedular
    gamma = 0.8709635899560806             # decay rate of EP_Schedular
    epoch = 50                             # total epoch
    seed = 1234                            # seed to sample training videos
    log_dir = './models/logs'              # log dirs
    data_dir = 'dataset/data_curated'      # VID trian dir
    radius = 16                            # radius of positive label
    response_scale = 1e-3                  # normalize of response, adjust the scale of response
    max_translate = 3                      # max translation of random shift

    # tracking related
    scale_step = 1.0375                    # scale step of instance image
    num_scale = 3                          # number of scales
    scale_lr = 0.59                        # scale learning rate
    response_up = 16                       # response upsample stride
    score_size = 17                       # response size
    train_score_size = 15                 # train response size
    window_influence = 0.350               # window influence #0.176
    scale_penalty = 0.9745                 # scale penalty
    total_stride = 8                       # total stride of backbone
    windowing = 'cosine'
    sample_type = 'uniform'                # sample strategy of VID dataset
    gray_ratio = 0.25
    blur_ratio = 0.15

    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1

config = Config()
