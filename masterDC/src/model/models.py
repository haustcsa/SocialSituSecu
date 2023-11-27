def create_model(cfg):

    if cfg.model == "DCGAN":
        # from .DCGAN import DCGAN
        print('DCGAN')
        model = 'DCAGN'
    if cfg.model == "GAN":

        print('GAN')
        model = 'GAN'

    return model
