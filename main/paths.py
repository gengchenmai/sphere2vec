def get_paths(variable_name):
    paths = {
    'mask_dir': '../data/',
    'inat_2017_data_dir': '../geo_prior_data/data/inat_2017/',
    'inat_2018_data_dir': '../geo_prior_data/data/inat_2018/',
    'inat_2018_img_dir': '../geo_prior_data/data/inat_2018/',
    'birdsnap_data_dir': '../geo_prior_data/data/birdsnap/',
    'nabirds_data_dir': '../geo_prior_data/data/nabirds/',
    'yfcc_data_dir': '../geo_prior_data/data/yfcc/',
    'syntconsband_data_dir': '../geo_prior_data/data/syntconsband/',
    'syntvarband_data_dir': '../geo_prior_data/data/syntvarband/',
    'fmow_data_dir': '../geo_prior_data/data/fmow/',
    'vmf_data_dir': '../geo_prior_data/data/vmf/',
    # 'vmfC50S100L1H32_data_dir': '../geo_prior_data/data/vmfC50S100L1H32/',
    }
    return paths[variable_name]
