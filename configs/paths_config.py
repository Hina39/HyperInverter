dataset_paths = {
    #  Human Face (FFHQ - train , CelebA-HQ - test)
    "ffhq": "/home/challenger/hyperinverter/data/69000-20231115T153103Z-001/69000",
    #今ないので、とりあえず、ffhqを使う
    "celeba_test": "/home/challenger/hyperinverter/data/drive-download-20231115T154027Z-002",
    # Churches (LSUN Churches)
    "church_train": "",
    "church_test": "",
}

model_paths = {
    # ---------------------------------------------------------------------------#
    # W Encoder
    "w_encoder_ffhq": "/home/challenger/hyperinverter/pretrained_models/w_encoder_psp_ffhq_encode.pt",
    "w_encoder_church": "",
    # Our HyperInverter models
    "hyper_inverter_ffhq": "/home/challenger/hyperinverter/pretrained_models/hyper_inverter_psp_ffhq_encode_large.pt",
    "hyper_inverter_church": "",
    # ---------------------------------------------------------------------------#
    # Official pre-trained models from other SOTA methods
    # ===> Human Faces
    "official_psp_ffhq": "/home/challenger/hyperinverter/pretrained_models/w_encoder_psp_ffhq_encode.pt",
    "official_e4e_ffhq": "/home/challenger/hyperinverter/pretrained_models/w_encoder_e4e_ffhq_encode.pt",
    "official_restyle_e4e_ffhq": "",
    # ===> Church
    "official_psp_church": "",
    "official_e4e_church": "",
    "official_restyle_e4e_church": "",
    # ---------------------------------------------------------------------------#
    # StyleGAN2 pretrained weights
    "stylegan2_ada_ffhq": "pretrained_models/stylegan2-ffhq-config-f.pkl",
    "stylegan2_ada_church": "pretrained_models/stylegan2-church-config-f.pkl",
    # ---------------------------------------------------------------------------#
    # Auxiliary pretrained models
    "ir_se50": "pretrained_models/model_ir_se50.pth",
    "circular_face": "pretrained_models/CurricularFace_Backbone.pth",
    "mtcnn_pnet": "pretrained_models/mtcnn/pnet.npy",
    "mtcnn_rnet": "pretrained_models/mtcnn/rnet.npy",
    "mtcnn_onet": "pretrained_models/mtcnn/onet.npy",
    "shape_predictor": "pretrained_models/shape_predictor_68_face_landmarks.dat",
    "moco": "pretrained_models/moco_v2_800ep_pretrain.pt",
    "resnet34": "pretrained_models/resnet34-333f7ec4.pth",
}


editing_paths = {
    "interfacegan_age": "editings/interfacegan_directions/age.pt",
    "interfacegan_smile": "editings/interfacegan_directions/smile.pt",
    "interfacegan_rotation": "editings/interfacegan_directions/rotation.pt",
    "ffhq_pca": "editings/ganspace_pca/ffhq_pca.pt",
    "church_pca": "editings/ganspace_pca/church_pca.pt",
}

styleclip_paths = {
    "style_clip_pretrained_mappers": "pretrained_models/styleclip_mappers",
}
