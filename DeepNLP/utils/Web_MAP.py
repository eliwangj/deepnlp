from __future__ import (absolute_import, division, print_function, unicode_literals)

NER_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-middle-uncased': {'WN16': ['http://106.75.176.107:8088/DNER/saved_models/en_NER_BERT_WN16_md_0.1.0/model/', 'en_NER_BERT_WN16_md_0.1.0']},
    'bert-base-uncased': {'WN16': ['http://106.75.176.107:8088/DNER/saved_models/en_NER_BERT_WN16_bs_0.1.0/model/', 'en_NER_BERT_WN16_bs_0.1.0']},
    'bert-large-uncased': {'WN16': ['http://106.75.176.107:8088/DNER/saved_models/en_NER_BERT_WN16_ls_0.1.0/model/', 'en_NER_BERT_WN16_ls_0.1.0']},
    'bert-middle-chinese': {
        'chemed': ['http://106.75.176.107:8088/DNER/saved_models/chemed_NER_BERT_md_0.1.0/model/', 'chemed_NER_BERT_md_0.1.0'],
        'RE': ['http://106.75.176.107:8088/DNER/saved_models/zh_NER_BERT_RE_md_0.1.0/model/', 'zh_NER_BERT_RE_md_0.1.0'],
    },
    'bert-base-chinese': {
        'chemed': ['http://106.75.176.107:8088/DNER/saved_models/chemed_NER_BERT_bs_0.1.0/model/', 'chemed_NER_BERT_bs_0.1.0'],
        'RE': ['http://106.75.176.107:8088/DNER/saved_models/zh_NER_BERT_RE_bs_0.1.0/model/', 'zh_NER_BERT_RE_bs_0.1.0']
    },
    'BiLSTM': {
        'chemed': ['http://106.75.176.107:8088/DNER/saved_models/chemed_NER_BiLSTM_sm_0.1.0/model/', 'chemed_NER_BiLSTM_CTB5_sm_0.1.0'],
        'RE': ['http://106.75.176.107:8088/DNER/saved_models/zh_NER_BiLSTM_RE_sm_0.1.0/model/', 'zh_NER_BiLSTM_RE_sm_0.1.0'],
        'WN16': ['http://106.75.176.107:8088/DNER/saved_models/en_NER_BiLSTM_WN16_sm_0.1.0/model/', 'en_NER_BiLSTM_WN16_sm_0.1.0']
    }
}

Par_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-middle-uncased': {
        'en_Biaffine': ['http://106.75.176.107:8088/DPar/saved_models/en_DepPar_BERT_Biaffine_UD(EWT)_md_0.1.0/model/',
                     'en_DepPar_BERT_Biaffine_UD(EWT)_md_0.1.0'],
        'en_Base': ['http://106.75.176.107:8088/DPar/saved_models/en_DepPar_BERT_UD(EWT)_md_0.1.0/model/',
                 'en_DepPar_BERT_UD(EWT)_md_0.1.0']
    },
    'bert-base-uncased': {
        'en_Biaffine':  ['http://106.75.176.107:8088/DPar/saved_models/en_DepPar_BERT_Biaffine_UD(EWT)_bs_0.1.0/model/', 'en_DepPar_BERT_Biaffine_UD(EWT)_bs_0.1.0'],
        'en_Base': ['http://106.75.176.107:8088/DPar/saved_models/en_DepPar_BERT_UD(EWT)_bs_0.1.0/model/', 'en_DepPar_BERT_UD(EWT)_bs_0.1.0']
    },
    'bert-large-uncased': {
        'en_Biaffine':  ['http://106.75.176.107:8088/DPar/saved_models/en_DepPar_BERT_Biaffine_UD(EWT)_ls_0.1.0/model/', 'en_DepPar_BERT_Biaffine_UD(EWT)_ls_0.1.0'],
        'en_Base': ['http://106.75.176.107:8088/DPar/saved_models/en_DepPar_BERT_UD(EWT)_ls_0.1.0/model/', 'en_DepPar_BERT_UD(EWT)_ls_0.1.0']
    },
    'bert-middle-chinese': {
        'zh_Biaffine': ['http://106.75.176.107:8088/DPar/saved_models/zh_DepPar_BERT_Biaffine_UD(GSDSimp)_md_0.1.0/model/',
                     'zh_DepPar_BERT_Biaffine_UD(GSDSimp)_md_0.1.0'],
        'zh_Base': ['http://106.75.176.107:8088/DPar/saved_models/zh_DepPar_BERT_UD(GSDSimp)_md_0.1.0/model/',
                 'zh_DepPar_BERT_UD(GSDSimp)_md_0.1.0']
    },
    'bert-base-chinese': {
        'zh_Biaffine': ['http://106.75.176.107:8088/DPar/saved_models/zh_DepPar_BERT_Biaffine_UD(GSDSimp)_bs_0.1.0/model/',
                     'zh_DepPar_BERT_Biaffine_UD(GSDSimp)_bs_0.1.0'],
        'zh_Base': ['http://106.75.176.107:8088/DPar/saved_models/zh_DepPar_BERT_UD(GSDSimp)_bs_0.1.0/model/',
                 'zh_DepPar_BERT_UD(GSDSimp)_bs_0.1.0']
    },
    'BiLSTM': {
        'zh_Biaffine': ['http://106.75.176.107:8088/DPar/saved_models/zh_DepPar_BiLSTM_Biaffine_UD(GSDSimp)_sm_0.1.0/model/',
                     'zh_DepPar_BiLSTM_Biaffine_UD(GSDSimp)_sm_0.1.0'],
        'zh_Base': ['http://106.75.176.107:8088/DPar/saved_models/zh_DepPar_BiLSTM_UD(GSDSimp)_sm_0.1.0/model/',
                 'zh_DepPar_BiLSTM_UD(GSDSimp)_sm_0.1.0'],
        'en_Biaffine': ['http://106.75.176.107:8088/DPar/saved_models/en_DepPar_BiLSTM_Biaffine_UD(EWT)_sm_0.1.0/model/',
                     'en_DepPar_BiLSTM_Biaffine_UD(EWT)_sm_0.1.0'],
        'en_Base': ['http://106.75.176.107:8088/DPar/saved_models/en_DepPar_BiLSTM_UD(EWT)_sm_0.1.0/model/',
                 'en_DepPar_BiLSTM_UD(EWT)_sm_0.1.0']
    }
}

POS_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-middle-uncased': {'en_POS': ['http://106.75.176.107:8088/DPar/saved_models/en_POS_BERT_md_0.1.0/model/',
                             'en_POS_BERT_md_0.1.0']},
    'bert-base-uncased': {'en_POS': ['http://106.75.176.107:8088/DPar/saved_models/en_POS_BERT_bs_0.1.0/model/',
                             'en_POS_BERT_bs_0.1.0']},
    'bert-large-uncased': {'en_POS': ['http://106.75.176.107:8088/DPar/saved_models/en_POS_BERT_ls_0.1.0/model/',
                           'en_POS_BERT_ls_0.1.0']},
    'bert-middle-chinese': {
        'zh_POS': ['http://106.75.176.107:8088/DPar/saved_models/zh_POS_BERT_CTB5_md_0.1.0/model/',
                     'zh_POS_BERT_CTB5_md_0.1.0'],
        'zh_SP_McASP': ['http://106.75.176.107:8088/DPar/saved_models/zh_SP_BERT_McASP_CTB5_md_0.1.0/model/',
                     'zh_SP_BERT_McASP_CTB5_md_0.1.0'],
        'zh_SP': ['http://106.75.176.107:8088/DPar/saved_models/zh_SP_BERT_CTB5_md_0.1.0/model/',
                 'zh_SP_BERT_CTB5_md_0.1.0']
    },
    'bert-base-chinese': {
        'zh_POS': ['http://106.75.176.107:8088/DPar/saved_models/zh_POS_BERT_CTB5_bs_0.1.0/model/',
                     'zh_POS_BERT_CTB5_bs_0.1.0'],
        'zh_SP_McASP': ['http://106.75.176.107:8088/DPar/saved_models/zh_SP_BERT_McASP_CTB5_bs_0.1.0/model/',
                     'zh_SP_BERT_McASP_CTB5_bs_0.1.0'],
        'zh_SP': ['http://106.75.176.107:8088/DPar/saved_models/zh_SP_BERT_CTB5_bs_0.1.0/model/',
                 'zh_SP_BERT_CTB5_bs_0.1.0']
    },
    'BiLSTM': {
        'en_POS': ['http://106.75.176.107:8088/DPar/saved_models/en_POS_BiLSTM_sm_0.1.0/model/',
                     'en_POS_BiLSTM_sm_0.1.0'],
        'zh_POS': ['http://106.75.176.107:8088/DPar/saved_models/zh_POS_BiLSTM_CTB5_sm_0.1.0/model/',
                     'zh_POS_BiLSTM_CTB5_sm_0.1.0'],
        'zh_SP': ['http://106.75.176.107:8088/DPar/saved_models/zh_SP_BiLSTM_CTB5_sm_0.1.0/model/',
                 'zh_SP_BiLSTM_CTB5_sm_0.1.0']
    }
}

Rel_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-middle-uncased': {
        'ace2005en': ['http://106.75.176.107:8088/DPar/saved_models/en_Rel_BERT_ace2005en_md_0.1.0/model/',
                     'en_Rel_BERT_ace2005en_md_0.1.0'],
        'mevalse': ['http://106.75.176.107:8088/DPar/saved_models/zh_Rel_BERT_mevalse_bs_0.1.0/model/',
                     'zh_Rel_BERT_mevalse_md_0.1.0']
    },
    'bert-base-uncased': {
        'ace2005en': ['http://106.75.176.107:8088/DPar/saved_models/en_Rel_BERT_ace2005en_bs_0.1.0/model/',
                   'en_Rel_BERT_ace2005en_bs_0.1.0'],
        'mevalse': ['http://106.75.176.107:8088/DPar/saved_models/en_Rel_BERT_mevalse_bs_0.1.0/model/',
                   'en_Rel_BERT_mevalse_bs_0.1.0']
    },
    'bert-large-uncased': {
        'ace2005en': ['http://106.75.176.107:8088/DPar/saved_models/en_Rel_BERT_ace2005en_ls_0.1.0/model/',
                   'en_Rel_BERT_ace2005en_ls_0.1.0'],
        'mevalse': ['http://106.75.176.107:8088/DPar/saved_models/en_Rel_BERT_mevalse_ls_0.1.0/model/',
                   'en_Rel_BERT_mevalse_ls_0.1.0']
    },
    'BiLSTM': {
        'ace2005en': ['http://106.75.176.107:8088/DPar/saved_models/en_Rel_BiLSTM_ace2005en_sm_0.1.0/model/',
                     'en_Rel_BiLSTM_ace2005en_sm_0.1.0'],
        'mevalse': ['http://106.75.176.107:8088/DPar/saved_models/zh_Rel_BiLSTM_mevalse_sm_0.1.0/model/',
                     'zh_Rel_BiLSTM_mevalse_sm_0.1.0']
    }
}

Seg_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-middle-uncased': {
        'chemed_KVMN': ['http://106.75.176.107:8088/DSeg/saved_models/chemed_Seg_BERT_KVMN_md_0.1.0/model/',
                     'chemed_Seg_BERT_KVMN_md_0.1.0'],
        'chemed': ['http://106.75.176.107:8088/DSeg/saved_models/chemed_Seg_BERT_md_0.1.0/model/',
                   'chemed_Seg_BERT_md_0.1.0'],
        'Base_KVMN': ['http://106.75.176.107:8088/DSeg/saved_models/zh_Seg_BERT_CTB5_KVMN_md_0.1.0/model/',
                 'zh_Seg_BERT_CTB5_KVMN_md_0.1.0'],
        'Base': ['http://106.75.176.107:8088/DSeg/saved_models/zh_Seg_BERT_CTB5_md_0.1.0/model/',
                   'zh_Seg_BERT_CTB5_md_0.1.0'],
    },
    'bert-base-uncased': {
        'chemed_KVMN': ['http://106.75.176.107:8088/DSeg/saved_models/chemed_Seg_BERT_KVMN_bs_0.1.0/model/',
                        'chemed_Seg_BERT_KVMN_bs_0.1.0'],
        'chemed': ['http://106.75.176.107:8088/DSeg/saved_models/chemed_Seg_BERT_bs_0.1.0/model/',
                   'chemed_Seg_BERT_bs_0.1.0'],
        'Base_KVMN': ['http://106.75.176.107:8088/DSeg/saved_models/zh_Seg_BERT_CTB5_KVMN_sm_0.1.0/model/',
                      'zh_Seg_BERT_CTB5_KVMN_sm_0.1.0'],
        'Base': ['http://106.75.176.107:8088/DSeg/saved_models/zh_Seg_BERT_CTB5_sm_0.1.0/model/',
                   'zh_Seg_BERT_CTB5_sm_0.1.0'],
    },
    'BiLSTM': {
        'chemed_KVMN': ['http://106.75.176.107:8088/DSeg/saved_models/chemed_Seg_BiLSTM_KVMN_sm_0.1.0/model/',
                     'chemed_Seg_BiLSTM_KVMN_sm_0.1.0'],
        'chemed': ['http://106.75.176.107:8088/DSeg/saved_models/chemed_Seg_BiLSTM_KVMN_sm_0.1.0/model/',
                   'chemed_Seg_BiLSTM_KVMN_sm_0.1.0'],
        'Base_KVMN': ['http://106.75.176.107:8088/DSeg/saved_models/zh_Seg_BiLSTM_CTB5_KVMN_sm_0.1.0/model/',
                 'zh_Seg_BiLSTM_CTB5_KVMN_sm_0.1.0'],
        'Base': ['http://106.75.176.107:8088/DSeg/saved_models/zh_Seg_BiLSTM_CTB5_KVMN_sm_0.1.0/model/',
                      'zh_Seg_BiLSTM_CTB5_KVMN_sm_0.1.0']
    }
}


Snt_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'SA': {
        'bert-middle-uncased': {'SST5': ['http://106.75.176.107:8088/DSnt/saved_models/en_SA_BERT_SST5_md_0.1.0/model/',
                                'en_SA_BERT_SST5_md_0.1.0']},
        'bert-base-uncased': {'SST5': ['http://106.75.176.107:8088/DSnt/saved_models/en_SA_BERT_SST5_bs_0.1.0/model/',
                              'en_SA_BERT_SST5_bs_0.1.0']},
        'bert-large-uncased': {'SST5': ['http://106.75.176.107:8088/DSnt/saved_models/en_SA_BERT_SST5_ls_0.1.0/model/',
                               'en_SA_BERT_SST5_ls_0.1.0']},
        'bert-middle-chinese': {'chnsenticorp': ['http://106.75.176.107:8088/DSnt/saved_models/zh_SA_BERT_chnsenticorp_md_0.1.0/model/',
                               'zh_SA_BERT_chnsenticorp_md_0.1.0']},
        'bert-base-chinese': {'chnsenticorp': ['http://106.75.176.107:8088/DSnt/saved_models/zh_SA_BERT_chnsenticorp_bs_0.1.0/model/',
                               'zh_SA_BERT_chnsenticorp_bs_0.1.0']},
        'BiLSTM': {
            'chnsenticorp': ['http://106.75.176.107:8088/DSnt/saved_models/zh_SA_BiLSTM_chnsenticorp_sm_0.1.0/model/',
                               'zh_SA_BiLSTM_chnsenticorp_sm_0.1.0'],
            'SST5': ['http://106.75.176.107:8088/DSnt/saved_models/en_SA_BiLSTM_SST5_sm_0.1.0/model/',
                               'en_SA_BiLSTM_SST5_sm_0.1.0'],
        }
    },
    'ABSA': {
        'bert-middle-uncased': {
            'twitter': ['http://106.75.176.107:8088/DSeg/en_ABSA_BERT_twitter_md_0.1.0/model/',
                        'en_ABSA_BERT_twitter_md_0.1.0'],
            'laptop': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_laptop_md_0.1.0/model/',
                       'en_ABSA_BERT_laptop_md_0.1.0'],
            'rest14': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_rest14_md_0.1.0/model/',
                       'en_ABSA_BERT_rest14_md_0.1.0'],
            'rest15': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_rest15_md_0.1.0/model/',
                       'en_ABSA_BERT_rest15_md_0.1.0'],
            'rest16': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_rest16_md_0.1.0/model/',
                       'en_ABSA_BERT_rest16_md_0.1.0'],
            'MAMS': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_MAMS_md_0.1.0/model/',
                     'en_ABSA_BERT_MAMS_md_0.1.0']
        },
        'bert-base-uncased': {
            'twitter': ['http://106.75.176.107:8088/DSeg/en_ABSA_BERT_twitter_bs_0.1.0/model/',
                        'en_ABSA_BERT_twitter_bs_0.1.0'],
            'laptop': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_laptop_bs_0.1.0/model/',
                       'en_ABSA_BERT_laptop_bs_0.1.0'],
            'rest14': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_rest14_bs_0.1.0/model/',
                       'en_ABSA_BERT_rest14_bs_0.1.0'],
            'rest15': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_rest15_bs_0.1.0/model/',
                       'en_ABSA_BERT_rest15_bs_0.1.0'],
            'rest16': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_rest16_bs_0.1.0/model/',
                       'en_ABSA_BERT_rest16_bs_0.1.0'],
            'MAMS': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_MAMS_bs_0.1.0/model/',
                     'en_ABSA_BERT_MAMS_bs_0.1.0']
        },
        'bert-large-uncased': {
            'twitter': ['http://106.75.176.107:8088/DSeg/en_ABSA_BERT_twitter_ls_0.1.0/model/',
                        'en_ABSA_BERT_twitter_ls_0.1.0'],
            'laptop': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_laptop_ls_0.1.0/model/',
                       'en_ABSA_BERT_laptop_ls_0.1.0'],
            'rest14': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_rest14_ls_0.1.0/model/',
                       'en_ABSA_BERT_rest14_ls_0.1.0'],
            'rest15': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_rest15_ls_0.1.0/model/',
                       'en_ABSA_BERT_rest15_ls_0.1.0'],
            'rest16': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_rest16_ls_0.1.0/model/',
                       'en_ABSA_BERT_rest16_ls_0.1.0'],
            'MAMS': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_MAMS_ls_0.1.0/model/',
                     'en_ABSA_BERT_MAMS_ls_0.1.0']
        },
        'BiLSTM': {
            'twitter': ['http://106.75.176.107:8088/DSeg/en_ABSA_BiLSTM_twitter_sm_0.1.0/model/',
                        'en_ABSA_BiLSTM_twitter_sm_0.1.0'],
            'laptop': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BiLSTM_laptop_sm_0.1.0/model/',
                       'en_ABSA_BiLSTM_laptop_sm_0.1.0'],
            'rest14': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BiLSTM_rest14_sm_0.1.0/model/',
                       'en_ABSA_BiLSTM_rest14_sm_0.1.0'],
            'rest15': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BiLSTM_rest15_sm_0.1.0/model/',
                       'en_ABSA_BiLSTM_rest15_sm_0.1.0'],
            'rest16': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BiLSTM_rest16_sm_0.1.0/model/',
                       'en_ABSA_BiLSTM_rest16_sm_0.1.0'],
            'MAMS': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BiLSTM_MAMS_sm_0.1.0/model/',
                     'en_ABSA_BiLSTM_MAMS_sm_0.1.0']
        },
    },
    'ABSA_tgcn': {
        'bert-middle-uncased': {
            'twitter': ['http://106.75.176.107:8088/DSeg/en_ABSA_BERT_tgcn_twitter_md_0.1.0/model/',
                        'en_ABSA_BERT_tgcn_twitter_md_0.1.0'],
            'laptop': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_laptop_md_0.1.0/model/',
                       'en_ABSA_BERT_tgcn_laptop_md_0.1.0'],
            'rest14': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_rest14_md_0.1.0/model/',
                       'en_ABSA_BERT_tgcn_rest14_md_0.1.0'],
            'rest15': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_rest15_md_0.1.0/model/',
                       'en_ABSA_BERT_tgcn_rest15_md_0.1.0'],
            'rest16': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_rest16_md_0.1.0/model/',
                       'en_ABSA_BERT_tgcn_rest16_md_0.1.0'],
            'MAMS': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_MAMS_md_0.1.0/model/',
                     'en_ABSA_BERT_tgcn_MAMS_md_0.1.0']
        },
        'bert-base-uncased': {
            'twitter': ['http://106.75.176.107:8088/DSeg/en_ABSA_BERT_tgcn_twitter_bs_0.1.0/model/',
                        'en_ABSA_BERT_tgcn_twitter_bs_0.1.0'],
            'laptop': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_laptop_bs_0.1.0/model/',
                       'en_ABSA_BERT_tgcn_laptop_bs_0.1.0'],
            'rest14': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_rest14_bs_0.1.0/model/',
                       'en_ABSA_BERT_tgcn_rest14_bs_0.1.0'],
            'rest15': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_rest15_bs_0.1.0/model/',
                       'en_ABSA_BERT_tgcn_rest15_bs_0.1.0'],
            'rest16': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_rest16_bs_0.1.0/model/',
                       'en_ABSA_BERT_tgcn_rest16_bs_0.1.0'],
            'MAMS': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_MAMS_bs_0.1.0/model/',
                     'en_ABSA_BERT_tgcn_MAMS_bs_0.1.0']
        },
        'bert-large-uncased': {
            'twitter': ['http://106.75.176.107:8088/DSeg/en_ABSA_BERT_tgcn_twitter_ls_0.1.0/model/',
                        'en_ABSA_BERT_tgcn_twitter_ls_0.1.0'],
            'laptop': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_laptop_ls_0.1.0/model/',
                       'en_ABSA_BERT_tgcn_laptop_ls_0.1.0'],
            'rest14': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_rest14_ls_0.1.0/model/',
                       'en_ABSA_BERT_tgcn_rest14_ls_0.1.0'],
            'rest15': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_rest15_ls_0.1.0/model/',
                       'en_ABSA_BERT_tgcn_rest15_ls_0.1.0'],
            'rest16': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_rest16_ls_0.1.0/model/',
                       'en_ABSA_BERT_tgcn_rest16_ls_0.1.0'],
            'MAMS': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BERT_tgcn_MAMS_ls_0.1.0/model/',
                     'en_ABSA_BERT_tgcn_MAMS_ls_0.1.0']
        },
        'BiLSTM': {
            'twitter': ['http://106.75.176.107:8088/DSeg/en_ABSA_BiLSTM_tgcn_twitter_sm_0.1.0/model/',
                        'en_ABSA_BiLSTM_tgcn_twitter_sm_0.1.0'],
            'laptop': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BiLSTM_tgcn_laptop_sm_0.1.0/model/',
                       'en_ABSA_BiLSTM_tgcn_laptop_sm_0.1.0'],
            'rest14': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BiLSTM_tgcn_rest14_sm_0.1.0/model/',
                       'en_ABSA_BiLSTM_tgcn_rest14_sm_0.1.0'],
            'rest15': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BiLSTM_tgcn_rest15_sm_0.1.0/model/',
                       'en_ABSA_BiLSTM_tgcn_rest15_sm_0.1.0'],
            'rest16': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BiLSTM_tgcn_rest16_sm_0.1.0/model/',
                       'en_ABSA_BiLSTM_tgcn_rest16_sm_0.1.0'],
            'MAMS': ['http://106.75.176.107:8088/DSeg/saved_models/en_ABSA_BiLSTM_tgcn_MAMS_sm_0.1.0/model/',
                     'en_ABSA_BiLSTM_tgcn_MAMS_sm_0.1.0']
        },
    },
}


SRL_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-middle-uncased': {'CoNLL05': ['http://106.75.176.107:8088/DSRL/saved_models/en_SRL_BERT_CoNLL05_md_0.1.0/model/',
                             'en_SRL_BERT_CoNLL05_md_0.1.0'],
                            'CoNLL12': ['http://106.75.176.107:8088/DSRL/saved_models/en_SRL_BERT_CoNLL12_md_0.1.0/model/',
                             'en_SRL_BERT_CoNLL12_md_0.1.0']},
    'bert-base-uncased': {'CoNLL05': ['http://106.75.176.107:8088/DSRL/saved_models/en_SRL_BERT_CoNLL05_bs_0.1.0/model/',
                             'en_SRL_BERT_CoNLL05_bs_0.1.0'],
                            'CoNLL12': ['http://106.75.176.107:8088/DSRL/saved_models/en_SRL_BERT_CoNLL12_bs_0.1.0/model/',
                             'en_SRL_BERT_CoNLL12_bs_0.1.0']},
    'bert-large-uncased': {'CoNLL05': ['http://106.75.176.107:8088/DSRL/saved_models/en_SRL_BERT_CoNLL05_ls_0.1.0/model/',
                             'en_SRL_BERT_CoNLL05_ls_0.1.0'],
                            'CoNLL12': ['http://106.75.176.107:8088/DSRL/saved_models/en_SRL_BERT_CoNLL12_ls_0.1.0/model/',
                             'en_SRL_BERT_CoNLL12_ls_0.1.0']},
    'bert-middle-chinese': {'CPB2.0': ['http://106.75.176.107:8088/DSRL/saved_models/zh_SRL_BERT_CPB2.0_ls_0.1.0/model/', 'zh_SRL_BERT_CPB2.0_.1.0']},
    'bert-base-chinese': {'CPB2.0': ['http://106.75.176.107:8088/DSRL/saved_models/zh_SRL_BERT_CPB2.0_ls_0.1.0/model/', 'zh_SRL_BERT_CPB2.0_0.1.0']},
    'BiLSTM': {
        'CPB2.0': ['http://106.75.176.107:8088/DSRL/saved_models/zh_SRL_BiLSTM_CPB2.0_sm_0.1.0/model/', 'zh_SRL_BiLSTM_CPB2.0_0.1.0'],
        'CoNLL05': ['http://106.75.176.107:8088/DSRL/saved_models/en_SRL_BiLSTM_CoNLL05_sm_0.1.0/model/','en_SRL_BiLSTM_CoNLL05_sm_0.1.0'],
        'CoNLL12': ['http://106.75.176.107:8088/DSRL/saved_models/en_SRL_BiLSTM_CoNLL12_sm_0.1.0/model/','en_SRL_BiLSTM_CoNLL12_sm_0.1.0']
    }
}