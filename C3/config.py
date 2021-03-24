class Config:
    def __init__(self):
        self.embed_dense = True
        self.embed_dense_dim = 512  # 对BERT的Embedding降维
        self.warmup_proportion = 0.05
        self.use_bert = True
        self.keep_prob = 0.9
        self.relation_num = 21  # 实体的种类

        self.decay_rate = 0.1
        self.decay_step = 400
        self.num_checkpoints = 200

        self.train_epoch = 20
        self.sequence_length = 360  # Fine-tune sequence_length
        self.train_batch_size = 2  # Fine-tune batch_size

        self.learning_rate = 1e-4  # 下接结构的学习率 1e-4
        self.embed_learning_rate = 3e-5  # BERT的微调学习率 3e-5 5e-5

        self.model_type = "ALBERT"  # BERT XLNET

        # albert预训练模型的存放地址
        self.bert_file = '/home/hezoujie/Models/albert/model.ckpt-best'
        self.bert_config_file = '/home/hezoujie/Models/albert/albert_config.json'
        self.vocab_file = '/home/hezoujie/Models/albert/vocab_chinese.txt'

        # xlnet预训练模型的存放地址
        # self.bert_file = '/home/hezoujie/Models/xlnet/xlnet_model.ckpt'
        # self.xlnet_config_file = '/home/hezoujie/Models/xlnet/xlnet_config.json'
        # self.vocab_file = '/home/hezoujie/Models/albert/vocab_chinese.txt'
        # self.dropatt = 0.1
        # self.clamp_len = -1
        # self.is_training = True
        # self.use_tpu = False
        # self.use_bfloat16 = False
        # self.init = "normal"
        # self.init_range = 0.1
        # self.init_std = 0.02

        # roberta_wwm_large预训练模型的存放地址
        # self.bert_file = '/home/hezoujie/Models/roberta_wwm_large/bert_model.ckpt'
        # self.bert_config_file = '/home/hezoujie/Models/roberta_wwm_large/bert_config.json'
        # self.vocab_file = '/home/hezoujie/Models/roberta_wwm_large/vocab.txt'

        # predict.py ensemble.py get_ensemble_final_result.py post_ensemble_final_result.py的结果路径
        self.ensemble_source_file = '/home/hezoujie/Competition/CCKS_Military_NER/data/Military_entity_recog/ensemble/source_file/'
        self.ensemble_result_file = '/home/hezoujie/Competition/CCKS_Military_NER/data/Military_entity_recog/ensemble/result_file/'

        # 存放的模型名称，用以预测
        self.continue_training = False
        self.checkpoint_path = "/home/hezoujie/Competition/CCKS_Military_NER/data/Military_entity_recog/model/runs_7/1598627829/model_0.89_0.79_0.8378-693"

        self.model_dir = '/home/hezoujie/Competition/CCKS_Military_NER/data/Military_entity_recog/model'  # 模型存放地址
        self.new_data_process_quarter_final = '/home/hezoujie/Competition/CCKS_Military_NER/data/Military_entity_recog/clear_csv_data/data_process_quarter_final/'  # 数据预处理的结果路径
        self.source_data_dir = '/home/hezoujie/Competition/CCKS_Military_NER/data/Military_entity_recog/clear_csv_data/'  # 原始数据集

        self.lstm_dim = 512
        self.dropout = 0.4
        self.use_origin_bert = False  # True:使用原生bert, False:使用动态融合bert
