from functools import partial

import torch
### Path configurations ###
bucket = 'ngap--customer-data-science--prod--us-east-1'
folder = 'airbot/uat/bot_detection_nlp'
local_dir = '/tmp'

# Folders without s3 prefix
data_folder = '%s/data' % folder
result_folder = '%s/result' % folder
model_folder = '%s/model' % folder
config_folder = '%s/config' % folder

# Folders with s3 prefix
s3_data_folder = 's3://%s/%s/data' % (bucket, folder)
s3_result_folder = 's3://%s/%s/result' % (bucket, folder)
s3_model_folder = 's3://%s/%s/model' % (bucket, folder)
s3_config_folder = 's3://%s/%s/config' % (bucket, folder)

### Configuration for datasets ###
# Initialized di

# User behavior dataset
user_behavior_table = 'digital_exp.fact_clickstream_session_snkrsapp'
session_start_dt = '2022-09-25' # starting session date to generate session-level user data
where_clause = "%s >= '%s'" % ('session_start_dt', session_start_dt)
max_behavior_len = 10 # maximum allowed length for behavior sequence

# Column name of row index
row_idx_colname = 'row_idx'

# Primary key columns (i.e., columns with unique values)
primary_key_cols = ['session_skey', 'upm_id', 'visitor_skey']

# Categorical features of user behavior data
cat_user_behavior_columns = [
                "upm_id",
                "session_skey",
                "visitor_skey",
                "member_id",
                "ip_domain_id",
                "country_skey",
                "city_skey",
#                 "user_agent",
                "mobile_id",
                "mobile_carrier_skey",
                "mobile_day_of_week_desc",
              ]

# Column name for label
target_column = 'target_col'

# Continuous features of user behavior data
cont_user_behavior_columns = [
                "hit_cnt",
                "pv_cnt",
                "purchase_cnt",
                "session_elapsed_time",
                "session_seq_num",
                "mobile_days_since_first_use",
                "mobile_days_since_last_use",
               ]

# Add target columns to continuous column
cont_user_behavior_columns += [target_column]

# Sequential(i.e., the value of which is array-like object) features of user behavior data
seq_user_behavior_columns = [
#                              "search_terms",
                             "browsed_product_list",
                            ]

user_behavior_columns = cat_user_behavior_columns + cont_user_behavior_columns

# Groupby columns used for analysis granulation
groupby_user_behavior_columns = ["upm_id", "session_skey"]

# Data prefix for embedding usage, may be varied for different data
emb_data_prefix = '%s_%s_membership_score_matched' % (user_behavior_table.split('.')[-1], session_start_dt.replace('-','_'))

# Paths to save embedding result files
user_emb_fname = '%s_seq_embedding_result.parquet' % emb_data_prefix
user_emb_fpath = '%s/%s' % (s3_result_folder, user_emb_fname)

# Paths to save embedding module (object) files
user_emb_mod_fname = '%s_user_emb_module.pkl' % emb_data_prefix
user_emb_mod_fpath = '%s/%s' % (result_folder, user_emb_mod_fname)

### NVtabluar configurations ###
# Categorical columns
nvt_cat_cols = cat_user_behavior_columns
# Continuous columns
nvt_cont_cols = cont_user_behavior_columns
# Groupby key columns
nvt_groupby_cols = []
# Dictionary of aggregation manipulation for nvtabluar
nvt_groupby_aggdict = {}
# Seperation character tag for aggregated columns
nvt_groupby_sep = '_'
# Padding value for ragged sequential columns
nvt_padvalue = 0
# Columns involved in groupby operation
nvt_groupby_invol_cols = nvt_groupby_cols + list(nvt_groupby_aggdict.keys())
# Column names of sequential columns
nvt_seq_cols = seq_user_behavior_columns
# Maximum length for sequential columns in nvtabular workflow
nvt_seq_maxlen = max_behavior_len
# Postfix/Suffix for processed sequential columns
nvt_seq_col_postfix = '_trim_%d' % nvt_seq_maxlen
# Folder to save workflow related files
nvt_workflow_folder = 'nvt_workflow_jeff'
# Folder to save files of nvtabular processed (i.e., fit_tranform) dataset
nvt_processed_files_folder = 'processed_nvt_jeff'
nvt_processed_files_path = '%s/%s' % (local_dir, nvt_processed_files_folder)
s3_nvt_processed_files_path = '%s/%s' % (s3_result_folder, nvt_processed_files_folder)

# nvtabular dataset load mode (e.g., True for CPU or False for GPU)
nvt_cpu = False

# nvt_processed_files_path = '%s/%s' % (s3_result_folder, nvt_processed_files_folder)
# path to save workflow files
# Currently we can't use s3 path to save workflow files because 'fit/fit_transform' requires local directory to implement
nvt_workflow_directory = '%s/%s' % (local_dir, nvt_workflow_folder)

# nvt_workflow_directory = '%s/%s' % (s3_result_folder, nvt_workflow_folder)
# File names and paths for nvtabular related files
nvt_schema_fname = 'schema.pbtxt'
nvt_schema_path = '%s/%s' % (nvt_processed_files_path, nvt_schema_fname)
nvt_schema_jspath = nvt_schema_path.replace('.pbtxt', '.json')
# schema.json path that stored at s3
s3_nvt_schema_jspath = '%s/%s/%s' % (result_folder, nvt_processed_files_folder, 'schema.json')
# schema.json path that download from s3 to local directory
download_local_nvt_schema_jspath = '%s/schema_jeff.json' % local_dir

######### Configurations for deep learning #########
# Whether to use pre-trained module for input building
use_pretrain = False

### Embedding configurations ###
embedding_dim_dict = {
                      'browsed_product_list%s' % nvt_seq_col_postfix : 128,
                      'DEFAULT' : 64,
                     }
embedding_initializer_dict = {
                      'browsed_product_list%s' % nvt_seq_col_postfix: partial(torch.nn.init.normal_, mean = 1, std = 1.5),
                      'DEFAULT' : partial(torch.nn.init.normal_, mean = 0, std = 1),
                             }

embedding_dict = {
  'dim' : embedding_dim_dict,
  'initializer' : embedding_initializer_dict,
                  }

emb_agg_method = 'concat'
transformer_mask = 'clm'

### Model structure configurations ###
mlp_out_dim = 128

### Training configuration ###
trainer_config_dict = {
                      'output_dir' : '%s/%s' % (local_dir, 'transformers4rec'),
                      'data_loader_engine' : 'pyarrow', # Alternative parameter value 'nvtabular' (for GPU)
                      'num_train_epochs' : 3,
                      'dataloader_drop_last' : False,
                      'per_device_train_batch_size' : 256,
                      'per_device_eval_batch_size' : 32,
                      'gradient_accumulation_steps' : 1,
                      'learning_rate' : 0.000666,
                      'report_to' : [],
                      'logging_steps' : 200,
                      'dataloader_num_workers' : 2,
                      'dataloader_pin_memory' : False,
                      }
dataloader_config_dict = {
                          'batch_size' : 256,
                          'max_sequence_length' : nvt_seq_maxlen,
                          'shuffle' : True,
#                           'shuffle_buffer_size' : 0,
                          'num_workers' : 1,
                          'pin_memory' : True,
                          'max_sequence_length' : nvt_seq_maxlen,
                          'collate_fn' : lambda x: x,  # This parameter merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
                          }

model_fit_config_dict = {
                        'optimizer' : torch.optim.Adam,
                        'eval_dataloader' : None,
                        'num_epochs' : 2,
                        'amp' : False,
                        'train' : True,
                        'verbose' : True,
                        }