%pip install sentence_transformers==2.2.2 fugashi==1.3.0 unidic-lite==1.0.8 dgl==1.1.2 torch==1.13.1 pydantic==1.9.0 pyyaml==5.4.1 networkx==2.8.5 s3pathlib==2.1.2 --quiet

%run /Repos/Jefferson.He@nike.com/bot_detection_nlp/notebook/address_analysis/address_utils

%run /Repos/Jefferson.He@nike.com/bot_detection_nlp/notebook/address_analysis/address_training/address_configs

%run /Repos/Jefferson.He@nike.com/bot_detection_nlp/notebook/neptune_graph_building/neptune_utils

%run /Repos/Jefferson.He@nike.com/bot_detection_nlp/notebook/neptune_graph_building/graph_configs

from pyspark.sql.functions import col, length, lower, sum, avg, max as spark_max, count, collect_list, collect_set, concat_ws, udf, arrays_zip, aggregate, sqrt, countDistinct, lit, datediff, current_date, explode, map_keys, broadcast, monotonically_increasing_id, lit
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.sql.functions import lit, row_number
from numpy.random import rand, randint
from itertools import combinations
from numpy import dot
from numpy.linalg import norm
from s3fs.core import S3FileSystem
from subprocess import run
# from torch import nn
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util, models
from s3pathlib import S3Path, not_
from functools import partial
from s3pathlib import context

import boto3
import pandas as pd
import pyspark.sql.functions as F
import boto3
import numpy as np
import math
import time
import pickle as pkl
import sys
import torch
import re
import json
# import dgl
import torch as th
import datetime
import re
import os
import networkx as nx


%pip install torch_geometric

import torch
from torch import Tensor,ones,randn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
    dropout_edge,
)

in_channels = 8
heads, out_channels = 1, 24
lin_src = Linear(in_channels, heads * out_channels, bias=False,)

input_ts = torch.randn(4,in_channels)
att_src = torch.Tensor(1, heads, out_channels)
x_src = lin_src(input_ts).view(-1, heads, out_channels)
att_src.size()


alpha_src = (x_src * att_src).sum(dim=-1)
alpha_src.size()

import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
data


isinstance(edge_index, SparseTensor)

dir(data)

data.csr()

from datetime import date, timedelta, datetime
dt_fmt = '%Y-%m-%d'
lookback_start_date = '2024-03-29'
back_fill_num_day = 14

start_date_increment = 0
back_fill_start_date = datetime.strftime(datetime.strptime(lookback_start_date, dt_fmt) + timedelta(days = -start_date_increment), dt_fmt)

backfill_query = '''
select count(distinct SHIPPING_ADDRESS_ADDRESS1) as uni_addr_num, max(received_date) as max_date, min(received_date) as min_date
from launch_secure.launch_entries
where UPMID is not null and LOCALE rlike 'es[-_](MX|LA)' and SHIPPING_ADDRESS_COUNTRY in ('MX')
and received_date >= date_add('{back_fill_start_date}', -datediff(current_date(), date_add('{back_fill_start_date}', -1))*({back_fill_num_day}-1)) and received_date <= date_add('{back_fill_start_date}', -datediff(current_date(), '{back_fill_start_date}')*({back_fill_num_day}-1))
'''.format(
  back_fill_start_date = '{}'.format(back_fill_start_date),
  back_fill_num_day = back_fill_num_day
  )

print(backfill_query)


s3_path_obj = S3Path(s3_path)
bucket_name = s3_path_obj.bucket
# Generate boto3 session with valid aws credential
boto3_session = gen_valid_boto3_s3_session(bucket_name)
# Attach valid boto3 session to s3pathlib
context.attach_boto_session(boto3_session)


days = [('2024', '04', '07'), ('2024', '04', '06'), ('2024', '04', '05'), ('2024', '04', '04'), ('2024', '04', '03'), ('2024', '04', '02'), ('2024', '04', '01'), ('2024', '03', '31'), ('2024', '03', '30'), ('2024', '03', '29'), ('2024', '03', '28'), ('2024', '03', '27'), ('2024', '03', '26'), ('2024', '03', '25'), ('2024', '03', '24'), ('2024', '03', '23'), ('2024', '03', '22'), ('2024', '03', '21'), ('2024', '03', '20'), ('2024', '03', '19'), ('2024', '03', '18'), ('2024', '03', '17'), ('2024', '03', '16'), ('2024', '03', '15'), ('2024', '03', '14'), ('2024', '03', '13'), ('2024', '03', '12'), ('2024', '03', '11'), ('2024', '03', '10'), ('2024', '03', '09'), ('2024', '03', '08'), ('2024', '03', '07'), ('2024', '03', '06'), ('2024', '03', '05'), ('2024', '03', '04'), ('2024', '03', '03'), ('2024', '03', '02'), ('2024', '03', '01'), ('2024', '02', '29'), ('2024', '02', '28'), ('2024', '02', '27'), ('2024', '02', '26'), ('2024', '02', '25'), ('2024', '02', '24'), ('2024', '02', '23'), ('2024', '02', '22'), ('2024', '02', '21'), ('2024', '02', '20'), ('2024', '02', '19'), ('2024', '02', '18'), ('2024', '02', '17'), ('2024', '02', '16'), ('2024', '02', '15'), ('2024', '02', '14'), ('2024', '02', '13'), ('2024', '02', '12'), ('2024', '02', '11'), ('2024', '02', '10'), ('2024', '02', '09'), ('2024', '02', '08'), ('2024', '02', '07'), ('2024', '02', '06'), ('2024', '02', '05'), ('2024', '02', '04'), ('2024', '02', '03'), ('2024', '02', '02'), ('2024', '02', '01'), ('2024', '01', '31'), ('2024', '01', '30'), ('2024', '01', '29'), ('2024', '01', '28'), ('2024', '01', '27'), ('2024', '01', '26'), ('2024', '01', '25'), ('2024', '01', '24'), ('2024', '01', '23'), ('2024', '01', '22'), ('2024', '01', '21'), ('2024', '01', '20'), ('2024', '01', '19'), ('2024', '01', '18'), ('2024', '01', '17'), ('2024', '01', '16'), ('2024', '01', '15'), ('2024', '01', '14'), ('2024', '01', '13'), ('2024', '01', '12'), ('2024', '01', '11'), ('2024', '01', '10'), ('2024', '01', '09'), ('2024', '01', '08'), ('2024', '01', '07'), ('2024', '01', '06'), ('2024', '01', '05'), ('2024', '01', '04'), ('2024', '01', '03'), ('2024', '01', '02'), ('2024', '01', '01'), ('2023', '12', '31'), ('2023', '12', '30'), ('2023', '12', '29'), ('2023', '12', '28'), ('2023', '12', '27'), ('2023', '12', '26'), ('2023', '12', '25'), ('2023', '12', '24'), ('2023', '12', '23'), ('2023', '12', '22'), ('2023', '12', '21'), ('2023', '12', '20'), ('2023', '12', '19'), ('2023', '12', '18'), ('2023', '12', '17'), ('2023', '12', '16'), ('2023', '12', '15'), ('2023', '12', '14'), ('2023', '12', '13'), ('2023', '12', '12'), ('2023', '12', '11'), ('2023', '12', '10'), ('2023', '12', '09'), ('2023', '12', '08'), ('2023', '12', '07'), ('2023', '12', '06'), ('2023', '12', '05'), ('2023', '12', '04'), ('2023', '12', '03'), ('2023', '12', '02'), ('2023', '12', '01'), ('2023', '11', '30'), ('2023', '11', '29'), ('2023', '11', '28'), ('2023', '11', '27'), ('2023', '11', '26'), ('2023', '11', '25'), ('2023', '11', '24'), ('2023', '11', '23'), ('2023', '11', '22'), ('2023', '11', '21'), ('2023', '11', '20'), ('2023', '11', '19'), ('2023', '11', '18'), ('2023', '11', '17'), ('2023', '11', '16'), ('2023', '11', '15'), ('2023', '11', '14'), ('2023', '11', '13'), ('2023', '11', '12'), ('2023', '11', '11'), ('2023', '11', '10'), ('2023', '11', '09'), ('2023', '11', '08'), ('2023', '11', '07'), ('2023', '11', '06'), ('2023', '11', '05'), ('2023', '11', '04'), ('2023', '11', '03'), ('2023', '11', '02'), ('2023', '11', '01'), ('2023', '10', '31'), ('2023', '10', '30'), ('2023', '10', '29'), ('2023', '10', '28'), ('2023', '10', '27'), ('2023', '10', '26'), ('2023', '10', '25'), ('2023', '10', '24'), ('2023', '10', '23'), ('2023', '10', '22'), ('2023', '10', '21'), ('2023', '10', '20'), ('2023', '10', '19'), ('2023', '10', '18'), ('2023', '10', '17'), ('2023', '10', '16'), ('2023', '10', '15'), ('2023', '10', '14'), ('2023', '10', '13'), ('2023', '10', '12'), ('2023', '10', '11')]
path = '{' + ",".join(["day={}-{}-{}".format(year,month,day) for year,month,day in days]) +'}'

us_user_akamai_features = 's3://ngap--customer-data-science--prod--us-east-1/airbot/prod/membership_detection/data/us/user_akamai_features/'
akamai_log_df = spark.read.parquet("{}{}/".format(us_user_akamai_features,path), header=True).drop_duplicates()


%sh

aws s3 ls s3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/config/

%sh

aws s3 ls --h --s --recursive s3://ngap--customer-data-science--prod--us-east-1/airbot/prod/sole-volume/address_community_detection/config/


%sh

aws s3 ls s3://ngap--customer-data-science--prod--us-east-1/airbot/prod/sole-volume/address_community_detection/kr_korean/inference_datetime_20240409_041009/ --h --s --recursive

%sh

aws s3 ls s3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/test/


%sh

aws s3 cp s3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/result/address_analysis_datetime_20230510_111130/inference_datetime_20240409_041009/ /tmp/inference_datetime_20240409_041009/ --recursive


%sh

ls -lh /tmp/inference_datetime_20240409_041009.zip


%sh

zip -r /tmp/inference_datetime_20240409_041009.zip /tmp/inference_datetime_20240409_041009/

%sh

aws s3 cp 's3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/result/address_analysis_20230327_08353/inference_datetime_20230426_070130/community_detection_results/graph_node_UPMID_thrs_0.85.gpkl' s3://ngap--customer-data-science--prod--us-east-1/airbot/prod/sole-volume/address_community_detection/sample_graph/

%sh

aws s3 cp /tmp/inference_datetime_20240409_041009.zip s3://ngap--customer-data-science--prod--us-east-1/airbot/prod/sole-volume/


%sh

aws s3 ls s3://ngap--customer-data-science--prod--us-east-1/airbot/prod/sole-volume/


dbutils.fs.cp('/Volumes/ngap_hms_east_deprecated/airbot_prod/airbots_schema_vol/inference_datetime_20240409_041009.zip',
              's3://bt-airbot0-useast1-features/address_community_detection/kr_korean/inference_datetime_20240409_041009.zip')



%sh

zip -r -s 4G /tmp/inference_datetime_20240409_041009_splitted.zip /tmp/inference_datetime_20240409_041009/

%sh
aws s3 ls s3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/result/address_analysis_datetime_20230510_111130/inference_datetime_20240330_040727/node_white_list.parquet/


'{:,}'.format(df_org.count())


df_res = df_org.join(df_forter, on = ['trans_id'], how = 'inner')
df_res.display()


'{:,}'.format(df_res.dropDuplicates(subset=['trans_id']).count())


def valid_addr_result_s3_dir(
                             s3path,
                             addr_comm_res_dirname,
                             graph_suffix
                             ):
  valid_graph_file_cnt = 0
  for sub_dir in s3path.iterdir().filter(S3Path.basename == addr_comm_res_dirname):
    valid_graph_file_cnt += len(sub_dir.iter_objects(recursive = False).filter(S3Path.ext == graph_suffix).all())
  return valid_graph_file_cnt == 1

valid_addr_result_s3_dir_partial = partial(valid_addr_result_s3_dir, addr_comm_res_dirname = comm_detect_result_directory, graph_suffix = '.{}'.format(graph_suffix))


s3_dir = 's3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/result/address_analysis_datetime_20240202_143955/inference_datetime_20240208_040737/'
# Directory that saves community detection results
s3_dir_obj = S3Path(s3_dir)



for sub_dir in s3_dir_obj.iterdir().filter(S3Path.basename == comm_detect_result_directory):
  for sub_s3path in sub_dir.iterdir():
    print(sub_s3path.basename.endswith('louvain_communities.parquet') == True)


for x in s3_dir_obj.iterdir().all():
  print(x.basename.endswith('louvain_communities.parquet'))


# Function that targets all the louvain community result parquet files
def get_comm_res_parquet(s3_path, graph_parent_folder_name):
    for sub_dir in s3_path.iterdir().filter(S3Path.basename == graph_parent_folder_name):
        for sub_s3path in sub_dir.iterdir():
            return sub_s3path.basename.endswith('louvain_communities.parquet')


get_comm_res_parquet_partial = partial(get_comm_res_parquet, graph_parent_folder_name=comm_detect_result_directory)


# Function that targets all the networkX graph files at the S3 directory of specific pipeline running
def get_networkx_graph_files(s3_path, graph_parent_folder_name, graph_suffix):
    for sub_dir in s3_path.iterdir().filter(S3Path.basename == graph_parent_folder_name):
        for sub_s3path in sub_dir.iter_objects(recursive=False):
            return sub_s3path.ext == graph_suffix


get_networkx_graph_files_partial = partial(get_networkx_graph_files,
                                           graph_parent_folder_name=comm_detect_result_directory,
                                           graph_suffix='.{}'.format(graph_suffix))

def gen_sub_s3path_list(s3_path, cond_func, include_folder = True, recursive = False, exclude = False, convert_to_s3url = False):
  s3_path_obj = S3Path(s3_path)

  logging.info('Start filtering sub S3 paths which {} the input condition at S3 path "{}", sub directories are {}.'.format('satisfy' if not exclude else 'DON\'T satisfy',
                                                                                                                           s3_path,
                                                                                                                           'included' if include_folder else 'NOT included'))

  if include_folder:
    s3_proxy = s3_path_obj.iterdir()
  else:
    s3_proxy = s3_path_obj.iter_objects(recursive = recursive)

  if not exclude:
    sub_s3path_obj_lt = s3_proxy.filter(cond_func).all()
  else:
    sub_s3path_obj_lt = s3_proxy.filter(not_(cond_func)).all()

  if convert_to_s3url:
    sub_s3path_obj_lt = [x.uri for x in sub_s3path_obj_lt]

  return sub_s3path_obj_lt


s3_dir = 's3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/result/address_analysis_datetime_20240202_143955/'
# Directory that saves community detection results
s3_path = S3Path(s3_dir)

res_lt = gen_sub_s3path_list(s3_dir, get_comm_res_parquet_partial, exclude = False)
res_lt



[x.uri for x in res_lt]


%sh

aws s3 ls s3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/result/address_analysis_datetime_20240202_143955/inference_datetime_20240205_065102/community_detection_results/


### Function that generate list of first level files or sub-directories for specific s3 directory ###
def gen_fst_lvl_sub_lt(s3_dir):
  shell_cmd = 'aws s3 ls {}/ | awk \'{{FS = " " ; print $2}}\''.format(s3_dir)
  result = run(shell_cmd, capture_output=True, shell=True)
  reuslt_str = result.stdout.decode('utf8')
  subfolder_lt = list(filter(lambda x : len(x) > 0, map(lambda x : re.sub('/$', '', x), reuslt_str.split('\n')))) if reuslt_str else []
  return subfolder_lt

s3_dir = 's3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/result/address_analysis_20230327_08353'
subfolder_lt = gen_fst_lvl_sub_lt(s3_dir)


s3_path = 's3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/data/'
s3_path_obj = S3Path(s3_path)

s3_path_obj.iterdir().all()
# s3_path_obj.iter_objects(recursive = False, include_folder = True).all()


for x in S3Path('s3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/data/address_abbreviations.xlsx').iterdir():
  print(x.basename)



%sh
aws s3 ls s3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/data/

s3_path_obj.delete()


help(s3_path_obj.delete)

s3_path_obj.delete()

%sh

aws s3 ls s3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/result/trained_model.onnx


### Filter functions for S3Pathlib.S3Path ###
# Function that filter valid s3 directory for address analysis result
def valid_addr_result_s3_dir(s3path,
                             addr_comm_res_dirname,
                             graph_suffix):
  valid_graph_file_cnt = 0
  for sub_dir in s3path.iterdir().filter(S3Path.basename == addr_comm_res_dirname):
    valid_graph_file_cnt += len(sub_dir.iter_objects(recursive = False).filter(S3Path.ext == graph_suffix).all())
  return valid_graph_file_cnt == 1

len(s3_path.iterdir().filter(valid_addr_result_s3_dir).all())


invalid_subdir_lt = []
for s3_sub_dir in s3_path.iterdir().filter(S3Path.basename == comm_detect_result_directory).filter(S3Path.iter_objects(recursive = False).filter(S3Path.ext == graph_suffix)):
  # s3_target_dir = None
  # for path in s3_sub_dir.iterdir():
  #   if path.basename == comm_detect_result_directory:
  #     s3_target_dir = path
  # if s3_target_dir:
  s3_graph_file_path_lt = list(map(lambda x : x.uri, s3_target_dir.iter_objects(recursive = False).filter(S3Path.ext == graph_suffix)))
  # else:
  #   s3_graph_file_path_lt = []

  if len(s3_graph_file_path_lt) != 1:
    invalid_subdir_lt.append(s3_sub_dir)

print('{:} subfolders and {:,} invalid ones.'.format(len(s3_path.iterdir().all()), len(invalid_subdir_lt)))


sub_s3_dir
s3_path.fname()

# Find invalid subfolder, i.e., folder without graph file ###
graph_suffix = '.gpkl'
invalid_subfolder_lt = []
for subfolder in subfolder_lt:
  graph_file_lt = gen_target_pattern_s3_obj_list('{}/{}'.format(s3_dir, subfolder), 'Key', '\.{}$'.format(graph_suffix))
  if not len(graph_file_lt) == 1:
    invalid_subfolder_lt.append(subfolder)

print('{:} subfolders and {:,} invalid ones.'.format(len(subfolder_lt), len(invalid_subfolder_lt)))


gen_target_pattern_s3_obj_list(s3_addr_analy_result_parent_directory, 'Key', '\.{}$'.format(graph_suffix))

gen_fst_lvl_sub_lt

s3_path = 's3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/result/address_analysis_20230327_08353'
bucket_name, obj_key = seperate_s3_path(s3_path)
s3_client = gen_valid_s3_client(bucket_name)
paginator = s3_client.get_paginator('list_objects_v2')

pages = paginator.paginate(Bucket = bucket_name, Prefix = obj_key)
for page in pages:
  s3_dir_contents = page.get('Contents', [])
  break

print(s3_dir_contents)


### Function that checks the uniqueness of specific file at s3 directory ###
def check_file_uniqueness(s3_path, pattern):
  bucket_name, ___ = seperate_s3_path(s3_path)
  file_obj_list = gen_target_pattern_s3_obj_list(s3_path, 'Key', pattern)

  if not file_obj_list:
    raise NotImplementedError('No file has been found in the s3 directory:"{}" and thus this directory is NOT valid.Please review the configuration.'.format(s3_path))
  else:
    if len(file_obj_list) > 1:
      logging.info('More than 1 ({:,} actually) file has been found at s3 directory:"{}" and thus this directory is NOT valid.Please review the configuration.'.format(len(file_obj_list), s3_path))
    else:
      logging.info('1 file has been found at s3 directory:"{}" and this directory is VALID.'.format(s3_path))
      s3_unique_file_obj_path = 's3://{}/{}'.format(bucket_name, file_obj_list[0])

  return s3_unique_file_obj_path

s3_path = 's3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/result/address_analysis_20230327_08353/inference_datetime_20240316_040826/'
pattern = '\.{}$'.format('gpkl')
check_file_uniqueness(s3_path, pattern)


start_after_key = 'airbot/uat/bot_detection_nlp/result/address_analysis_20230327_08353/inference_run_999589728_datetime_20230829_040047'
obj_key = 'airbot/uat/bot_detection_nlp/result/address_analysis_20230327_08353/inference_run_999589728_datetime_20230829_040047/inference_datetime_20240319_040747'
s3_client = gen_valid_s3_client(bucket_name)
result = s3_client.list_objects_v2(Bucket = bucket_name, Prefix = obj_key)
[x.get('Key', None) for x in result.get('Contents', None)]

logging.info('Start searching for graph pickle file(s) at "{}", it may take a while.'.format(
    s3_addr_analy_result_parent_directory))
s3_graph_obj_list = gen_target_pattern_s3_obj_list(s3_addr_analy_result_parent_directory, 'Key',
                                                   '\.{}$'.format(graph_suffix))
if not s3_graph_obj_list:
    raise NotImplementedError(
        'No graph file has been found in the s3 directory:"{}" and thus no base folder for inference process. Please review the configuration.'.format(
            s3_addr_analy_result_parent_directory))
else:
    logging.info('{:,} graph file(s) has(have) been found.'.format(len(s3_graph_obj_list)))

    for s3_graph_path_without_bucket in s3_graph_obj_list:
        subfolder_indicator = get_s3_obj_attrdict(bucket, s3_graph_path_without_bucket)[subfolder_indicator_name]
        subfolder_name = \
        s3_graph_path_without_bucket.split(s3_addr_analy_result_parent_directory_without_bucket + '/')[-1].split('/')[0]
        subfolder_indicator_dict[subfolder_name] = {'graph_file': s3_graph_path_without_bucket,
                                                    'indicator': subfolder_indicator}


sql_query = 'select {} from {}'.format(node_id, 'tmp_table')
res_df = spark.sql(sql_query)
res_df.display()


elect count(distinct shipping_address) as uni_addr_num, max(txn_dt) as max_txn_date, min(txn_dt) as min_txn_date
from fraud_secure.forter_type1
where account_id is not null and ship_country = 'JP'
and txn_dt between date_add('{back_fill_start_date}', -(datediff(current_date(), date_add('{back_fill_start_date}', -1))*{back_fill_num_day})+1) and date_add('{back_fill_start_date}', -datediff(current_date(), '{back_fill_start_date}')*({back_fill_num_day}))

for start_date_increment in range(124):
    backfill_query = '''
  select count(distinct shipping_address) as uni_addr_num, max(txn_dt) as max_txn_date, min(txn_dt) as min_txn_date
  from fraud_secure.forter_type1
  where account_id is not null and ship_country = 'JP'
  and txn_dt >= date_add('{back_fill_start_date}', -datediff(date_add(current_date(),{start_date_increment}), date_add('{back_fill_start_date}', -1))*({back_fill_num_day}-1)) and txn_dt <= date_add('{back_fill_start_date}', -datediff(date_add(current_date(),{start_date_increment}), '{back_fill_start_date}')*({back_fill_num_day}-1))
  '''.format(back_fill_start_date='2024-02-05',
             back_fill_num_day=7,
             start_date_increment=start_date_increment)

    df = spark.sql(backfill_query)
    if start_date_increment == 0:
        result_df = df
    else:
        result_df = result_df.union(df)

    if start_date_increment == 3:
        break

result_df.display()

### Retrieve the number of addresses to be backfilled for each interal ###

from datetime import date, timedelta, datetime

dt_fmt = '%Y-%m-%d'
lookback_start_date = '2024-02-05'

for start_date_increment in range(123):
    back_fill_start_date = datetime.strftime(
        datetime.strptime(lookback_start_date, dt_fmt) + timedelta(days=-start_date_increment), dt_fmt)

    backfill_query = '''
  select count(distinct shipping_address) as uni_addr_num, max(txn_dt) as max_txn_date, min(txn_dt) as min_txn_date
  from fraud_secure.forter_type1
  where account_id is not null and ship_country = 'JP'
  and txn_dt >= date_add('{back_fill_start_date}', -datediff(current_date(), date_add('{back_fill_start_date}', -1))*({back_fill_num_day}-1)) and txn_dt <= date_add('{back_fill_start_date}', -datediff(current_date(), '{back_fill_start_date}')*({back_fill_num_day}-1))
  '''.format(back_fill_start_date='{}'.format(back_fill_start_date),
             back_fill_num_day=7)

    df = spark.sql(backfill_query)
    if start_date_increment == 0:
        result_df = df
    else:
        result_df = result_df.union(df)

result_df.display()


node_lt = [
  {
    "Node_Source_Table": None,
    "Node_ID": "concat_ws('__', lit('upm'), col('upmid'))",
    "Node_S3_Path": "s3://ngap--customer-data-science--prod--us-east-1/airbot/prod/membership_detection/data/neptune_source/us/node/nike_member",
    "Node_Label_Name": "gnn_member",
    "Node_Attributes": {
      "account_age": {
        "Aggregate": False,
        "Aggregate_Method": []
      },
      "is_bot": {
        "Aggregate": False,
        "Aggregate_Method": []
      }
    }
  },
  {
    "Node_Source_Table": None,
    "Node_ID": "concat_ws('__', lit('sub'), col('subscription_id'))",
    "Node_S3_Path": "s3://ngap--customer-data-science--prod--us-east-1/airbot/prod/membership_detection/data/neptune_source/us/node/nike_sub",
    "Node_Label_Name": "gnn_subscription",
    "Node_Attributes": {
      "upmid_set": {
        "Aggregate": False,
        "Aggregate_Method": []
      }
    }
  },
  {
    "Node_Source_Table": None,
    "Node_ID": "concat_ws('__', lit('siteid'), col('registration_siteid'))",
    "Node_S3_Path": "s3://ngap--customer-data-science--prod--us-east-1/airbot/prod/membership_detection/data/neptune_source/us/node/nike_site",
    "Node_Label_Name": "gnn_registration_site",
    "Node_Attributes": {
      "upmid_set": {
        "Aggregate": False,
        "Aggregate_Method": []
      }
    }
  },
  {
    "Node_Source_Table": None,
    "Node_ID": "concat_ws('__', lit('memb_email'), col('email_domain'))",
    "Node_S3_Path": "s3://ngap--customer-data-science--prod--us-east-1/airbot/prod/membership_detection/data/neptune_source/us/node/nike_member_email",
    "Node_Label_Name": "gnn_member_email_domain",
    "Node_Attributes": {
      "upmid_set": {
        "Aggregate": False,
        "Aggregate_Method": []
      }
    }
  },
  {
    "Node_Source_Table": None,
    "Node_ID": "concat_ws('__', lit('memb_phno'), col('phone_number'))",
    "Node_S3_Path": "s3://ngap--customer-data-science--prod--us-east-1/airbot/prod/membership_detection/data/neptune_source/us/node/nike_member_phone_number",
    "Node_Label_Name": "gnn_member_phone_number",
    "Node_Attributes": {
      "upmid_set": {
        "Aggregate": False,
        "Aggregate_Method": []
      }
    }
  },
  {
    "Node_Source_Table": None,
    "Node_ID": "concat_ws('__', lit('akam_sbn'), col('akamai_subnet'))",
    "Node_S3_Path": "s3://ngap--customer-data-science--prod--us-east-1/airbot/prod/membership_detection/data/neptune_source/us/node/akamai_subnet",
    "Node_Label_Name": "gnn_akamai_subnet",
    "Node_Attributes": {
      "upmid_set": {
        "Aggregate": False,
        "Aggregate_Method": []
      }
    }
  },
  {
    "Node_Source_Table": None,
    "Node_ID": "concat_ws('__', lit('akam_local'), col('akamai_location'))",
    "Node_S3_Path": "s3://ngap--customer-data-science--prod--us-east-1/airbot/prod/membership_detection/data/neptune_source/us/node/akamai_location",
    "Node_Label_Name": "gnn_akamai_location",
    "Node_Attributes": {
      "upmid_set": {
        "Aggregate": False,
        "Aggregate_Method": []
      }
    }
  },
  {
    "Node_Source_Table": None,
    "Node_ID": "concat_ws('__', lit('akam_app'), col('app_id'))",
    "Node_S3_Path": "s3://ngap--customer-data-science--prod--us-east-1/airbot/prod/membership_detection/data/neptune_source/us/node/akamai_app",
    "Node_Label_Name": "gnn_akamai_app",
    "Node_Attributes": {
      "upmid_set": {
        "Aggregate": False,
        "Aggregate_Method": []
      }
    }
  },
  {
    "Node_Source_Table": None,
    "Node_ID": "concat_ws('__', lit('akam_agt'), col('user_agent'))",
    "Node_S3_Path": "s3://ngap--customer-data-science--prod--us-east-1/airbot/prod/membership_detection/data/neptune_source/us/node/akamai_user_agent",
    "Node_Label_Name": "gnn_akamai_user_agent",
    "Node_Attributes": {
      "upmid_set": {
        "Aggregate": False,
        "Aggregate_Method": []
      }
    }
  }
]

node_label_lt = []

for node_dict in node_lt:
  node_label_lt.append(node_dict['Node_Label_Name'])


node_label_lt

eDF = spark.createDataFrame([Row(a=1, intlist=[1,2,3], mapfield={"a": "b"})])
eDF.select(explode(eDF.mapfield)).show()

backfill_query = '''
select count(distinct shipping_address) as uni_addr_num, max(txn_dt) as max_txn_date, min(txn_dt) as min_txn_date
from fraud_secure.forter_type1
where account_id is not null and ship_country = 'JP'
and txn_dt >= date_add('{back_fill_start_date}', -datediff(current_date(), date_add('{back_fill_start_date}', -1))*({back_fill_num_day}-1)) and txn_dt <= date_add('{back_fill_start_date}', -datediff(current_date(), '{back_fill_start_date}')*({back_fill_num_day}-1))
'''.format(back_fill_start_date = '2024-01-31',
           back_fill_num_day = 7)
print(backfill_query)

df = spark.sql(backfill_query)
df.display()


select count(distinct shipping_address) as uni_addr_num, max(txn_dt) as max_txn_date, min(txn_dt) as min_txn_date
from fraud_secure.forter_type1
where account_id is not null and ship_country = 'JP'
and txn_dt >= date_add('2024-02-01', -datediff(current_date(), date_add('2024-02-01', -1))*(7-1)) and txn_dt <= date_add('2024-02-01', -datediff(current_date(), '2024-02-01')*(7-1))


backfill_query = '''
select count(distinct shipping_address) as uni_addr_num, max(txn_dt) as max_txn_date, min(txn_dt) as min_txn_date
from fraud_secure.forter_type1
where account_id is not null and ship_country = 'JP'
and txn_dt >= date_add('{back_fill_start_date}', -datediff(current_date(), date_add('{back_fill_start_date}', -1))*({back_fill_num_day}-1)) and txn_dt <= date_add('{back_fill_start_date}', -datediff(current_date(), '{back_fill_start_date}')*({back_fill_num_day}-1))
'''.format(back_fill_start_date = '2023-12-30',
           back_fill_num_day = 7)
print(backfill_query)

df = spark.sql(backfill_query)
df.display()


backfill_query = '''
select count(distinct shipping_address) as uni_addr_num, max(txn_dt) as max_txn_date, min(txn_dt) as min_txn_date
from fraud_secure.forter_type1
where account_id is not null and ship_country = 'JP'
and txn_dt >= date_add('{back_fill_start_date}', -datediff(current_date(), date_add('{back_fill_start_date}', -1))*{back_fill_num_day}) and txn_dt <= date_add('{back_fill_start_date}', -datediff(current_date(), '{back_fill_start_date}')*{back_fill_num_day})
'''.format(back_fill_start_date = '2023-12-30',
           back_fill_num_day = 7)
print(backfill_query)

df = spark.sql(backfill_query)
df.display()


var_dict = {'a':1,'b':2}
assign_var_from_dict(var_dict)

expr1 = '"a" in globals()'
expr2 = '"a" in locals()'
print(expr1, eval(expr1))
print(expr2, eval(expr2))


path = 's3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/data/address_analysis_20230327_08353/inference_datetime_20240327_040755/address_analysis_data_address_processed.parquet'
df = spark.read.parquet(path)
df.display()


%run ./draft_callee


### Test sentence-transformers models ###
s3_model_dir = 's3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/model'
local_dir = '/tmp'
model_name_lt = [
                  'deberta-v2-large-japanese',
                  # 'bert-base-japanese-v3',
                ]

for model_name in model_name_lt:
  local_model_path = '{}/{}'.format(local_dir, model_name)
  s3_model_path = '{}/{}'.format(s3_model_dir, model_name)
  shell_cmd = 'aws s3 cp {} {} --recursive'.format(s3_model_path, local_model_path)
  run(shell_cmd, shell = True, capture_output = True)
  model = SentenceTransformer(local_model_path)
  print('Model {} has passed testing.'.format(model_name))
  if not os.path.isdir(local_model_path):
    shell_cmd = 'rm {}'.format(local_model_path)
  else:
    shell_cmd = 'rm {} --recursive'.format(local_model_path)
  run(shell_cmd, shell = True, capture_output = True)




path = 's3://ngap--customer-data-science--prod--us-east-1/airbot/uat/bot_detection_nlp/result/address_analysis_datetime_20240131_094948/base/community_detection_results/address_analysis_data_address_comm_deberta_v2_large_japanese_thrs_0.95_louvain_communities.parquet'
