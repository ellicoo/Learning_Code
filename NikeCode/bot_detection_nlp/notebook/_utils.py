notebook_name = dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().get().split('/')[-1]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:09:27 2022

@author: JHe54
"""

from collections import Callable
from torch import Tensor
from numpy import ndarray
from tqdm import trange
from boto3.s3.transfer import TransferConfig
from pyspark.mllib.linalg.distributed import CoordinateMatrix, IndexedRowMatrix
from numpy.linalg import norm
from pyspark.ml.linalg import Vectors
from pyspark.sql.session import SparkSession
from collections import Callable
from py4j.protocol import Py4JJavaError
from itertools import product
from botocore.exceptions import ClientError
from pyspark.sql.functions import arrays_zip, aggregate, sqrt, expr, explode, map_keys, broadcast
from subprocess import run

import logging
import pandas as pd
import numpy as np
import time
import boto3
import pickle as pkl
import os
import re
import json
import tqdm
import pyspark.sql.functions as F
import datetime


logging.basicConfig(
    level=logging.INFO,
    format= "%(levelname)s:%(asctime)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)


# Create a spark session object whose APP name is the notebook name
def create_spark_session():
    spark = SparkSession.builder.appName(notebook_name).enableHiveSupport().getOrCreate()
    return spark


# Functions that caculates text similarities using specific model
def text_similarity(similarity_func, text_1, text_2, embeddings_1, embeddings_2):
    """


    Parameters
    ----------
    similarity_func : Callable
        callable object used to caculate similarity
    text_1: iterable container objects, list, numpy.array, tuple, etc.
        array of string.
    text_2: iterable container objects, list, numpy.array, tuple, etc.
        array of string.
    embeddings_1 : array of number(e.g., float, int)
        embedded text_1 array
    embeddings_2 : array of number(e.g., float, int)
        embedded text_2 array


    Returns
    -------
    pandas.DataFrame object displaying pair similarities in descending order

    """

    if not isinstance(similarity_func, Callable):
        raise TypeError(
            '"similarity_func" shoud be a callable object, type %s found !'
            % type(similarity_func)
        )

    # Compute similarity between embeddings
    sim_result = similarity_func(embeddings_1, embeddings_2)
    # Convert similarity result to numpy ndarray
    if isinstance(sim_result, ndarray):
        pass
    elif isinstance(sim_result, Tensor):
        sim_result = sim_result.data.numpy()
    elif isinstance(sim_result, (pd.DataFrame, pd.Series)):
        sim_result = sim_result.to_numpy()
    else:
        sim_result = np.array(sim_result)

    if sim_result.ndim != 2:
        raise ValueError(
            "Similarity result sholud be numpy array with dimension 2, dimension %s found."
            % sim_result.ndim
        )

    # # Resul similarity pairs in descending order
    # pairs = []
    # for i in range(len(sim_result) - 1):
    #     for j in range(i + 1, len(sim_result)):
    #         pairs.append([[i, j], [text_1[i], text_1[j]], sim_result[i][j]])
    # pairs = np.array(sorted(pairs, key=lambda x: x[-1], reverse=True), dtype=object)
    # pair_df = pd.DataFrame(pairs, columns=["index", "address", "addr_similarity"])
    # pair_df.set_index("index", inplace=True)
    # pair_df["addr_similarity"] = pair_df["addr_similarity"].astype(float)

    # Don't get confused with index and column names (i.e., text_1 is index)
    sim_df = pd.DataFrame(sim_result, index=text_1, columns=text_2)

    return sim_df


# Functions that caculates text similarities using specific model
def gen_sim_pairs(similarity_func, text_1, text_2, embeddings_1, embeddings_2, n_decimal=4):
    """


    Parameters
    ----------
    similarity_func : Callable
        callable object used to caculate similarity
    text_1: iterable container objects, list, numpy.array, tuple, etc.
        array of string.
    text_2: iterable container objects, list, numpy.array, tuple, etc.
        array of string.
    embeddings_1 : array of number(e.g., float, int)
        embedded text_1 array
    embeddings_2 : array of number(e.g., float, int)
        embedded text_2 array


    Returns
    -------
    list of address pair similarity

    """

    if not isinstance(similarity_func, Callable):
        raise TypeError(
            '"similarity_func" shoud be a callable object, type %s found !'
            % type(similarity_func)
        )

    # Compute similarity between embeddings
    sim_result = similarity_func(embeddings_1, embeddings_2)
    # Convert similarity result to numpy ndarray
    if isinstance(sim_result, ndarray):
        sim_result_arr = sim_result
    elif isinstance(sim_result, Tensor):
        sim_result_arr = sim_result.data.numpy()
    elif isinstance(sim_result, (pd.DataFrame, pd.Series)):
        sim_result_arr = sim_result.to_numpy()
    else:
        sim_result_arr = np.array(sim_result)

    if sim_result_arr.ndim != 2:
        raise ValueError(
            "Similarity result sholud be numpy array with dimension 2, dimension %s found."
            % sim_result_arr.ndim
        )
    sim_pair_lt = []
    for x, y in zip(*np.tril_indices(sim_result_arr.shape[0], -1)):
        sim_pair_lt.append((text_1[x], text_2[y], round(float(sim_result_arr[x, y]),
                                                        n_decimal)))  # Similarity value must be converted to float64, otherwise errors will be raised when generating spark dataframe (can't infer schema for float32 numeric value)
    return sim_pair_lt


# Functions that generates text similarity pairs array and save it to s3 fragementally
def gen_sim_pairs_frag_and_save(similarity_func, text_1, text_2, embeddings_1, embeddings_2,
                                max_arr_len, sparkdf_schema, spark, addr_map_df, spark_write_dict,
                                addr_col_name, addr_id_list_colname, addr_column_mapping_dict,
                                n_decimal=4, self_cal=True, disable=False):
    """
address_colnames_lt = [addr_colname1, addr_colname2]

    Parameters
    ----------
    similarity_func : Callable
        callable object used to caculate similarity
    text_1: iterable container objects, list, numpy.array, tuple, etc.
        array of string.
    text_2: iterable container objects, list, numpy.array, tuple, etc.
        array of string.
    embeddings_1 : array of number(e.g., float, int)
        embedded text_1 array
    embeddings_2 : array of number(e.g., float, int)
        embedded text_2 array


    Returns
    -------
    list of address pair similarity

    """

    address_pair_colnames = addr_column_mapping_dict[addr_col_name]
    addr_id_list_colname_lt = addr_column_mapping_dict[addr_id_list_colname]

    if not isinstance(similarity_func, Callable):
        raise TypeError(
            '"similarity_func" shoud be a callable object, type %s found !'
            % type(similarity_func)
        )

    # Compute similarity between embeddings
    sim_result = similarity_func(embeddings_1, embeddings_2)
    # Convert similarity result to numpy ndarray
    if isinstance(sim_result, ndarray):
        sim_result_arr = sim_result
    elif isinstance(sim_result, Tensor):
        sim_result_arr = sim_result.data.numpy()
    elif isinstance(sim_result, (pd.DataFrame, pd.Series)):
        sim_result_arr = sim_result.to_numpy()
    else:
        sim_result_arr = np.array(sim_result)

    if sim_result_arr.ndim != 2:
        raise ValueError("Similarity result sholud be numpy array with dimension 2, dimension %s found."
                         % sim_result_arr.ndim)

    if self_cal:
        idx_iter = zip(*np.tril_indices(sim_result_arr.shape[0], -1))
        num_iter = int((sim_result_arr.shape[0] * (sim_result_arr.shape[0] - 1)) / 2)
    else:
        idx_iter = product(range(sim_result_arr.shape[0]), range(sim_result_arr.shape[1]))
        num_iter = sim_result_arr.shape[0] * sim_result_arr.shape[1]

    logging.info('There will be a number of {:,} values to calculate.'.format(num_iter))

    # Progress bar for similarity calculation and saving
    if disable:
        logging.warning('The progress bar will NOT be displayed.')
    progress = tqdm.tqdm(idx_iter, total=num_iter, position=0, dynamic_ncols=True, disable=disable)
    progress.set_description('Progress', refresh=False)

    sim_pair_lt = []
    partition_cnt = 0
    cal_cnt = 0
    save_cnt = 0
    for idx, (x, y) in enumerate(progress):
        # Similarity value must be converted to float64, otherwise errors will be raised when generating spark dataframe (can't infer schema for float32 numeric value)
        sim_value = round(float(sim_result_arr[x, y]), n_decimal)
        sim_pair_lt.append((text_1[x], text_2[y], sim_value))
        cal_cnt += 1

        # Information displayed on the posfix of progress bar
        info_dict = {'calculated': '{:,}'.format(cal_cnt), 'saved': '{:,}'.format(save_cnt),
                     'saved partition': '{:,}'.format(partition_cnt - 1)}
        # Force Refresh must be set to 'False' to aovid overwhelmed output
        progress.set_postfix(info_dict, refresh=False)

        if len(sim_pair_lt) < max_arr_len and idx < num_iter - 1:
            continue
        elif len(sim_pair_lt) == max_arr_len:
            progress.write(
                'Current similarity pairs array reaches the maximum length threshold: {:,}.'.format(max_arr_len))
            progress.write('Starts converting it spark data frame and will save it to s3 path.')
        else:  # Indicates the iteration reaches the end
            progress.write('Iteration has reached the end.'.format(max_arr_len))
            progress.write('The last similarity pairs sub-array has a length of {:,}.'.format(len(sim_pair_lt)))

        sub_arr_len = len(sim_pair_lt)
        addr_pair_sim_df = spark.createDataFrame(sim_pair_lt, sparkdf_schema)
        sim_pair_lt = []

        assert len(address_pair_colnames) == len(addr_id_list_colname_lt)
        for idx, addr_name_in_pair in enumerate(address_pair_colnames):
            addr_pair_sim_df = addr_pair_sim_df.join(addr_map_df, addr_pair_sim_df[addr_name_in_pair] == addr_map_df[
                addr_col_name]).withColumnRenamed(addr_id_list_colname, addr_id_list_colname_lt[idx]).drop(
                addr_col_name)

        # Save fragmented spark dataframe to s3
        spark_write_dict_sub = deepcopy(spark_write_dict)
        spark_write_dict_sub['path'] += '/part={}'.format(partition_cnt)
        save_spark_df_to_s3_from_dict(addr_pair_sim_df, spark_write_dict_sub)
        # addr_pair_sim_df.write.save(**spark_write_dict_sub)
        progress.write('Data has been saved to %s successfully.' % spark_write_dict_sub['path'])
        del addr_pair_sim_df
        save_cnt += sub_arr_len
        partition_cnt += 1

    # Calculate aggregated length of similarity pairs array
    logging.info(
        'The similarity pair array has an aggregated length of {:,} and has been saved to {} in {} partitions.'.format(
            save_cnt, spark_write_dict['path'], partition_cnt))

    return 'Process Finished'


# Function that saves multiple sheets into one excel file
def save_sheets_to_excel(
        fpath,
        sheet_df_pairs,
        mode="w",
        encoding="utf8",
        merge_cells=False,
        float_format="%.4f",
        header=True,
        engine="xlsxwriter",
):
    n_sheet = len(sheet_df_pairs)
    logging.info("%d sheets to write for file '%s'." % (n_sheet, fpath.split("/")[-1]))

    with pd.ExcelWriter(fpath, mode=mode) as excel_writer:
        general_paras = {
            "encoding": encoding,
            "merge_cells": merge_cells,
            "float_format": float_format,
            "header": header,
            "engine": engine,
        }
        for i, [sheet_name, sheet_df] in zip(trange(n_sheet), sheet_df_pairs):
            sheet_df.to_excel(excel_writer, **general_paras, sheet_name=sheet_name)
    logging.info("File %s has been saved successfully." % fpath)

    return True


# Function that runs shell command in python and return status and information with exception handling
def run_shell(shell_command, decoding='utf8', capture_output=True, shell=True):
    run_result = run(shell_command, capture_output=capture_output, shell=shell)

    if run_result.returncode == 0:
        print('Shell command runs successfully. The output is \n"%s".' % run_result.stdout.decode(decoding))
    else:
        print('Shell command runs failed. The return code is %d error information is \n"%s".' % (
        run_result.returncode, run_result.stderr.decode(decoding)))
        raise NotImplementedError

    # Function that convert input column data type to specific type base on target type iterables


def convert_col_type(col, data_type_list, to_replace, fill_missing, fill_val,
                     regex):  # Current version will raise NotImplementedError. if regex is set to 'True'
    if not hasattr(col, 'astype'):
        raise ValueError(
            'Must ensure input col has "astype" method to implement! Input column data type is: %s.' % type(col))

    if not data_type_list:
        raise ValueError('Please ensure input data_type_list is valid, actual input is: %s.' % data_type_list)

    logging.info('Processing columns "%s".' % col.name)
    if fill_missing:
        col = col.replace(to_replace=to_replace, value=fill_val,
                          regex=regex)  # Replace all string formatted blank values
        col = col.fillna(fill_val)  # Replace all numeric formatted blank values

    for data_type in data_type_list:
        try:
            col = col.astype(data_type)
            logging.info('Succeeded in converting data type of column to : %s.' % data_type)
            break
        except Exception as e:
            logging.info('Failed to convert data type of column to: %s.' % data_type)
            logging.info('Failure information: %s' % e)
            logging.info('The data values in column: %s' % list(col.unique()))
            if data_type == data_type_list[-1]:
                raise ValueError(
                    'Failed to convert column to the input following data types: %s. Please try another input.' % data_type_list)

    return col


# Function that search and generate input parameters for callable object from pre-defined config dictionary
def gen_para_dict(call_obj, config_dict):
    if not isinstance(call_obj, Callable):
        raise TypeError('Input object is not a callable object!')
    if not isinstance(config_dict, dict):
        raise TypeError('Input configuration parameter is not a dict object!')

    config_para_dict = {}  # Load configs from dict
    for para, value in config_dict.items():
        if para in inspect.signature(call_obj).parameters.keys():
            config_para_dict[para] = value

    return config_para_dict

#%run ./aws_s3_utils

### Functions that implement pyspark dataframe manipulation ###
### Function that generate null value rate statistical dictionary for columns in spark dataframe ###
def gen_null_rate_dict(df):
    nrows_data = df.count()
    null_rate_dict = {}
    for columns in df.columns:
        col_null_nrows = df.filter(col(columns).isNull()).count()
        null_rate_dict[columns] = '{:.4%}'.format(col_null_nrows / nrows_data)
    return null_rate_dict


# Function that conduct dot operation between two large-scale matrices (i.e., M*N multplies N*M generates a M*M matrix)
def matrix_dot_pyspark(matrix_1, matrix_2, spark):
    if matrix_1.shape[1] != matrix_2.shape[0]:
        raise ValueError("Input matrices shape doesn' match! Matrix 1 has %d columns while matrix 2 has %d rows!" % (
        matrix_1.shape[1], matrix_2.shape[0]))
    spark_context = spark.sparkContext
    matrix_tuple_iters_1 = map(lambda x: (x[0][0], x[0][1], x[1]), np.ndenumerate(matrix_1))
    matrix_tuple_iters_2 = map(lambda x: (x[0][0], x[0][1], x[1]), np.ndenumerate(matrix_2))

    matrix_rdds_1 = spark_context.parallelize(matrix_tuple_iters_1)
    matrix_rdds_2 = spark_context.parallelize(matrix_tuple_iters_2)

    matrix_block_matrix_1 = CoordinateMatrix(matrix_rdds_1).toBlockMatrix()
    matrix_block_matrix_2 = CoordinateMatrix(matrix_rdds_2).toBlockMatrix()
    result_matrix = matrix_block_matrix_1.multiply(matrix_block_matrix_2).toLocalMatrix()
    logging.info("Dense matrix of dot result generated, the deatil is %s." % result_matrix)
    logging.info("Converting dense matrix to numpy array.")
    result_np_matrix = result_matrix.toArray()

    return result_np_matrix


# Function that calculate math operation results between two large-scale matrices
def matrix_opt_pyspark(matrix_1, matrix_2, spark, operation='dot'):
    logging.info("Input matrix 1 has %d rows and %d columns." % matrix_1.shape)
    logging.info("Input matrix 2 has %d rows and %d columns." % matrix_2.shape)

    if operation in ['dot', 'cosine']:
        result_np_matrix = matrix_dot_pyspark(matrix_1, matrix_2, spark)
        logging.info("Dot calculation finished.")
        if operation == 'cosine':  # Needs to calculate L2 norm for each matrix
            logging.info("Starts calculating norm for each matrix.")
            matrix_norms_1 = norm(matrix_1, axis=1, ord=2)
            matrix_norms_1 = matrix_norms_1.reshape(matrix_norms_1.shape[0], 1)
            matrix_norms_2 = norm(matrix_2, axis=0, ord=2)
            matrix_norms_2 = matrix_norms_2.reshape(1, matrix_norms_2.shape[0])
            norm_matrix = np.multiply(matrix_norms_1, matrix_norms_2)
            result_np_matrix = result_np_matrix / norm_matrix
        #     elif: # TODO: add other operations
        #       pass
        else:
            raise ValueError('Operation name "%s" is invalid!' % operation)

    logging.info("Calculation finished.")
    return result_np_matrix


# Function that calculates cosine similarity between vectors in the matrix represented by row vectors using spark
# (i.e., each row is a vector and the number of rows represents the number of vectors)
def vector_cossim_spark(row_vector_matrix, spark):
    spark_context = spark.sparkContext
    logging.info("Input row vector matrix has %d rows and %d columns." % row_vector_matrix.shape)
    index_rows_iter = map(lambda x: (x[0], x[1]), enumerate(row_vector_matrix.T))
    index_rows_rdd = spark_context.parallelize(index_rows_iter)
    idx_row_mat = IndexedRowMatrix(index_rows_rdd)
    logging.info(
        "Accordingly indexed row matrix (transposed input matrix) generated, starts calculating cosine similarities between columns of this matrix.")

    # Calculate similarities between columns(i.e., similarities between embedding vectors in this case)
    cosine_coor_mat = idx_row_mat.columnSimilarities()  # Result is a coordinate matrix
    logging.info("Shape of cosine similarity result matrix:(%d,%d). Starts generating rdd for this coordinate matrix"
                 % (cosine_coor_mat.numRows(), cosine_coor_mat.numCols()))

    # Get rdd for caculated cosine result stored as MatrixEntries
    cosine_rdd = cosine_coor_mat.entries
    logging.info("RDD of cosine result generated.")

    return cosine_rdd


# Funciton that convert array into spark dataframe and save to s3 path based on saving configuration dict
def convert_to_sparkdf_and_save(array, sparkdf_schema, spark_write_dict):
    spark_df = spark.createDataFrame(array, sparkdf_schema)
    save_spark_df_to_s3_from_dict(spark_df, spark_write_dict)
    return


# Function that split long array into pieces, convert them to spark dataframes and save them to s3 paths accordingly
def arr_split_and_convert_to_sparkdf(array, sub_arr_len_max, spark, sparkdf_schema):
    array_len = len(array)
    n_arrays = array_len // sub_arr_len_max + 1
    spark_df_lt = []
    logging.info(
        'The input array has a length of %d and will be splitted into %d pieces with each piece no more than the length of %d.' % (
        array_len, n_arrays, sub_arr_len_max))
    for i in range(n_arrays):
        sub_array = array[i * sub_arr_len_max: (i + 1) * sub_arr_len_max]
        spark_df = spark.createDataFrame(sub_array, sparkdf_schema)
        spark_df_lt.append(spark_df)

    return spark_df_lt


# Function that extract specific type columns from spark dataframe
def gen_specific_type_columns(df, typeName):
    cond_func = lambda x: df.schema[x].dataType.typeName() == typeName
    specific_column_lt = [column for column in df.columns if cond_func(column) == True]
    return specific_column_lt


# Function that expand spark dataframe based on array type columns
def expand_df_on_array_columns(df, array_column_lt, keep_column_lt):
    df = df.withColumn("tmp", arrays_zip(*array_column_lt)).withColumn("tmp", explode("tmp")).select(*keep_column_lt,
                                                                                                     *[col('tmp.%s' % x)
                                                                                                       for x in
                                                                                                       array_column_lt])
    return df


# Function that expand map/dict column to multiple columns
def expand_map_column_spark(df, map_colname, keep_columns):
    keysDF = df.select(explode(map_keys(col(map_colname)))).distinct()
    keysList = keysDF.rdd.map(lambda x: x[0]).collect()
    keyCols = list(map(lambda x: col(map_colname).getItem(x).alias(str(x)), keysList))
    return df.select(*keep_columns, *keyCols)


# Function that rename multiple column names of spark dataframe based on mapping dictionary
def rename_multi_colnames(spark_df, mapping_dict):
    spark_df = spark_df.select([col(c).alias(mapping_dict.get(c, c)) for c in spark_df.columns])
    return spark_df


# Function that merge (join) two dataframes of different keys and remove the key of the data joined in the right in a 'data name - spark dataframe' mapping dictionary
def merge_and_remove_right_data(sparkdataframe_dict, left_data_name, right_data_name, join_cond, join_type):
    # Join the grouped data combination based on configuration
    left_df, right_df = sparkdataframe_dict[left_data_name], sparkdataframe_dict[right_data_name]
    joined_df = left_df.join(right_df, on=join_cond, how=join_type)
    # The joined dataframe will be assigned to the left data
    sparkdataframe_dict[left_data_name] = joined_df
    # The key of the right data will be removed since it has been joined with the left data
    sparkdataframe_dict.pop(right_data_name)
    return True


# Function that generate string expression of join clause based on join config dict
def gen_join_expr(join_config_dict, left_data='left', right_data='right', join_cond='cond', join_type='type'):
    left_data, right_data = join_config_dict[left_data], join_config_dict[right_data]
    join_cond, join_type = join_config_dict[join_cond], join_config_dict[join_type]
    join_expr = "{}.join({}, on = eval('{}'), how = '{}')".format(left_data, right_data, join_cond, join_type)
    return join_expr


# Function that calculates dot product between two array columns of spark dataframe
def dot_between_array_cols(col1, col2):
    return F.expr('aggregate(arrays_zip({0}, {1}), 0D, (acc, x) -> acc + (x.{0} * x.{1}))'.format(col1, col2))


# Function that calculates 1-norm for the array type column of spark dataframe
def norm_for_array_col(x):
    return F.expr('sqrt(aggregate({}, 0D, (acc, x) -> acc + (x * x)))'.format(x))


# Function that calculates cosine (similarity) value between two array columns of spark dataframe
def cosine_between_array_cols(col1, col2):
    from pyspark.sql.functions import arrays_zip, aggregate, sqrt, expr

    def dot_between_array_cols(col1, col2):
        return expr('aggregate(arrays_zip({0}, {1}), 0D, (acc, x) -> acc + (x.{0} * x.{1}))'.format(col1, col2))

    def norm_for_array_col(x):
        return expr('sqrt(aggregate({}, 0D, (acc, x) -> acc + (x * x)))'.format(x))

    return dot_between_array_cols(col1, col2) / (norm_for_array_col(col1) * norm_for_array_col(col2))


# Function that filter out not included values of the (non-array) column to be contained in 'array_contains' operation, boosting the performance of 'array_contains' operation
def filter_out_value_df(array_col_df, value_col_df, array_col, value_col):
    # Generate list that contains all values in the array column
    array_values_list = \
    array_col_df.select(explode(col(array_col)).alias('array_value')).select(['array_value']).distinct().toPandas()[
        'array_value'].values.tolist()
    logging.info(
        'Finished generating to filter value list, the list contains {:,} values.'.format(len(array_values_list)))
    filtered_value_col_df = value_col_df.join(
        broadcast(
            spark.createDataFrame(
                [(value,) for value in array_values_list],
                [value_col],
            )
        ),
        on=value_col,
    )
    # filtered_value_col_df = value_col_df.filter(col(value_col).isin(array_values_list))
    return filtered_value_col_df

### Functions that implement pyspark dataframe manipulation ###



### Functions that regarding graph manipulations ###

# Function that generate 2-dim tuple type input for networkX add_nodes_from calling
def gen_node_attr_tuple_from_dict(dictionary, node_name):
  node_value = dictionary[node_name]
  dictionary.pop(node_name)
  return (node_value, dictionary)

### Functions that regarding graph manipulations ###

### Functions that manipulate tensors ###
# Function that unsqueeze tensor to target dimension
def unsqueeze_ts_to_target(tensor_to_unsqueeze, target_dim):
    tensor_dim = tensor_to_unsqueeze.ndim
    diff_ndim = target_dim - tensor_dim
    if diff_ndim > 0:
        logging.info('Input tensor is %d dimension and will be unsqueezed to %d dimension.' % (tensor_dim, target_dim))
        for dim_to_insert in range(target_dim - 1, tensor_dim - 1, -1):
            tensor_squeezed = torch.unsqueeze(tensor_to_unsqueeze, dim_to_insert)
    else:
        logging.info('Input tensor already has target dimension: %d. No need for unsqueeze.' % target_dim)
        tensor_squeezed = tensor_to_unsqueeze

    return tensor_squeezed


### Functions that manipulages files on databricks machines ###
# Function that delete local file on databricks machine
def del_local_file(file_path):
  if os.path.isdir(file_path):
    shell_command = "rm --recursive '{}'".format(file_path)
  else:
    shell_command = "rm '{}'".format(file_path)
  run(shell_command, capture_output=True, shell=True)

  return


### Functions that solely for regex ###
# Function that for regex that matches prefix, body and suffix string but only catches body string
def regex_catch_body_only(prefix_regex, body_regex, suffix_regex):
  match_not_catch_prefix_regex = "(?<=(?:{}))".format(prefix_regex)
  match_not_catch_suffix_regex = "(?=(?:{}))".format(suffix_regex)
  regex = match_not_catch_prefix_regex + body_regex + match_not_catch_suffix_regex
  return regex


# Function that generate specified list of keys in dict which are sorted based on corresponding values
def get_sorted_keys(dc, max_key_num, reverse = True):
  return list(map(lambda x:x[0], sorted(list(dc.items()),key = lambda x : x[1], reverse = reverse)[:max_key_num]))

# Function that assign variables, including identify the representative value of string, from dictionary
def assign_var_from_dict(var_dict):
  for var, value in var_dict.items():
    globals()[var] = value
    logging.info('Succeeded in assigning variable "{}".'.format(var))
    if isinstance(value, str):
      logging.info('The value of variable "{}" is string "{}", check whether it\'s a string represntative of any other objects.'.format(var, value))
      try:
        globals()[var] = eval(globals()[var])
        logging.warning('Variable "{}" IS a string representative of object "{}".'.format(var, globals()[var]))
      except (NameError, SyntaxError):
        logging.warning('Variable "{}" is NOT a string representative of any other objects.'.format(var))
    else:
      pass


### Functions that manipulate databricks notebook widgets ###

# Function that assign variable from widget var list
def assign_var_from_widvar_lt(widget_var_lt):
  for widget_var in widget_var_lt:
    if not (widget_var in globals()) and not (widget_var in locals()):
      globals()[widget_var] = dbutils.widgets.get(widget_var)
      try: # Try to get the string represented value of a widget if it has
        globals()[widget_var] = eval(globals()[widget_var])
      except (NameError, SyntaxError):
        logging.warning('Widget var "{}" does NOT have string representative value.'.format(widget_var))
      finally:
        logging.info('Succeeded in assigning widget variable "{widget_var}".'.format(
        widget_var = widget_var))
    else:
        logging.warning('"{widget_var}" has already been assigned.'.format(widget_var = widget_var))


# ### Functions that interact with databricks notebook ###
# # Function that save all parameters defined in input notebook to json file at s3 path
# def save_notebook_paras_to_json(paras_dict, s3_json_path)

# Function that generate date list of specific date from which looking back n days
def gen_look_back_date_list(start_date, n_day_lookback, date_fmt = '%Y-%m-%d'):
  lookback_day_lt = []
  for i in range(n_day_lookback+1):
    lookback_day_lt.append(datetime.date.strftime(datetime.datetime.strptime(start_date,date_fmt) - datetime.timedelta(i), date_fmt))

  return lookback_day_lt


