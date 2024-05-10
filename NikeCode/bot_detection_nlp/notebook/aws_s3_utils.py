########################### This notebook contains functions for aws s3 operations ###########################
from boto3.s3.transfer import TransferConfig
from py4j.protocol import Py4JJavaError
from botocore.exceptions import ClientError
from subprocess import run
from s3pathlib import S3Path, not_

import boto3
import pickle as pkl
import os
import re
import json
import logging
import threading
import sys

logging.basicConfig(
    level=logging.INFO,
    format= "%(levelname)s:%(asctime)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)

class ProgressPercentage(object):
    def __init__(self, client, bucket, filename):
        self._filename = filename
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self._size = client.head_object(Bucket=bucket, Key=filename)['ContentLength']

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()


# Config for download/upload operation
mb_size_to_induce_multipart = 25
config = TransferConfig(
                        multipart_threshold = 1024 * mb_size_to_induce_multipart, # Size to induce mulitpart upload/download, in KB unit
                        max_concurrency = 10, # Num of threads to conduct multipart upload or download
                        multipart_chunksize = 1024 * mb_size_to_induce_multipart, # Size of each part for a multi-part transfer, in KB unit
                        use_threads=True # Whether to use threads or not
                       )


# Function that seperate s3 path into bucket and object key
def seperate_s3_path(s3_path):
    s3_bucket_pat = regex_catch_body_only(re.escape('s3://'), '(\w*[-]{1,2}\w*)*', '/')  # Regex for bucket name
    bucket = re.search(s3_bucket_pat, s3_path).group()  # bucket name
    obj_key = re.split(s3_bucket_pat + '/', s3_path)[-1]

    return bucket, obj_key


# Function that fetch aws secrets(e.g., key_id, access_key, session_token, etc.) from databricks local crendtial file
def fetch_aws_secrets(credential_path='/ae-compute-resources/aws_config/credentials',
                      aws_secret_pattern='aws[a-zA-Z_]+=',
                      token_type='boto3'):
    aws_credential_dict = {}

    if os.path.exists(credential_path):
        with open(credential_path) as f:
            credential_str = f.read()
    else:
        logging.warning('The credential path {} does NOT exists.'.format(credential_path))
        return {}

    aws_secrets_lt = list(map(lambda x: x.replace(' ', ''), credential_str.split('\n')))

    for aws_secrets_str in aws_secrets_lt:
        matched = re.search(aws_secret_pattern, aws_secrets_str)
        if matched is not None:
            matched_str = matched.group(0)
            value = re.sub(aws_secret_pattern, '', aws_secrets_str)

            if token_type == 'spark':  # Generate spark aws secrets
                if matched_str.startswith('aws_access_key_id'):
                    aws_credential_dict['key'] = value
                elif matched_str.startswith('aws_secret_access_key'):
                    aws_credential_dict['secret'] = value
                elif matched_str.startswith('aws_session_token'):
                    aws_credential_dict['token'] = value
                else:
                    pass
            elif token_type == 'boto3':  # Generate boto3 aws secrets
                matched_str = matched_str.rstrip('=')
                aws_credential_dict[matched_str] = value
            else:
                raise ValueError('"token_type" parameter should in the following: ["boto3", "spark"], %s.' % token_type)

    return aws_credential_dict


# Function that generate s3 client object with valid s3 token
def gen_valid_s3_client(bucket_name,
                        aws_credential_path_lt=[
                            '/ae-compute-resources/aws_config/credentials',
                            '/ae-compute-resources/aws_config/token',
                        ]):
    try:
        s3_client = boto3.client("s3")
        s3_client.list_objects_v2(Bucket=bucket_name, Prefix='')
        logging.debug('Default aws token is valid, use this token to build boto3 connection.')
        return s3_client
    except Exception as ce:
        logging.warning("Default aws token is not valid, try cluster's local aws token.")
        logging.error('Exception info for default aws token:\n"{}"'.format(ce))
        logging.info('Try fetching aws token from the following paths:"{}".'.format(aws_credential_path_lt))
        for aws_credential_path in aws_credential_path_lt:
            aws_credential_dict = fetch_aws_secrets(credential_path=aws_credential_path)
            logging.info('Fetched aws_credential_dict from path "{}" is \n"{}".'.format(aws_credential_path,
                                                                                        aws_credential_dict))
            if aws_credential_dict:
                s3_client = boto3.client("s3", **aws_credential_dict)
                try:
                    s3_client.list_objects_v2(Bucket=bucket_name, Prefix='')
                    return s3_client
                except Exception as ce:
                    logging.warning(
                        'Local aws token from the path "{}" is not valid as well. The detailed error information is:\n"{}".'.format(
                            aws_credential_path, ce))
            else:
                logging.warning(
                    'Fetched aws_credential_dict is not valid, try fetching aws_credential_dict from other paths.')
                continue

        raise NotImplementedError(
            'Failed to fetch valid aws token from the following local path:"{}".'.format(aws_credential_path_lt))


# Function that generate s3 client object with valid s3 token
def gen_valid_boto3_s3_session(
        bucket_name,
        aws_credential_path_lt=[
            '/ae-compute-resources/aws_config/credentials',
            '/ae-compute-resources/aws_config/token',
        ],
        Prefix='',
):
    try:
        boto3_session = boto3.session.Session()
        s3_client = boto3_session.client('s3')
        s3_client.list_objects_v2(Bucket=bucket_name, Prefix=Prefix)
        logging.debug('Default aws token is valid, use this token to build boto3 connection.')
        return boto3_session
    except Exception as ce:
        logging.warning("Default aws token is not valid, try cluster's local aws token.")
        logging.error('Exception info for default aws token:\n"{}"'.format(ce))
        logging.info('Try fetching aws token from the following paths:"{}".'.format(aws_credential_path_lt))
        for aws_credential_path in aws_credential_path_lt:
            aws_credential_dict = fetch_aws_secrets(credential_path=aws_credential_path)
            logging.info('Fetched aws_credential_dict from path "{}" is \n"{}".'.format(aws_credential_path,
                                                                                        aws_credential_dict))
            if aws_credential_dict:
                boto3_session = boto3.session.Session(**aws_credential_dict)
                s3_client = boto3.client("s3")
                try:
                    s3_client.list_objects_v2(Bucket=bucket_name, Prefix='')
                    return boto3_session
                except Exception as ce:
                    logging.warning(
                        'Local aws token from the path "{}" is not valid as well. The detailed error information is:\n"{}".'.format(
                            aws_credential_path, ce))
            else:
                logging.warning(
                    'Fetched aws_credential_dict is not valid, try fetching aws_credential_dict from other paths.')
                continue

        raise NotImplementedError(
            'Failed to fetch valid aws token from the following local path:"{}".'.format(aws_credential_path_lt))


def load_s3_json(bucket_name, obj_key, decoding='utf-8'):
    s3_client = gen_valid_s3_client(bucket_name)
    obj = s3_client.get_object(Bucket=bucket_name, Key=obj_key)
    js_obj = json.loads(obj.get('Body').read().decode(decoding))
    logging.info('Succeed in loading json file at "s3://{}/{}".'.format(bucket_name, obj_key))

    return js_obj


def load_s3_json_direct_path(s3_path, decoding='utf-8'):
    bucket_name, obj_key = seperate_s3_path(s3_path)
    return load_s3_json(bucket_name, obj_key, decoding)


def save_json_to_s3(bucket_name, obj_key, json_obj, encoding='utf-8'):
    s3_client = gen_valid_s3_client(bucket_name)
    logging.info(
        'Trying to write json file to "%s" using "%s" encoding.' % ('s3://' + bucket_name + '/' + obj_key, encoding))
    s3_client.put_object(Body=bytes(json.dumps(json_obj).encode(encoding)),
                         Bucket=bucket_name,
                         Key=obj_key)
    logging.info('Succeed in writing json file to "s3://{}/{}".'.format(bucket_name, obj_key))

    return


def save_s3_json_direct_path(s3_path, json_obj, encoding='utf-8'):
    bucket_name, obj_key = seperate_s3_path(s3_path)
    return save_json_to_s3(bucket_name, obj_key, json_obj, encoding)


def save_pkl_to_s3(bucket_name, obj_key, pkl_obj):
    s3_client = gen_valid_s3_client(bucket_name)
    logging.info('Trying to write pickle file to "s3://{}/{}".'.format(bucket_name, obj_key))
    s3_client.put_object(
        Body=pkl.dumps(pkl_obj),
        Bucket=bucket_name,
        Key=obj_key
    )
    logging.info('Succeed in writing pickle file to "s3://{}/{}".'.format(bucket_name, obj_key))

    return


def upload_to_s3_multipart(bucket_name, obj_key, to_upload_path,
                           config=config,
                           max_obj_size=1024 ** 2 * 100):
    """
      Upload the contents of a folder directory
      Args:
          bucket_name: the name of the s3 bucket
          obj_key: the path in s3 (i.e., 'Key' in boto3, can be either a file path or directory path). e.g., airbot/uat/
          to_upload_path: local path for file(s) to upload, can be either a file path or directory path.
    """

    s3_client = gen_valid_s3_client(bucket_name)

    # Strip the ending forward slashes for both to upload path and uploaded path
    to_upload_path = to_upload_path.rstrip('/')
    obj_key = obj_key.rstrip('/')

    # When both to upload path and uploaded path are file names
    if os.path.isfile(to_upload_path) and obj_key.find('.') != -1:
        if os.path.getsize(to_upload_path) > max_obj_size:
            logging.warning(
                'To upload file "{}" has a size of {:,.2f} Mib, which excceeds maximum size ({:,.2f} Mib) for boto3 uploading.'.format(
                    to_upload_path, os.path.getsize(to_upload_path) / 1024 ** 2, max_obj_size / 1024 ** 2))
            logging.info('Use aws command line to upload file instead.')
            shell_command = "aws s3 cp '{}' 's3://{}/{}'".format(to_upload_path, bucket_name, obj_key)
            run(shell_command, capture_output=True, shell=True)
        else:
            s3_client.upload_file(
                to_upload_path,
                bucket_name,
                obj_key,
                Config=config,
            )
        logging.info("Uploaded file '%s' to s3 path '%s/%s'." % (to_upload_path, 's3://' + bucket_name, obj_key))
        return

    # Determine whether input local path is a directory
    if os.path.isdir(to_upload_path):
        for file_name in os.listdir(to_upload_path):
            file_path = '%s/%s' % (to_upload_path, file_name)
            if not os.path.isdir(file_path):
                s3_file_path_without_bucket = '%s/%s' % (obj_key, file_name)
                s3_client.upload_file(
                    file_path,
                    bucket_name,
                    s3_file_path_without_bucket,
                    Config=config,
                )
                logging.info("Uploaded file '%s' to s3 path '%s/%s'." % (
                file_path, 's3://' + bucket_name, s3_file_path_without_bucket))

            else:
                # Don't forget to add a slash in the end of path otherwise the returned folder name may be the parent folder name if the path doesn't end with slash
                folder_name = os.path.basename(os.path.dirname(file_path + '/'))
                upload_to_s3_multipart(bucket_name, obj_key + '/%s' % folder_name, file_path, config=config)
    else:
        file_name = os.path.basename(to_upload_path)
        s3_file_path_without_bucket = '%s/%s' % (obj_key, file_name)

        if os.path.getsize(to_upload_path) > max_obj_size:
            logging.warning(
                'To upload file "{}" has a size of {:,.2f} Mib, which excceeds maximum size ({:,.2f} Mib) for boto3 uploading.'.format(
                    to_upload_path, os.path.getsize(to_upload_path) / 1024 ** 2, max_obj_size / 1024 ** 2))
            logging.info('Use aws command line to upload file instead.')
            shell_command = "aws s3 cp '{}' 's3://{}/{}'".format(to_upload_path, bucket_name,
                                                                 s3_file_path_without_bucket)
            run(shell_command, capture_output=True, shell=True)
        else:
            s3_client.upload_file(
                to_upload_path,
                bucket_name,
                s3_file_path_without_bucket,
                Config=config,
            )
        logging.info("Uploaded file '%s' to s3 path '%s/%s'." % (
        to_upload_path, 's3://' + bucket_name, s3_file_path_without_bucket))

    return


def download_from_s3_multipart(bucket_name, obj_key, target_path, config=config, max_obj_size=1024 ** 2 * 100):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket.
        obj_key: the path to download at s3, can be either a file path or a directory path.
        target_path: local path for file(s) to download, can be either a file path or a directory path.
    """
    s3_client = gen_valid_s3_client(bucket_name)
    paginator = s3_client.get_paginator('list_objects_v2')

    # Strip the ending forward slash for target path if it exists
    target_path = target_path.rstrip('/')
    # Fill the source path with trailing forward slash
    obj_key = '{}/'.format(obj_key.rstrip('/')) if obj_key.find('.') == -1 else obj_key

    pages = paginator.paginate(Bucket=bucket_name, Prefix=obj_key)

    logging.info("Download file(s) from '%s' to '%s'." % ('s3://' + bucket_name + '/' + obj_key, target_path))

    obj_name_list = []
    for page in pages:
        s3_dir_contents = page.get('Contents', [])
        if s3_dir_contents:
            obj_name_list.extend([(obj['Key'], obj['Size']) for obj in s3_dir_contents if obj['Size'] > 0])

    if not obj_name_list:
        # Stop if no feasible objects (object with size more than zero) to download
        logging.warning('No feasible object to download at s3 path "s3://{}/{}".'.format(bucket_name, obj_key))
        return False

    # Lambda function that determines the exact target object path based on the type 'target_path'
    target_path_func = lambda x, y: '{}/{}'.format(x, y)

    logging.info('The maximum object size for boto3 download is {:,.2f} Mib.'.format(max_obj_size / 1024 ** 2))

    for obj_name, obj_size in obj_name_list:
        # Assign the target path as downloaded object path if both source path and target path are file names
        if len(obj_name_list) == 1 and obj_name.rsplit(obj_key)[-1] == '' and target_path.find('.') != -1:
            target_obj_path = target_path
        else:
            # Feasbile target object path loacted in target path, choosing the one with maximum string length
            feasible_obj_paths = [os.path.basename(obj_name), obj_name.rsplit(obj_key)[-1]]
            obj_path = [x for x in feasible_obj_paths if len(x) == max(map(len, feasible_obj_paths))][
                0]  # The object path located in the target path

            target_obj_path = target_path_func(target_path, obj_path)

        logging.info('Download s3 file "s3://{}/{}" to "{}".'.format(bucket_name, obj_name, target_obj_path))

        if not os.path.exists(os.path.dirname(target_obj_path)):
            os.makedirs(os.path.dirname(target_obj_path))

        if obj_size <= max_obj_size:
            s3_client.download_file(bucket_name, obj_name, target_obj_path, Config=config,
                                    # Callback = ProgressPercentage(s3_client, bucket_name, obj_name)
                                    )
        else:
            logging.warning(
                'Object "{}" has size of {:,.2f} Mib, which excceds maximum size allowed to use boto3 download.'.format(
                    obj_name, obj_size / 1024 ** 2))
            logging.warning('Use aws command line to download object instead.')
            shell_command = "aws s3 cp 's3://{}/{}' '{}'".format(bucket_name, obj_name, target_obj_path)
            run(shell_command, capture_output=True, shell=True)

        logging.info(
            "File %s/%s has been downloaded to %s successfully." % ('s3://' + bucket_name, obj_name, target_obj_path))

    logging.info('Download finished')

    return True


# Function that copy entire directory to target path
def copy_dir_shell_command(local_dir, s3_dir, command_options='--recursive'):
    # Upload directory to s3 path
    shell_command = "aws s3 cp '{}' '{}' {}".format(local_dir, s3_dir, command_options)
    logging.info('The shell command to execute is "{}".'.format(shell_command))
    output = run(shell_command, capture_output=True, shell=True)
    # Error capture if it exists
    if output.stderr.decode('utf-8'):
        raise NotImplementedError('%s' % output.stderr.decode('utf-8'))
    else:
        logging.info('Command execution succeed. The output is %s.' % output.stdout.decode('utf-8'))

    return True


# Function that write file to s3 based on saving configuration dict and try different kinds of compression protocols
def save_spark_df_to_s3_from_dict(spark_df, spark_write_dict):
    try:
        spark_df.write.save(**spark_write_dict)
    except Py4JJavaError as pye:
        logging.warning('Py4JJavaError raised when writing spark dataframe to s3. Try other spark configuration.')
        data_format = spark_write_dict['format']
        if data_format == 'parquet':
            compress_codec_lt = ['gzip', 'lzo', 'lz4']
        elif data_format == 'csv':
            compress_codec_lt = ['snappy', 'bzip2', 'gzip', 'lz4', 'deflate']
        else:
            raise ValueError('To write data type %s is not supported yet.' % data_format)
        logging.info('The to-write data format is %s and will try the following compression codecs:%s.' % (
        data_format, compress_codec_lt))

        for idx, compress_codec in enumerate(compress_codec_lt):
            spark.conf.set("spark.sql.%s.compression.codec" % data_format, "%s" % compress_codec)
            logging.info('Try compressed codec %s for %s format file.' % (compress_codec, data_format))
            try:
                spark_df.write.save(**spark_write_dict)
            except Py4JJavaError:
                logging.warning('Compressed codec %s for %s format file failed.' % (compress_codec, data_format))
                if idx == len(compress_codec_lt) - 1:
                    logging.warning('Address similarities data failed to write to %s.' % spark_write_dict['path'])
                raise
    logging.info('Data has been saved to %s successfully.' % spark_write_dict['path'])


# Function that get all direct (no recursive) subfolders attributes in specific s3 path
def get_s3_subfolders(bucket_name, obj_key):
    s3_client = gen_valid_s3_client(bucket_name)
    parent_objects = s3_client.list_objects(Bucket=bucket_name,
                                            Prefix=obj_key + '/' if not obj_key.endswith('/') else obj_key,
                                            Delimiter='/')
    subfolder_lt = list(map(lambda x: x['Prefix'].rstrip('/'), parent_objects.get('CommonPrefixes')))

    return subfolder_lt


# Function that get attributes dict for specific s3 file (MUST BE a file otherwise error will be raised)
def get_s3_obj_attrdict(bucket_name, obj_key):
    s3_client = gen_valid_s3_client(bucket_name)
    return s3_client.get_object(Bucket=bucket_name, Key=obj_key)


# # Function that generate list of specific s3 objects' attribute that satisfy specific regex pattern
# def gen_target_pattern_s3_obj_list(s3_path, attr_name, file_regex_pat):
#   bucket_name, obj_key = seperate_s3_path(s3_path)
#   s3_client = gen_valid_s3_client(bucket_name)
#   paginator = s3_client.get_paginator('list_objects_v2')

#   obj_attr_list = []
#   pages = paginator.paginate(Bucket = bucket_name, Prefix = obj_key)
#   for page in pages:
#     s3_dir_contents = page.get('Contents', [])
#     if s3_dir_contents:
#       obj_attr_list.extend([obj[attr_name] for obj in s3_dir_contents if re.search(file_regex_pat, obj[attr_name]) != None])

#   return obj_attr_list

# Function that generate list of specific s3 objects' attribute that satisfy specific regex pattern
def gen_target_pattern_s3_obj_list(s3_path, attr_name, file_regex_pat,
                                   starting_token=None,
                                   obj_attr_list=[],
                                   ):
    if starting_token == 'Finished':
        return []
    else:
        latest_cont_token = starting_token

    bucket_name, obj_key = seperate_s3_path(s3_path)
    s3_client = gen_valid_s3_client(bucket_name)

    paginator = s3_client.get_paginator('list_objects_v2')
    PaginationConfig = {
        'StartingToken': starting_token,
    }

    pages = paginator.paginate(
        Bucket=bucket_name,
        Prefix=obj_key,
        PaginationConfig=PaginationConfig
    )

    pages = paginator.paginate(Bucket=bucket_name, Prefix=obj_key)
    try:
        for page in pages:
            latest_cont_token = page.get('NextContinuationToken', 'Finished')
            s3_dir_contents = page.get('Contents', [])
            if s3_dir_contents:
                obj_attrs_to_exentd = [obj[attr_name] for obj in s3_dir_contents if
                                       re.search(file_regex_pat, obj[attr_name]) != None]
                if obj_attrs_to_exentd:
                    obj_attr_list.extend(obj_attrs_to_exentd)

        logging.info('Iteration of paginator has been finished.')

    except ClientError as ce:
        logging.info(
            '"ClientError" occured when iterating the paginator. The lastest continuation token is "{}".'.format(
                latest_cont_token))
        logging.info('The "obj_attr_list" just before exception: "{}".'.format(obj_attr_list))
        logging.info('Try resuming the iteration of paginator from the lastest continuation token.')
        obj_attr_list_resumed = gen_target_pattern_s3_obj_list(s3_path, attr_name,
                                                               file_regex_pat,
                                                               starting_token=latest_cont_token,
                                                               obj_attr_list=obj_attr_list,
                                                               )
        obj_attr_list_resumed = list(set(obj_attr_list_resumed) - set(obj_attr_list))
        logging.info('{:,} satisfied object(s) has(have) been resumed.'.format(len(obj_attr_list)))
        obj_attr_list.extend(obj_attr_list_resumed)

    finally:
        return list(set(obj_attr_list))


# Function that delete s3 object
def delete_s3_object(bucket_name, obj_key):
    s3_client = gen_valid_s3_client(bucket_name)
    s3_client.delete_object(Bucket=bucket_name, Key=obj_key)

    return


# Function that check the existence of specific s3 object, return True if the object exists, otherwise return False
def is_s3_obj_exist(s3_path):
    shell_command = "aws s3 ls '{}'".format(s3_path)
    result = run(shell_command, capture_output=True, shell=True)
    if result.stdout.decode('utf8'):
        return True
    else:
        return False


# Function that generate list of first level files or sub-directories for specific s3 directory
def gen_fst_lvl_sub_lt(s3_path):
    shell_cmd = 'aws s3 ls {}/ | awk \'{{FS = " " ; print $2}}\''.format(s3_path)
    result = run(shell_cmd, capture_output=True, shell=True)
    reuslt_str = result.stdout.decode('utf8')
    subfolder_lt = list(
        filter(lambda x: len(x) > 0, map(lambda x: re.sub('/$', '', x), reuslt_str.split('\n')))) if reuslt_str else []
    return subfolder_lt


# Function that delete specific s3 path by aws cli
# This function will delte: the object if the s3 path refers to an object or all objects at the s3 path refers to a directory
def del_s3_path_awscli(s3_path):
    s3_path_obj = S3Path(s3_path)
    del_cmd = 'aws s3 rm {}'.format(s3_path_obj.uri)
    if not s3_path_obj.is_file():  # Delete all objects at the directory
        del_cmd += ' --recursive'
    logging.info('The deletion command to be executed is:\n"{}".'.format(del_cmd))
    run(del_cmd, capture_output=True, shell=True)

    return 'Finished Deletion'


# Function for conditional deletion of sub paths at specific s3 path
def gen_sub_s3path_list(s3_path, cond_func, include_folder=True, recursive=False, exclude=False,
                        convert_to_s3url=False):
    s3_path_obj = S3Path(s3_path)

    logging.info(
        'Start filtering sub S3 paths which {} the input condition at S3 path "{}", sub directories are {}.'.format(
            'SATISFY' if not exclude else 'DON\'T satisfy',
            s3_path,
            'included' if include_folder else 'NOT included'))

    if include_folder:
        s3_proxy = s3_path_obj.iterdir()
    else:
        s3_proxy = s3_path_obj.iter_objects(recursive=recursive)

    if not exclude:
        sub_s3path_obj_lt = s3_proxy.filter(cond_func).all()
    else:
        sub_s3path_obj_lt = s3_proxy.filter(not_(cond_func)).all()

    if convert_to_s3url:
        logging.info('All elements in result list will be converted to the S3 URL.')
        sub_s3path_obj_lt = [x.uri for x in sub_s3path_obj_lt]

    return sub_s3path_obj_lt


# Function that delete batch of input S3 pahts
def del_batch_s3paths(s3_path_iter, aws_cli_del=True):
    for s3_path in s3_path_iter:
        logging.info('Deleting s3 path "{}".'.format(s3_path))
        if aws_cli_del:
            del_s3_path_awscli(s3_path)
        else:
            S3Path(s3_path).delete(is_hard_delete=True)

    return


########################### This notebook contains functions for aws s3 operations ###########################