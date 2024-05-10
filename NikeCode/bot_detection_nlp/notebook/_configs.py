from functools import partial
from datetime import datetime, timedelta
from subprocess import run

import pytz
import os

### Configurations for global paths ###
# AWS S3 bucket name
bucket = "ngap--customer-data-science--prod--us-east-1"
folder = "airbot/uat/bot_detection_nlp"
# Local directory
local_dir = "/tmp"

# Folders without s3 prefix
data_folder = "%s/data" % folder
result_folder = "%s/result" % folder
model_folder = "%s/model" % folder
config_folder = "%s/config" % folder

# Folders with s3 prefix
s3_data_folder = "s3://%s/%s/data" % (bucket, folder)
s3_result_folder = "s3://%s/%s/result" % (bucket, folder)
s3_model_folder = "s3://%s/%s/model" % (bucket, folder)
s3_config_folder = "s3://%s/%s/config" % (bucket, folder)

### Configurations for aws s3 ###
s3_boto3_token_path = '%s/aws_token_for_boto3.json' % s3_config_folder
token_refresh_interval = 600

# Unified date&time configurations
time_zone_str = "Asia/Shanghai"  # Time zone of date time, using WHQ time zone
time_zone = pytz.timezone(time_zone_str)
current_date = datetime.now(time_zone).date()  # current date in specific time zone