import logging
import boto3
from botocore.exceptions import ClientError

class DataS3(object):
  def __init__(self, config):
    self.s3_client = boto3.client('s3', aws_access_key_id=config['aws_access_key_id'],
                                  aws_secret_access_key=config['aws_secret_access_key'])
    self.s3_resource = boto3.resource('s3', aws_access_key_id=config['aws_access_key_id'],
                                  aws_secret_access_key=config['aws_secret_access_key'])
    self.bucket_name = config['s3_bucket_name']
    self.s3_folder = config['s3_folder']

  def get_max_weight_version(self):
    cont = True
    max_keys = 100
    max_version = -1
    try:
      while cont:
        if max_version == -1:
          obj = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix='%sweights/'%(self.s3_folder), MaxKeys=max_keys)
        else:
          obj = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix='%sweights/'%(self.s3_folder), \
                                               MaxKeys=max_keys, StartAfter='%sweights/%d'%(self.s3_folder, max_version))
        cont = obj['IsTruncated']
        if obj['KeyCount'] > 1:
          for obj_dict in obj['Contents']:
            object_name = obj_dict['Key']
            version = int(object_name.split('/')[-1])
            if verison > max_version:
              max_version=version
    except ClientError as e:
      logging.error(e)
    return max_version

  def get_max_data_version(self):
    cont = True
    max_keys = 100
    max_version = -1
    try:
      while cont:
        if max_version == -1:
          obj = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix='%sfeatures_train_'%(self.s3_folder), \
                                               MaxKeys=max_keys)
        else:
          obj = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix='%sfeatures_train_'%(self.s3_folder), \
                                               MaxKeys=max_keys, \
                                               StartAfter='%sfeatures_train_%d_part0.pklz'%(self.s3_folder, max_version))
        cont = obj['IsTruncated']
        for obj_dict in obj['Contents']:
          object_name = obj_dict['Key']
          feature_name = object_name.split('/')[-1]
          version = int(feature_name.split('_')[2])
          if version > max_version:
            max_version=version
    except ClientError as e:
      logging.error(e)
    return max_version

  def upload_file(self, file_name, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
      object_name = file_name

    # Upload the file
    try:
      obj = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=self.s3_folder + object_name, MaxKeys=1)
      if 'Contents' not in obj or len(obj['Contents']) == 0:
        response = self.s3_client.upload_file(file_name, self.bucket_name, self.s3_folder + object_name)
    except ClientError as e:
      logging.error(e)
      return False
    return True

  def download_file(self, object_name, local_path):
    try:
      self.s3_resource.Bucket(self.bucket_name).download_file(self.s3_folder + object_name, local_path)
    except ClientError as e:
      logging.exception('')
      if e.response['Error']['Code'] == "404":
        logging.error("The object does not exist.")
      else:
        raise

  def download_files(self, prefix, local_folder):
    file_names = []
    try:
      obj = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=self.s3_folder + prefix, MaxKeys=10)
      for obj_dict in obj['Contents']:
        object_name = obj_dict['Key']
        logging.info(object_name)
        file_name = object_name.split('/')[-1]
        self.s3_resource.Bucket(self.bucket_name).download_file(object_name, local_folder+file_name)
        file_names.append(local_folder+file_name)
    except ClientError as e:
      logging.error(e)
    return file_names
