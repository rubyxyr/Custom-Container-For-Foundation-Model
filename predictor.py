# -*- coding: utf-8 -*-
import bz2
import os
import json
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils
from google.oauth2 import service_account
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io
import skimage
import threading
from scipy.special import expit as sigmoid
from google.cloud import storage
from qdrant_client import models, QdrantClient
import pandas as pd
import requests
import time
import grpc
import uuid

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.compat.v1.enable_eager_execution()


gcs = storage.Client()
CLOUD_STORAGE_BUCKET = os.getenv('PRIVATE_BUCKET', '')
CLOUD_STORAGE_PUBLIC_BUCKET = os.getenv('PUBLIC_BUCKET', '')
GCS_DOMAIN = ''.join(['gs://', CLOUD_STORAGE_BUCKET, '/'])
bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)

PUBLIC_BUCKET_AUTH_PATH = f'gs://{CLOUD_STORAGE_BUCKET}/../xxx.json'
LOCAL_AUTH_TMP_PATH = '/.../xxxx.json'
PROJECT_NAME = 'xxxx'


def download_auth_file_to_local(client, remote_path, local_path):
    with open(local_path, 'wb') as fw:
        client.download_blob_to_file(remote_path, fw)


download_auth_file_to_local(gcs, PUBLIC_BUCKET_AUTH_PATH,
                            LOCAL_AUTH_TMP_PATH)
public_bucket_credentials = service_account.Credentials.from_service_account_file(
    LOCAL_AUTH_TMP_PATH)
public_storage_client = storage.Client(project=PROJECT_NAME,
                                       credentials=public_bucket_credentials)
adapter = requests.adapters.HTTPAdapter(pool_connections=128, pool_maxsize=128, max_retries=3, pool_block=True)
public_storage_client._http.mount("https://", adapter)
public_storage_client._http._auth_request.session.mount("https://", adapter)
public_bucket = public_storage_client.get_bucket(CLOUD_STORAGE_PUBLIC_BUCKET)


class CprPredictor(Predictor):

    def __init__(self):
        return

    def load(self, artifacts_uri: str):
        prediction_utils.download_model_artifacts(artifacts_uri)
        self._model = tf.saved_model.load(artifacts_uri)
        self.image_resize = int(os.getenv("IMAGE_INPUT_SIZE", 840))
        self.model_embed_dim = int(os.getenv("MODEL_EMBED_DIM", 768))

    def predict(self, instances):
        inputs = instances['instances'][0]
        json_filename = inputs['xxx']
        subcategory_query = inputs['xxx']
        class_query = inputs['xxx']
        image_name = inputs['xxx']
        project_name = inputs['xxx']
        text_query_length = inputs['xxx']
        qdrant_url = inputs['xxx']
        qdrant_api_key = inputs['xxx'] if inputs.get('xxx') else None
        qdrant_recreate_config = inputs['xxx'] if inputs.get('xxx') else None

        image = self.process_image(inputs['xxx'])
        h, w, _ = image.shape
        size = max(h, w)
        image_padded = np.pad(
            image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5)
        # Resize to model input size.
        input_image = skimage.transform.resize(
            image_padded,
            (self.image_resize, self.image_resize),
            anti_aliasing=True)

        raw_text_queries = inputs['xxx']
        query_inputs = tf.convert_to_tensor(inputs['xxx'])
        resized_query = query_inputs[None, ...]

        outputs = self._model({'images': input_image[None, ...],
                               'tokenized_queries': resized_query})
        predictions = tf.nest.map_structure(lambda x: x[0].numpy(), outputs)

        logits = predictions['pred_logits']
        logits = logits[..., :len(raw_text_queries)]  # Remove padding.
        if len(class_query) > 0:
            subcategory_logits = logits[:, :len(subcategory_query)]
            class_logits = logits[:, len(subcategory_query):]
            class_labels = np.argmax(class_logits, axis=-1)
            class_scores = sigmoid(np.max(class_logits, axis=-1))
            sorted_objectness_index = np.argsort(-subcategory_logits.flatten())
        else:
            subcategory_logits = logits
            sorted_object_type_index = np.argsort(-subcategory_logits, axis=1)[:, :1]
            sorted_logits = np.take_along_axis(subcategory_logits, sorted_object_type_index, axis=1)
            sorted_objectness_index = np.argsort(-sorted_logits.flatten())
            class_labels = None
            class_scores = None

        K = 200
        n = 1000

        class_embeddings = predictions['class_embeddings']
        post_class_embeddings = class_embeddings[sorted_objectness_index[:K]]
        pred_boxes = predictions['pred_boxes']
        post_boxes = pred_boxes[sorted_objectness_index[:K]]

        post_boxes_save = post_boxes * n
        post_boxes_save = post_boxes_save.astype(np.int16)

        pred_json_data = {'embeddings': post_class_embeddings.tolist(),
                          'boxes': post_boxes_save.tolist()}

        pred_data = {'embeddings': post_class_embeddings.tolist(),
                     'boxes': post_boxes.tolist()}

        subcategory_scores = sigmoid(np.max(subcategory_logits, axis=-1))
        subcategory_labels = np.argmax(subcategory_logits, axis=-1)

        post_subcategory_labels = subcategory_labels[sorted_objectness_index[:K]]
        post_subcategory_scores = subcategory_scores[sorted_objectness_index[:K]]

        post_class_labels = class_labels[sorted_objectness_index[:K]] if class_labels is not None else []
        post_class_scores = class_scores[sorted_objectness_index[:K]] if class_scores is not None else []

        uuid_list = [str(uuid.uuid4()) for _ in range(K)]
        pred_data['subcategory_scores'] = post_subcategory_scores.tolist()
        pred_data['subcategory_labels'] = post_subcategory_labels.tolist()
        pred_data['text_queries'] = raw_text_queries
        pred_data['point_uuid'] = uuid_list
        pred_data['class_scores'] = post_class_scores.tolist() if len(post_class_scores) > 0 else []
        pred_data['class_labels'] = post_class_labels.tolist() if len(post_class_labels) > 0 else []

        pred_json_data['subcategory_scores'] = (post_subcategory_scores * 10000).astype(np.int16).tolist()
        pred_json_data['subcategory_labels'] = post_subcategory_labels.tolist()
        pred_json_data['text_queries'] = raw_text_queries
        pred_json_data['point_uuid'] = uuid_list
        pred_json_data['class_scores'] = (post_class_scores * 10000).astype(np.int16).tolist() if len(post_class_scores) > 0 else []
        pred_json_data['class_labels'] = post_class_labels.tolist() if len(post_class_labels) > 0 else []

        return [pred_data,
                json_filename,
                image_name,
                project_name,
                text_query_length,
                qdrant_url,
                qdrant_api_key,
                class_query,
                subcategory_query,
                qdrant_recreate_config,
                pred_json_data]

    def postprocess(self, prediction_results):
        """Postprocesses the prediction results.

        Args:
            prediction_results (Any):
                Required. The prediction results.

        Returns:
            The postprocessed prediction results.
        """
        cached_json_filepath = prediction_results[1]
        _ = self.upload_image_to_gcs(cached_json_filepath, json.dumps(prediction_results[10]))
        vector_db_data_save_thread = threading.Thread(target=self.vector_db_save,
                                                      args=[prediction_results[0],
                                                            prediction_results[5],
                                                            prediction_results[6],
                                                            prediction_results[2],
                                                            prediction_results[3],
                                                            prediction_results[4],
                                                            prediction_results[7],
                                                            prediction_results[8],
                                                            prediction_results[9]])
        vector_db_data_save_thread.start()
        pred_data = {"predictions": [cached_json_filepath]}
        return pred_data

    def upload_image_to_gcs(self, gcs_path, pred_data_json):
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(bz2.compress(bytes(pred_data_json, 'utf-8')),
                                content_type='application/x-bzip2',
                                num_retries=2)
        image_url = f'gs://{CLOUD_STORAGE_BUCKET}/{gcs_path}'
        return image_url

    def read_gcs_image(self, filename):
        if filename.startswith('gs://'):
            tokens = filename.replace('gs://', '').split('/')
            bucket_name = tokens[0]
            filepath = '/'.join(tokens[1:])
            if bucket_name == CLOUD_STORAGE_PUBLIC_BUCKET:
                blob = public_bucket.blob(filepath)
            else:
                storage_client = storage.Client()
                image_bucket = storage_client.get_bucket(bucket_name)
                blob = image_bucket.get_blob(filepath)
            file_content = blob.download_as_string()
        else:
            raise ValueError(f'Invalid gcs path: {filename}')
        return file_content

    def process_image(self, filepath):
        img_bytes = self.read_gcs_image(filepath)
        image_pil = Image.open(io.BytesIO(img_bytes))
        rgb_im = image_pil.convert('RGB')
        image_uint8 = np.asarray(rgb_im)
        image = image_uint8.astype(np.float32) / 255.0
        return image

    def create_collection(self, qdrant, project_name, text_query_length, shard_number=2, replication_factor=2):
        retry_count = 0
        success = False
        while retry_count < 3:
            try:
                success = qdrant.recreate_collection(
                    collection_name=project_name,
                    vectors_config={
                        "image": models.VectorParams(size=self.model_embed_dim, distance=models.Distance.COSINE, on_disk=True),
                        "text": models.VectorParams(size=text_query_length, distance=models.Distance.COSINE, on_disk=True),
                    },
                    shard_number=shard_number,
                    replication_factor=replication_factor
                )
                _ = qdrant.create_payload_index(
                    collection_name=project_name,
                    field_name="image_name",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                break
            except Exception as e:
                print(f'[+] recreate error: {e}')
                retry_count += 1
                time.sleep(2)
        if not success:
            print(f'success: {success}, retry_count: {retry_count}')
            raise Exception(f'Collection {project_name} recreate failed.')
        return success

    def upload_records(self, qdrant, project_name, data, text_query_length, qdrant_recreate_config):
        if_recreated = False
        try:
            _ = qdrant.get_collection(project_name)
        except grpc.RpcError as grpc_e:
            if grpc_e.code() == grpc.StatusCode.NOT_FOUND:
                print(f'[+] collection {project_name} not exists. Recreate it.')
                if qdrant_recreate_config:
                    if_recreated = self.create_collection(qdrant, project_name, text_query_length,
                                                          **qdrant_recreate_config)
                else:
                    if_recreated = self.create_collection(qdrant, project_name, text_query_length)
            else:
                print(grpc_e)
                raise grpc.RpcError()
        qdrant.upload_records(
            collection_name=project_name,
            records=[
                models.Record(
                    id=record['point_uuid'],
                    vector={
                        "image": record['embeddings'],
                        # "text": record['label_embedding']
                    },
                    payload=record
                ) for _, record in enumerate(data)
            ],
            parallel=5
        )

    def preprocess_db_data(self, data, image_name, class_query, subcategory_query):
        json_result = data.copy()
        del json_result['text_queries']
        pd_data = pd.DataFrame(json_result)

        # save image_name in payload so that we can use filter to search records in particular image
        pd_data['image_name'] = image_name
        pd_data['subcategory_name'] = pd_data.subcategory_labels.apply(lambda x: subcategory_query[x])
        pd_data['class_name'] = pd_data.class_labels.apply(lambda x: class_query[x])

        list_results = list(pd_data.T.to_dict().values())
        return list_results

    def vector_db_save(self,
                       data,
                       qdrant_url,
                       qdrant_api_key,
                       image_name,
                       project_name,
                       text_query_length,
                       class_query,
                       subcategory_query,
                       qdrant_recreate_config):
        results = self.preprocess_db_data(data, image_name, class_query, subcategory_query)
        qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=True)
        self.upload_records(qdrant, str(project_name), results, text_query_length, qdrant_recreate_config)
