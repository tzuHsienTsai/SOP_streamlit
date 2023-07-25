import pickle
import logging
from google.cloud import storage


storage_client = storage.Client()
bucket_name = "ai_segmentation_embedding"
bucket = storage_client.bucket(bucket_name)


def get_blob_name(workflowId: str):
    return f"{workflowId}.pkl"


def exists_blob(source_blob_name):
    blob = bucket.blob(source_blob_name)
    return blob.exists()


def upload_blob(source_objects, destination_blob_name: str):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    blob = bucket.blob(destination_blob_name)
    contents = pickle.dumps(source_objects)
    blob.upload_from_string(contents)

    logging.info(
        f"Uploaded storage object {bucket_name}/{destination_blob_name}."
    )


def download_blob(source_blob_name: str):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The ID of your GCS object
    # source_blob_name = "storage-object-name"
    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.

    blob = bucket.blob(source_blob_name)
    content = blob.download_as_string()
    logging.info(f"Downloaded storage object {bucket_name}/{source_blob_name}")
    return pickle.loads(content)
