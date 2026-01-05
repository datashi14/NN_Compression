import os
import json
import matplotlib.pyplot as plt
import datetime
import uuid
import yaml

class ArtifactManager:
    def __init__(self, run_id=None, base_dir='runs', upload_to_blob=False):
        if run_id is None:
            # Generate a unique run ID: YYYYMMDD-HHMMSS-UUID
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            short_uuid = str(uuid.uuid4())[:6]
            self.run_id = f"{timestamp}-{short_uuid}"
        else:
            self.run_id = run_id
            
        self.run_dir = os.path.join(base_dir, self.run_id)
        self.upload_to_blob = upload_to_blob
        self.metrics = {}
        
        # Ensure run directory exists
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, 'checkpoints'), exist_ok=True)
        
        print(f"Initialized Run: {self.run_id}")
        print(f"Artifacts will be saved to: {self.run_dir}")

    def save_config(self, config):
        """Saves config to yaml."""
        path = os.path.join(self.run_dir, 'config.yaml')
        with open(path, 'w') as f:
            yaml.dump(config, f)
            
    def log_metric(self, key, value, epoch=None):
        """Logs a metric. If epoch is provided, stores history."""
        if key not in self.metrics:
            self.metrics[key] = []
        
        entry = {'value': value, 'timestamp': datetime.datetime.now().isoformat()}
        if epoch is not None:
            entry['epoch'] = epoch
            
        self.metrics[key].append(entry)

    def save_metrics(self):
        """Saves current metrics to metrics.json."""
        path = os.path.join(self.run_dir, 'metrics.json')
        # Flatten for summary? Or keep raw? Keeping raw structure is safer.
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def save_plot(self, name, fig):
        """Saves a matplotlib figure."""
        path = os.path.join(self.run_dir, 'plots', f"{name}.png")
        fig.savefig(path)
        plt.close(fig) # Close to save memory

    def save_checkpoint(self, state_dict, name):
        import torch
        path = os.path.join(self.run_dir, 'checkpoints', f"{name}.pt")
        torch.save(state_dict, path)
        
    def upload_artifacts(self, connection_string=None):
        if not self.upload_to_blob:
            return
            
        # Check for GCS (Google Cloud Storage) first
        container_env = os.getenv('AZURE_BLOB_CONTAINER', '')
        if container_env.startswith('gs://'):
            self._upload_to_gcs(container_env)
            return

        if not connection_string:
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            
        if not connection_string:
            print("Warning: Skipping upload, no connection string or GCS bucket found.")
            return

        try:
            from azure.storage.blob import BlobServiceClient
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            container_name = container_env if container_env else "ticketsmith-runs"
            container_client = blob_service_client.get_container_client(container_name)
            
            if not container_client.exists():
                container_client.create_container()
                
            print(f"Uploading artifacts to Azure container '{container_name}'...")
            
            for root, dirs, files in os.walk(self.run_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, self.run_dir)
                    blob_name = f"{self.run_id}/{rel_path}"
                    
                    blob_client = container_client.get_blob_client(blob_name)
                    with open(local_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)
            print("Azure upload complete.")
            
        except Exception as e:
            print(f"Error uploading artifacts to Azure: {e}")

    def _upload_to_gcs(self, bucket_uri):
        try:
            from google.cloud import storage
            client = storage.Client()
            
            # Parse bucket name from gs://bucket-name
            bucket_name = bucket_uri.replace('gs://', '')
            bucket = client.bucket(bucket_name)
            
            print(f"Uploading artifacts to GCS bucket '{bucket_name}'...")
            
            for root, dirs, files in os.walk(self.run_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, self.run_dir)
                    blob_name = f"{self.run_id}/{rel_path}"
                    
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(local_path)
                    
            print("GCS upload complete.")
            
        except Exception as e:
            print(f"Error uploading artifacts to GCS: {e}")
