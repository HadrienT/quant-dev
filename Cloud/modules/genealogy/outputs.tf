output "cloud_run_url" {
  description = "URL of the Genealogy API on Cloud Run"
  value       = google_cloud_run_v2_service.genealogy_api.uri
}

output "gcs_bucket_name" {
  description = "GCS bucket storing tree data"
  value       = google_storage_bucket.genealogy_data.name
}
