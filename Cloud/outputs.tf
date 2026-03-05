output "bq_reader_service_account" {
  description = "BigQuery reader service account email"
  value       = module.bq_access.service_account_email
}

output "bq_reader_key_path" {
  description = "Path to the generated BigQuery reader key JSON"
  value       = module.bq_access.key_file_path
}

output "genealogy_cloud_run_url" {
  description = "URL of the Genealogy API on Cloud Run"
  value       = var.enable_genealogy ? module.genealogy[0].cloud_run_url : null
}

output "genealogy_gcs_bucket" {
  description = "GCS bucket storing genealogy tree data"
  value       = var.enable_genealogy ? module.genealogy[0].gcs_bucket_name : null
}
