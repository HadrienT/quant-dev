output "service_account_email" {
  description = "BigQuery reader service account email"
  value       = google_service_account.bq_reader.email
}

output "key_file_path" {
  description = "Path to the generated BigQuery reader key JSON"
  value       = local_file.bq_reader_key_file.filename
}
