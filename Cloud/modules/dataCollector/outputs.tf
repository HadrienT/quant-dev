output "data_function_url" {
  description = "URL of the data retrieval Cloud Function"
  value       = google_cloudfunctions2_function.add_daily_function.service_config[0].uri
}

output "bigquery_dataset" {
  description = "ID of the main BigQuery dataset"
  value       = google_bigquery_dataset.dataset.dataset_id
}
