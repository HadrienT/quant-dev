output "data_function_url" {
  description = "URL of the data retrieval Cloud Function"
  value       = google_cloudfunctions_function.add_daily_function.https_trigger_url
}

output "bigquery_dataset" {
  description = "ID of the main BigQuery dataset"
  value       = google_bigquery_dataset.dataset.dataset_id
}
