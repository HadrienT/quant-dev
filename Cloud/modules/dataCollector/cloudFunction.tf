#############################
# Cloud Function to retrieve S&P 500 data
#############################
resource "google_cloudfunctions_function" "add_daily_function" {
  name        = "add_daily_sp500_data"
  runtime     = "python311"
  region      = var.region
  entry_point = "main"

  source_archive_bucket = google_storage_bucket.function_code_bucket.name
  source_archive_object = google_storage_bucket_object.function_code.name

  available_memory_mb = 512
  timeout             = 60
  trigger_http        = true

  environment_variables = {
    PROJECT_ID = var.project_id
  }
}
