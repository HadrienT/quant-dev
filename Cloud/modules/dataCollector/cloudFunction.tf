#############################
# Cloud Function (gen2) to retrieve S&P 500 data
#############################
resource "google_cloudfunctions2_function" "add_daily_function" {
  name     = "add_daily_sp500_data"
  location = var.region

  build_config {
    runtime     = "python311"
    entry_point = "main"
    source {
      storage_source {
        bucket = google_storage_bucket.function_code_bucket.name
        object = google_storage_bucket_object.function_code.name
      }
    }
  }

  service_config {
    available_memory = "512M"
    timeout_seconds  = 60
    ingress_settings = "ALLOW_ALL"
    environment_variables = {
      PROJECT_ID            = var.project_id
      DATASET_ID            = "financial_data"
      DATASET_LOCATION      = "EU"
      MAIN_TABLE_NAME       = "sp500_data"
      TEMP_TABLE_NAME       = "temp_sp500_data"
      TICKERS_PATH          = "./tickers.csv"
      CHUNK_SIZE            = "100"
      MAX_RETRIES           = "3"
      RETRY_BACKOFF_SECONDS = "1.5"
      MARKET_CALENDAR       = "NYSE"
      MARKET_TZ             = "America/New_York"
      LOG_LEVEL             = "INFO"
    }
  }
}
