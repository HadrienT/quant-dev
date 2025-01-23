resource "google_storage_bucket" "function_code_bucket" {
  name                        = var.bucket_data_collector_code
  location                    = var.region
  uniform_bucket_level_access = true
}

resource "google_storage_bucket_object" "function_code" {
  name   = "function-${filemd5("${path.module}/scripts/dataColCF.zip")}.zip"
  bucket = google_storage_bucket.function_code_bucket.name
  source = "${path.module}/scripts/dataColCF.zip"
}


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
