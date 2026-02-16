#############################
# Cloud Scheduler to trigger the daily data retrieval function
#############################
resource "google_cloud_scheduler_job" "daily_sp500_job" {
  name        = "daily-sp500-data"
  description = "Job to run daily retrieval of S&P 500 data"
  schedule    = "0 22 * * 1-5"
  time_zone   = "Europe/Paris"

  http_target {
    uri         = google_cloudfunctions2_function.add_daily_function.service_config[0].uri
    http_method = "POST"

    oidc_token {
      service_account_email = google_cloudfunctions2_function.add_daily_function.service_config[0].service_account_email
    }
  }
}
