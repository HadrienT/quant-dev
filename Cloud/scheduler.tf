resource "google_cloud_scheduler_job" "daily_sp500_job" {
  name        = "daily-sp500-data"
  description = "Job to run daily retrieval of S&P 500 data"

  schedule = "0 0 * * *" 
  time_zone = "UTC"

  http_target {
    uri = google_cloudfunctions_function.add_daily_function.https_trigger_url
    http_method = "POST"

    oidc_token {
      service_account_email = google_cloudfunctions_function.add_daily_function.service_account_email
    }
  }
}
