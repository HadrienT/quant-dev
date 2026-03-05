#############################
# IAM: Allow public invocation of the underlying Cloud Run service (gen2)
#############################
resource "google_cloud_run_service_iam_member" "invoker" {
  project  = var.project_id
  location = var.region
  service  = google_cloudfunctions2_function.add_daily_function.service_config[0].service

  role   = "roles/run.invoker"
  member = "allUsers"
}
