#############################
# IAM: Allow public access to the Cloud Run Service
#############################
resource "google_cloud_run_service_iam_binding" "public_invoker" {
  location = google_cloud_run_v2_service.service.location
  project  = google_cloud_run_v2_service.service.project
  service  = google_cloud_run_v2_service.service.name
  role     = "roles/run.invoker"

  members = [
    "serviceAccount:${var.project_number}-compute@developer.gserviceaccount.com",
    "allUsers"
  ]
}
